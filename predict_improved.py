# predict_improved.py
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

MODEL_PATH = Path("models/mnist_cnn.keras")

def load_image(path):
    img = Image.open(path).convert("L")  # grayscale
    return img

def enhance_and_binarize(pil_img):
    # Convert to numpy array and normalize size
    arr = np.array(pil_img)
    
    # Convert to numpy array - size will be handled by contour detection
    
    # Normalize contrast
    p5, p95 = np.percentile(arr, (5, 95))
    normalized = np.clip((arr - p5) * 255.0 / (p95 - p5), 0, 255).astype('uint8')
    
    # Try multiple preprocessing approaches and combine results
    results = []
    
    # Approach 1: Direct Otsu
    _, otsu = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(otsu)
    
    # Approach 2: CLAHE + Adaptive
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(normalized)
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    results.append(adaptive)
    
    # Approach 3: Blur + Threshold
    blurred = cv2.GaussianBlur(normalized, (3,3), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    results.append(thresh)
    
    # Combine results (take the most aggressive result)
    binary = np.minimum.reduce(results)
    
    # If background is bright, invert
    if np.mean(binary) > 127:
        binary = 255 - binary
        
    # Clean up noise with small morphological operations
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def largest_contour_bbox(binary):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
        
    H, W = binary.shape
    min_area = 150  # minimum area for a valid digit
    max_area = 5000  # maximum area for a valid digit
    
    valid_cnts = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # Skip if contour is too large (background/margins)
        if w > 0.8 * W or h > 0.8 * H:
            continue
            
        # Skip tiny or huge areas
        if area < min_area or area > max_area:
            continue
            
        # More strict fill ratio check for digit-like shapes
        fill_ratio = cv2.contourArea(c) / (w * h)
        if fill_ratio > 0.7 or fill_ratio < 0.2:  # digits should have reasonable density
            continue
            
        # Check aspect ratio (height/width) for digit-like shapes
        aspect = h / w
        if aspect < 0.8 or aspect > 2.5:  # relaxed to allow slightly wider digits like 5
            continue
            
        # Calculate white pixel ratio within the bounding box
        roi = binary[y:y+h, x:x+w]
        white_ratio = np.sum(roi) / (255.0 * w * h)
            
        valid_cnts.append((c, area, fill_ratio, white_ratio))
    
    if not valid_cnts:
        return None
    
    # Sort by a score that prefers:
    # - Medium-sized areas (not too small, not too large)
    # - Moderate fill ratios (around 0.4-0.6 for digit-like shapes)
    # - White pixel ratios typical of digits
    def contour_score(item):
        _, area, fill, white = item
        size_score = -abs(area - 1000)  # prefer areas around 1000px
        fill_score = -abs(fill - 0.5)  # prefer moderate fill ratios
        white_score = -abs(white - 0.4)  # prefer moderate white ratios
        return size_score + fill_score * 300 + white_score * 200
        
    # Pick best scoring contour
    best_cnt, _, _, _ = max(valid_cnts, key=contour_score)
    x, y, w, h = cv2.boundingRect(best_cnt)
    return x, y, w, h

def crop_and_center(binary, bbox, pad=20):
    h, w = binary.shape
    x, y, bw, bh = bbox
    
    # Add padding but stay inside image bounds
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    roi = binary[y0:y1, x0:x1]
    
    # Ensure digit is white on black background like MNIST
    if roi.mean() < 127:
        roi = 255 - roi
        
    # Clean up the ROI
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    
    return roi

def to_mnist(roi):
    h, w = roi.shape
    if h == 0 or w == 0:
        return np.zeros((28,28), dtype='float32')
    
    # First resize to 20x20 target (smaller than final to ensure margins)
    target_size = 20
    scale = target_size / max(h, w)
    nh = max(1, int(h * scale))
    nw = max(1, int(w * scale))
    
    # Use INTER_AREA for downscaling, INTER_LINEAR for upscaling
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    roi_resized = cv2.resize(roi, (nw, nh), interpolation=interp)
    
    # Create 28x28 canvas with black background
    canvas = np.zeros((28, 28), dtype='uint8')
    
    # Center the digit geometrically first
    y_off = (28 - nh) // 2
    x_off = (28 - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = roi_resized
    
    # Convert to float and normalize
    canvas_f = canvas.astype('float32') / 255.0
    
    # Find center of mass for fine-tuning position
    coords = np.nonzero(canvas_f)
    if len(coords[0]) > 0:
        cy, cx = coords[0].mean(), coords[1].mean()
        
        # Only shift if center of mass is significantly off
        if abs(cy - 14) > 1 or abs(cx - 14) > 1:
            shift_x = int(np.round(14 - cx))
            shift_y = int(np.round(14 - cy))
            M = np.float32([[1, 0, shift_x],[0, 1, shift_y]])
            canvas_f = cv2.warpAffine(canvas_f, M, (28,28), borderValue=0)
    
    # Use adaptive thresholding for final preprocessing
    canvas_uint = (canvas_f * 255).astype('uint8')
    canvas_thresh = cv2.adaptiveThreshold(
        canvas_uint, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 7, 2)
    canvas_f = canvas_thresh.astype('float32') / 255.0
    
    # Ensure we keep enough detail for digits like 5
    white_ratio = np.sum(canvas_f) / canvas_f.size
    if white_ratio < 0.03:  # too few white pixels
        canvas_thresh = cv2.adaptiveThreshold(
            canvas_uint, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 5, 1)  # more aggressive params
        canvas_f = canvas_thresh.astype('float32') / 255.0
    
    return canvas_f.astype('float32')

def save_debug_images(original_path, binary, mnist_img):
    p = Path(original_path)
    debug_dir = p.with_name(p.stem + "_debug")
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / "binary.png"), binary)
    mn = (mnist_img * 255).astype('uint8')
    cv2.imwrite(str(debug_dir / "mnist_input.png"), mn)
    print("Saved debug images to:", debug_dir)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_improved.py <path_to_image>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print("File not found:", img_path)
        sys.exit(1)

    # load model
    model = tf.keras.models.load_model(MODEL_PATH)

    pil = load_image(img_path)
    binary = enhance_and_binarize(pil)

    bbox = largest_contour_bbox(binary)
    if bbox is None:
        print("No contours found - try a clearer photo or crop closer.")
        sys.exit(1)

    roi = crop_and_center(binary, bbox, pad=12)
    mnist_img = to_mnist(roi)  # 28x28 float32 in 0..1

    # Save debug visuals
    save_debug_images(img_path, binary, mnist_img)

    x = mnist_img[None, ..., None]  # shape (1,28,28,1)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    print("Predicted digit:", pred)
    print("Probabilities:", np.round(probs, 3))

if __name__ == "__main__":
    main()