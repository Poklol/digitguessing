import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf

MODEL_PATH = Path("models/mnist_cnn.keras")
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(path: Path):
    img = Image.open(path).convert("L")  # grayscale
    # Denoise a bit (helps with phone pics)
    img = img.filter(ImageFilter.MedianFilter(size=3))

    # If background is bright, invert so digit is white-on-black (MNIST style)
    if np.mean(img) > 127:
        img = ImageOps.invert(img)

    # Adaptive threshold to isolate the digit
    arr = np.array(img)
    thr = max(30, int(arr.mean() * 0.7))
    mask = arr > thr

    # If nothing detected, fall back to center crop
    if mask.sum() == 0:
        img = ImageOps.fit(img, (28, 28), method=Image.Resampling.LANCZOS)
        arr = np.array(img).astype("float32") / 255.0
        return arr[None, ..., None]

    # Crop to bounding box of the digit
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    digit = img.crop((x0, y0, x1 + 1, y1 + 1))

    # Keep aspect ratio: fit into 20x20 and pad to 28x28 (like classic MNIST prep)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    # center the digit
    x_off = (28 - digit.size[0]) // 2
    y_off = (28 - digit.size[1]) // 2
    canvas.paste(digit, (x_off, y_off))

    arr = np.array(canvas).astype("float32") / 255.0
    return arr[None, ..., None]  # (1,28,28,1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_digit_image>")
        sys.exit(1)
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"File not found: {img_path}")
        sys.exit(1)

    x = preprocess(img_path)
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    print(f"Predicted digit: {pred}")
    print("Probabilities:", np.round(probs, 3))

if __name__ == "__main__":
    main()