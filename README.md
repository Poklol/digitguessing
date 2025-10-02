# MNIST Digit Recognition

A Python application for recognizing handwritten digits using a CNN model trained on the MNIST dataset.

## Features
- Real-time digit recognition from images
- Enhanced preprocessing pipeline for better accuracy
- Support for various image formats and sizes
- Debug visualization of preprocessing steps

## Files
- `predict.py`: Basic prediction script
- `predict_improved.py`: Enhanced version with better preprocessing
- `train.py`: Script for training the CNN model
- `mnist_debug.ipynb`: Jupyter notebook for debugging and visualization

## Setup
1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install "numpy==1.23.5" tensorflow==2.16.1 pillow opencv-python matplotlib
```

## Usage
```bash
python predict_improved.py path/to/your/image.jpg
```

The script will:
1. Preprocess the image
2. Detect and extract the digit
3. Make a prediction
4. Save debug visualizations