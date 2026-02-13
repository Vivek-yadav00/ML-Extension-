# ML Model Complexity Estimator

A browser extension that analyzes Machine Learning code on GitHub to estimate time and memory complexity.

## Features
- 🕵️ **Detects Frameworks**: Scikit-learn, TensorFlow, PyTorch.
- ⚡ **Estimates Complexity**: Provides Big-O notation for training/inference.
- 📊 **Dataset Assumptions**: Warns about scalability issues.

## Setup

### Backend
1.  Navigate to `backend/`.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Run server: `uvicorn app.main:app --reload`.

### Extension
1.  Open Chrome/Edge.
2.  Go to `chrome://extensions`.
3.  Enable **Developer Mode**.
4.  Click **Load Unpacked**.
5.  Select the `extension/` folder.

## Usage
1.  Go to a GitHub file (e.g., [this sklearn example](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/tests/test_svm.py)).
2.  Click the extension icon.
3.  Click "Analyze".
