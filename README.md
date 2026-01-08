# Leaf Classifier (Smooth vs Serrated)

Binary image classification project to distinguish between two leaf types:

- **dente** (serrated)
- **lisse** (smooth)

This repo contains both:

- a **training pipeline** (PyTorch) for experimenting with CNN / ConvNeXt
- a **Streamlit app** to run inference on uploaded images using a saved model file (`custom_model.pth`).

## Demo (Streamlit)

The Streamlit UI lets you upload a `.jpg/.jpeg/.png` leaf image and returns:

- the predicted class (`dente` or `lisse`)
- confidence scores for both classes

### Run locally

From the repository root:

```bash
pip install -r streamlit/requirements.txt
streamlit run streamlit/app.py
```

Then open `http://localhost:8501`.

> Note: `streamlit/app.py` loads `custom_model.pth` using a **relative path**. Running the command from the repo root ensures the model file is found.

## Project structure

- `streamlit/app.py` — Streamlit inference app
- `streamlit/requirements.txt` — runtime dependencies for the app
- `streamlit/DEPLOYEMENT.md` — deployment notes (Streamlit Cloud)
- `custom_model.pth` — trained model used by the Streamlit app
- `dataset/feuilles_plantes/` — dataset folder (class subfolders)
- `test_images/` — optional images for quick manual testing
- `leaf-classfication.ipynb` — notebook for a compact CNN pipeline
- `convnext-leaf-classifier.py` / `convnext-leaf-classifier.ipynb` — ConvNeXt Tiny training + evaluation pipeline

## Dataset

This project expects an ImageFolder-style layout. In this repo the dataset is organized as:

```text
dataset/
  feuilles_plantes/
    dente/
    lisse/
```

Each class folder contains leaf images (e.g., `.jpg`, `.png`).

## Training

There are two training entry points:

### 1) Notebook CNN pipeline

Use `leaf-classfication.ipynb` for a compact CNN with training + evaluation, including metrics and plots.

### 2) ConvNeXt Tiny pipeline

`convnext-leaf-classifier.py` trains a pretrained ConvNeXt Tiny model (ImageNet weights) and reports metrics like accuracy, precision/recall/F1, AUC-ROC, confusion matrix, etc.

To run it, open and execute the notebook/script after updating the dataset path.

> Important: `convnext-leaf-classifier.py` currently uses an **absolute** `DATA_DIR` path. Change it to your local dataset path before running.

## Deployment (Streamlit Cloud)

See `streamlit/DEPLOYEMENT.md` for step-by-step instructions.

Key requirement: the model file `custom_model.pth` must be available to the app (committed to the repo, Git LFS, or downloaded at runtime).

## Troubleshooting

- **`FileNotFoundError: custom_model.pth`**
  - Run Streamlit from the repo root (`streamlit run streamlit/app.py`), or place the model next to `streamlit/app.py` and update the path.
- **Slow inference on CPU**
  - Install a CUDA-enabled PyTorch build and run on a machine with an NVIDIA GPU.
- **Class name mismatch**
  - Update `CLASS_NAMES` in `streamlit/app.py` to match your folder names / training labels.

## Tech stack

- Python
- PyTorch + TorchVision
- Streamlit
- Pillow, NumPy

## License

This project is licensed under the **MIT License**. See `LICENSE`.
