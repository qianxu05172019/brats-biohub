# brats-biohub

BraTS 2021 Brain Tumor Segmentation Pipeline built with MONAI.

## Data

BraTS 2021 Training Data located at:
```
/workspace/DataChallenge/data/BraTS2021_Training_Data/
```

Each subject contains 4 MRI modalities (FLAIR, T1, T1ce, T2) and a segmentation mask, all in NIfTI format.

## Project Structure

```
brats-biohub/
  README.md
  pyproject.toml
  configs/
    train_3d_unet.yaml        # Training configuration for 3D U-Net
    infer_sliding_window.yaml  # Inference with sliding window
    preprocess.yaml            # Preprocessing pipeline settings
```

## Setup

```bash
pip install monai nibabel SimpleITK scikit-learn scikit-survival shap xgboost seaborn plotly scipy kaggle tqdm wandb pandas numpy pyyaml
```
