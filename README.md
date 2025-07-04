# Age Detection Using Pretrained CNN on UTKFace Dataset

## Overview

This project demonstrates the fine-tuning of a pretrained Convolutional Neural Network (CNN) for age prediction using facial images. The model is trained on the UTKFace dataset and evaluated based on accuracy, confusion matrix, precision, and recall.

## Dataset

- **Name**: UTKFace Dataset
- **Dataset Link**: 'https://www.kaggle.com/datasets/jangedoo/utkface-new'
- **Structure**: Filenames contain the age of the individual (e.g., '23_0_1_20170109150557335.jpg' → age = 23)
- **Filtering**: Ages between 1 and 119 are retained
- **Splitting**:
  - Training: 72.25%
  - Validation: 12.75%
  - Testing: 15%

## Data Preprocessing

- **Augmentation (Train Set)**:
  - Resize to 256x256, random crop to 224x224
  - Horizontal flip, brightness/contrast adjustment
  - Coarse dropout, normalization
- **Validation/Test Set**:
  - Resize to 224x224
  - Normalization using ImageNet stats

## Model

- **Architecture**: ConvNeXt-Base
- **Library**: 'timm' (pretrained weights)
- **Type**: CNN-based regression model
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau
- **Device**: CUDA (GPU) if available

## Training

- **Epochs**: 26
- **Batch Size**: 32
- **Mixed Precision Training**: Enabled with 'autocast' and 'GradScaler' for faster computation
- **Validation Monitoring**: Tracks loss to save the best-performing model

## Evaluation Metrics

- **Final Accuracy**: 70.28%
- **Confusion Matrix**: Computed using 'sklearn.metrics.confusion_matrix'
- **Classification Report**: Includes precision, recall, and F1-score using 'classification_report'

## Model Access

Use the link below:

[Download Model Weights](https://drive.google.com/file/d/19At49Go9z0GuKiqBZNwVduOh8s2ZmsmQ/view?usp=sharing)

## Setup Instructions

1. Clone the repository or download the folder.
2. Install the dependencies:
pip install -r requirements.txt

3. Open age-detection-model.ipynb in Jupyter Notebook or Colab.

4. Run the notebook to:
  - Load and preprocess the dataset
  - Train and validate the model
  - Evaluate metrics
  - Save model weights

## Requirements
The Python libraries used in this project are listed in requirements.txt.

## Output
All results, including accuracy and evaluation metrics, are printed and plotted in the final cells of the notebook.

## Notes
- This implementation does not include a GUI.
- The code and results follow the specifications for fine-tuning a CNN model on the UTKFace dataset.
