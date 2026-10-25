# Transfer Learning with ResNet18

## Overview
This project uses transfer learning with a pretrained ResNet18 model to classify images of ants and bees using PyTorch.

## Dataset
The Hymenoptera dataset contains training and validation images of two classes:
- Ants
- Bees

## Method
- Pretrained ResNet18 using ImageNet weights
- Frozen pretrained parameters
- Replaced the original classification layer with a two-class classifier
- CrossEntropyLoss
- Adam optimizer
- Data augmentation for training images
- Validation-based model checkpointing

## Results
- Best validation accuracy: 94.12%
- Training images: 244
- Validation images: 153
- Device: CPU

## Generated Outputs
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrix
- Sample predictions

## Technologies
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn

## Run
Install the dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python transfer_learning.py
```

The script downloads the dataset and pretrained weights when needed.
