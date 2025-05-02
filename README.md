# ComputerVision
Medical Image Classification with Enhanced ResNet50
Project Description

This repository contains an enhanced ResNet50 model with Squeeze-and-Excitation (SE) blocks for medical image classification and lesion counting. The model simultaneously performs classification (4 severity classes) and lesion counting tasks.

#File Structure
medical-image-classification/

├── data/                   
│   ├── JPEGImages/          # Medical images
│   ├── NNEW_trainval_[0-4].txt  # Training/validation splits
│   └── NNEW_test_[0-4].txt  # Test splits
│
├── model/
│   ├── resnet50.py          # Modified ResNet50 with SE blocks
│   └── __init__.py
│
├── transforms/
│   ├── affine_transforms.py # Image augmentation transforms
│   └── __init__.py
│
├── utils/
│   ├── genLD.py             # Label distribution generation
│   ├── report.py            # Metric reporting
│   ├── utils.py             # Utility functions
│   └── __init__.py
│
├── dataset.py               # Dataset processing
├── train.py                 # Main training script
└── README.md



Setup Instructions

1. Google Drive Setup

Create a folder called "Classification" in your Google Drive
Inside it, create:
"JPEGImages" folder for your medical images
Upload all NNEW_trainval_[0-4].txt and NNEW_test_[0-4].txt files
2. Google Colab Setup

Open a new Colab notebook
Mount Google Drive:
from google.colab import drive
drive.mount('/content/drive')

3. Dataset Path Configuration
In train.py, update these paths:
Original paths
DATA_PATH = '/content/drive/MyDrive/Classification/JPEGImages'
TRAIN_FILE = '/content/drive/MyDrive/Classification/NNEW_trainval_' + cross_val_index + '.txt'
TEST_FILE = '/content/drive/MyDrive/Classification/NNEW_test_' + cross_val_index + '.txt'
Update to your actual paths if different
Example:
DATA_PATH = '/content/drive/MyDrive/MedicalImages/MyDataset/Images'
