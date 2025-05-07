# src/data/make_dataset.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import shutil

def download_fer2013():
    """Download FER2013 dataset using Kaggle API."""
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        import kaggle
        print("Downloading FER2013 dataset from Kaggle...")
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            "msambare/fer2013", 
            path='data/raw',
            unzip=True
        )
        
        # Check what files were downloaded
        files = os.listdir('/Users/dionrizovelarde/Documents/ITESO/Semestre_10/Integración_de_servicios_de_aprendizaje_MLops/MLOps-Project/data/raw')
        print(f"Files in data/raw: {files}")
        
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTo use Kaggle API, follow these steps:")
        print("1. Create a Kaggle account if you don't have one")
        print("2. Go to Account → Create New API Token")
        print("3. Save the kaggle.json file to ~/.kaggle/")
        print("4. Install the Kaggle package: pip install kaggle")
        print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("\nPlease fix the issue and run again.")
        return
