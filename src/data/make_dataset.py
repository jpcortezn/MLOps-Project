# src/data/make_dataset.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import requests
import zipfile
from io import BytesIO

def download_fer2013():
    """
    Download FER2013 dataset or use a smaller sample version for testing.
    For this assignment, we'll use a smaller version to keep it manageable.
    """
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Check if dataset already exists
    if os.path.exists('data/raw/fer2013.csv'):
        print("Dataset already exists.")
        return
    
    # For the assignment, we'll use a smaller version of FER2013
    # In a real project, you would download the full dataset
    print("Downloading FER2013 sample...")
    
    # URL for a sample of FER2013 (you might need to replace with actual URL)
    # This is a placeholder - in a real scenario you'd download from kaggle or another source
    url = "https://www.dropbox.com/s/opuvvdv3uligypx/fer2013_sample.zip?dl=1"
    
    try:
        r = requests.get(url)
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall('data/raw/')
        print("Dataset downloaded and extracted successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please download FER2013 manually and place it in data/raw/")
        
class FERDataset(Dataset):
    """Facial Emotion Recognition Dataset."""
    
    def __init__(self, csv_file, root_dir, transform=None, subset='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            subset (string): 'train', 'val', or 'test' subset
        """
        self.emotions_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        
        # Filter by 'Usage' column if it exists, otherwise split manually
        if 'Usage' in self.emotions_frame.columns:
            if subset == 'train':
                self.emotions_frame = self.emotions_frame[self.emotions_frame['Usage'] == 'Training']
            elif subset == 'val':
                self.emotions_frame = self.emotions_frame[self.emotions_frame['Usage'] == 'PublicTest']
            elif subset == 'test':
                self.emotions_frame = self.emotions_frame[self.emotions_frame['Usage'] == 'PrivateTest']
        else:
            # Manual split (80/10/10)
            total = len(self.emotions_frame)
            if subset == 'train':
                self.emotions_frame = self.emotions_frame.iloc[:int(0.8*total)]
            elif subset == 'val':
                self.emotions_frame = self.emotions_frame.iloc[int(0.8*total):int(0.9*total)]
            elif subset == 'test':
                self.emotions_frame = self.emotions_frame.iloc[int(0.9*total):]
                
        # Map emotions to indices
        self.emotion_map = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
    def __len__(self):
        return len(self.emotions_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # For FER2013, pixels are space-separated values in the 'pixels' column
        pixels = self.emotions_frame.iloc[idx]['pixels'].split(' ')
        pixels = np.array(pixels, dtype=np.uint8)
        image = pixels.reshape(48, 48)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        # Get emotion label
        emotion = self.emotions_frame.iloc[idx]['emotion']
        
        if self.transform:
            image = self.transform(image)
            
        sample = {'image': image, 'emotion': emotion}
        return sample

def prepare_dataloaders(batch_size=64):
    """Prepare DataLoaders for the FER2013 dataset."""
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create datasets
    train_dataset = FERDataset(
        csv_file='data/raw/fer2013.csv',
        root_dir='data/raw',
        transform=train_transform,
        subset='train'
    )
    
    val_dataset = FERDataset(
        csv_file='data/raw/fer2013.csv',
        root_dir='data/raw',
        transform=val_test_transform,
        subset='val'
    )
    
    test_dataset = FERDataset(
        csv_file='data/raw/fer2013.csv',
        root_dir='data/raw',
        transform=val_test_transform,
        subset='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader

def main():
    """Main function to download and prepare the dataset."""
    download_fer2013()
    train_loader, val_loader, test_loader = prepare_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Save a sample for visualization
    dataiter = iter(train_loader)
    sample_batch = next(dataiter)
    sample_images = sample_batch['image'].numpy()
    sample_labels = sample_batch['emotion'].numpy()
    
    # Save the first 5 images and their labels
    for i in range(min(5, len(sample_images))):
        image = ((sample_images[i] * 0.5 + 0.5) * 255).astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))  # Convert from CxHxW to HxWxC
        image = Image.fromarray(image.squeeze(), 'L')  # Convert to grayscale PIL image
        os.makedirs('data/processed/samples', exist_ok=True)
        image.save(f'data/processed/samples/sample_{i}_emotion_{sample_labels[i]}.png')
    
    print("Data preparation completed!")

if __name__ == "__main__":
    main()