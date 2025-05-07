# src/models/emotion_cnn.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# Print PyTorch version and MPS availability
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}")
print(f"MPS built: {getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_built()}")

# Force MPS device
try:
    device = torch.device("mps")
    # Test if device works
    test_tensor = torch.ones(1, device=device)
    print(f"Using Apple MPS (Metal) device: {device}")
except Exception as e:
    print(f"Cannot use MPS: {e}")
    # Fall back to CUDA or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Falling back to: {device}")

# Define constants
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMAGE_SIZE = 100
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-4

# RGB Emotion CNN
class RGBEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(RGBEmotionCNN, self).__init__()
        
        # First convolutional block with residual connection
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 input channels for RGB
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.3)
        
        # Second convolutional block with residual connection
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.4)
        
        # Third convolutional block with residual connection
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.4)
        
        # Fourth convolutional block
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(0.5)
        
        # Calculate input size for fully connected layer
        # After four 2x2 max pooling layers: 100/(2^4) = 6.25, so 6×6×256
        self.fc1 = nn.Linear(6 * 6 * 256, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout_fc2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # First block with residual connection
        identity = x
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.bn1_2(self.conv1_2(x))
        x = F.relu(x)  # Apply ReLU after the residual connection
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block with residual connection
        identity = x
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.bn2_2(self.conv2_2(x))
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block with residual connection
        identity = x
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.bn3_2(self.conv3_2(x))
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fourth block
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=True, path='models/checkpoint.pth'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss, model, checkpoint_data):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_data)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_data)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, checkpoint_data):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        # Save model
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        checkpoint_data['model_state_dict'] = model.state_dict()
        torch.save(checkpoint_data, self.path)
        
        self.val_loss_min = val_loss

# Data transforms for RGB images
def get_train_transforms():
    """Returns data augmentation transforms for training RGB images"""
    return transforms.Compose([
        transforms.Resize((112, 112)),  # Slightly larger for random crops
        transforms.RandomCrop((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

def get_val_transforms():
    """Returns transforms for validation/testing RGB images"""
    return transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

def train_model():
    # Data transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Create datasets using ImageFolder (works with directory structure)
    train_dataset = datasets.ImageFolder(root='data/raw/train', transform=train_transform)
    test_dataset = datasets.ImageFolder(root='data/raw/test', transform=val_transform)
    
    # Map class indices to emotion labels
    class_to_emotion = {idx: emotion for idx, emotion in enumerate(sorted(os.listdir('data/raw/train')))}
    
    # Print class mapping
    print("Class to emotion mapping:")
    for class_idx, emotion in class_to_emotion.items():
        print(f"Class {class_idx}: {emotion}")
    
    # Create data loaders with fewer workers for macOS
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model, loss function, and optimizer
    model = RGBEmotionCNN(num_classes=len(class_to_emotion))
    model = model.to(device)
    
    # Verify model is on correct device
    print(f"Model device: {next(model.parameters()).device}")
    
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler - cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE/100)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, path='models/best_emotion_model.pth')
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
        
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        val_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix(loss=loss.item())
        
        epoch_val_loss = running_loss / len(test_dataset)
        epoch_val_acc = 100.0 * correct / total
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Update learning rate with scheduler
        scheduler.step()
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
        
        # Early stopping
        checkpoint_data = {
            'class_to_emotion': class_to_emotion,
            'epoch': epoch,
            'accuracy': epoch_val_acc
        }
        early_stopping(epoch_val_loss, model, checkpoint_data)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Acc: {epoch_val_acc:.2f}%")
        
        # Check if early stopping triggered
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/training_history.png')
    plt.show()
    
    # Load the best model for evaluation
    checkpoint = torch.load('models/best_emotion_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, device, test_loader, class_to_emotion

def evaluate_model(model, device, test_loader, class_to_emotion):
    # Set model to evaluation mode
    model.eval()
    
    # Collect all predictions and ground truth labels
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Get emotion names for labels
    emotion_names = [class_to_emotion[i] for i in range(len(class_to_emotion))]
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/confusion_matrix.png')
    plt.show()
    
    # Print classification report
    report = classification_report(all_labels, all_preds, target_names=emotion_names)
    print("Classification Report:")
    print(report)
    
    # Save report to file
    os.makedirs('reports', exist_ok=True)
    with open('reports/classification_report.txt', 'w') as f:
        f.write(report)

def predict_emotion(image_path, model_path='models/best_emotion_model.pth'):
    """
    Predict emotion for a single image
    """
    # Load the saved model and class mapping
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return None
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    class_to_emotion = checkpoint['class_to_emotion']
    
    # Create the model and load weights
    model = RGBEmotionCNN(num_classes=len(class_to_emotion))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # Prepare image
    transform = get_val_transforms()
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        prob_values, pred_idx = torch.max(probabilities, 1)
    
    predicted_emotion = class_to_emotion[pred_idx.item()]
    confidence = prob_values.item() * 100
    
    # Get probabilities for all emotions
    all_probs = probabilities.squeeze().cpu().numpy() * 100
    emotion_probs = {class_to_emotion[i]: prob for i, prob in enumerate(all_probs)}
    
    return predicted_emotion, confidence, emotion_probs

def main():
    # Train the model
    model, device, test_loader, class_to_emotion = train_model()
    
    # Evaluate the model
    evaluate_model(model, device, test_loader, class_to_emotion)
    
    print("\nTraining and evaluation completed!")
    print("The model has been saved to 'models/best_emotion_model.pth'")
    print("\nTo use the model for prediction, you can use the predict_emotion() function:")
    print("Example: emotion, confidence, probs = predict_emotion('path/to/image.jpg')")

if __name__ == "__main__":
    main()
