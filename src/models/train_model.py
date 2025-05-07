# src/models/train_model.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

# Import from our modules
from src.data.make_dataset import prepare_dataloaders
from src.models.model import EmotionCNN, ResidualEmotionCNN

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=25, device='cpu', save_dir='models'):
    """
    Train the model.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on ('cpu' or 'cuda')
        save_dir: Directory to save model checkpoints
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Set up MLflow tracking
    mlflow.set_experiment("emotion_recognition")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Track best model and its metrics
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Start MLflow run
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("model_type", model.__class__.__name__)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("learning_rate", optimizer.param_groups[0]['lr'])
        
        # Start training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    dataloader = train_loader
                else:
                    model.eval()   # Set model to evaluate mode
                    dataloader = val_loader
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data
                for batch_idx, data in enumerate(dataloader):
                    inputs = data['image'].to(device)
                    labels = data['emotion'].to(device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    # Track history only if in train phase
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # Print batch progress
                    if batch_idx % 20 == 0:
                        print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
                
                # Step scheduler if in train phase
                if phase == 'train' and scheduler is not None:
                    scheduler.step()
                
                # Calculate epoch metrics
                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)
                
                # Log metrics
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                    mlflow.log_metric("train_acc", epoch_acc.item(), step=epoch)
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())
                    mlflow.log_metric("val_loss", epoch_loss, step=epoch)
                    mlflow.log_metric("val_acc", epoch_acc.item(), step=epoch)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Save best model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
                    torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))
                    print(f'New best model saved with accuracy: {best_acc:.4f}')
                    
                    # Log best model with MLflow
                    mlflow.pytorch.log_model(model, "best_model")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                
            print()
        
        # Log training time
        time_elapsed = time.time() - start_time
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')
        
        # Load best model weights
        model.load_state_dict(best_model_wts)
        
        # Save final model
        torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
        
        # Log final model with MLflow
        mlflow.pytorch.log_model(model, "final_model")
        
        # Plot and save training curves
        plot_training_curves(history, save_dir)
        
    return model, history

def evaluate_model(model, test_loader, criterion, device='cpu'):
    """
    Evaluate the model on the test set.
    
    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        criterion: Loss function
        device: Device to evaluate on ('cpu' or 'cuda')
    
    Returns:
        test_loss: Average test loss
        test_acc: Test accuracy
        confusion_matrix: Confusion matrix
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    # For confusion matrix
    all_preds = []
    all_labels = []
    
    # Iterate over data
    with torch.no_grad():
        for data in test_loader:
            inputs = data['image'].to(device)
            labels = data['emotion'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Save predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    return test_loss, test_acc.item(), conf_mat

def plot_training_curves(history, save_dir):
    """
    Plot training and validation curves.
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
    """
    # Create figure directory if it doesn't exist
    fig_dir = os.path.join(save_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'training_curves.png'))
    plt.close()
    
def plot_confusion_matrix(conf_mat, class_names, save_dir):
    """
    Plot confusion matrix.
    
    Args:
        conf_mat: Confusion matrix
        class_names: Class names
        save_dir: Directory to save plots
    """
    # Create figure directory if it doesn't exist
    fig_dir = os.path.join(save_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks and label them with the respective list entries
    ax.set(xticks=np.arange(conf_mat.shape[1]),
           yticks=np.arange(conf_mat.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, format(conf_mat[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_mat[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    """Main function to train the emotion recognition model."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # If using M4 Pro, enable MPS (Metal Performance Shaders)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) on Mac M4 Pro")
    
    # Prepare dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(batch_size=64)
    
    # Create model
    # model = EmotionCNN(num_classes=7)
    model = ResidualEmotionCNN(num_classes=7)  # Using the better model
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=15,  # Reduced for faster training
        device=device,
        save_dir='models'
    )
    
    # Evaluate model on test set
    test_loss, test_acc, conf_mat = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Plot confusion matrix
    emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    plot_confusion_matrix(conf_mat, emotion_classes, 'models')
    
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model as ONNX for deployment
    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    onnx_path = os.path.join('models', 'emotion_recognition.onnx')
    torch.onnx.export(model, dummy_input, onnx_path, 
                     input_names=['input'], output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'}})
    
    print(f"Model exported to ONNX format at {onnx_path}")

if __name__ == "__main__":
    main()