import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.image_paths = []
        self.labels = []
        
        # Check if root directory exists
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
            
        # Load all image paths and labels
        for class_idx, emotion in enumerate(self.classes):
            class_dir = os.path.join(root_dir, emotion)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
                
            image_count = 0
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
                    image_count += 1
                    
            print(f"Loaded {image_count} images for emotion: {emotion}")
            
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}. Please ensure the directory contains subdirectories for each emotion with image files.")
            
        print(f"Total images loaded: {len(self.image_paths)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
            
        # Apply face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            image = image[y:y+h, x:x+w]
        
        # Resize to 48x48
        image = cv2.resize(image, (48, 48))
        
        # Convert to tensor and normalize
        image = torch.FloatTensor(image).unsqueeze(0) / 255.0
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

class ImprovedCNNModel(nn.Module):
    def __init__(self):
        super(ImprovedCNNModel, self).__init__()
        # First Convolutional Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Second Convolutional Block
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Third Convolutional Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_models(data_dir, batch_size=32, num_epochs=50):
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
        
    # Define paths for training and testing data
    train_dir = data_dir
    test_dir = os.path.join(os.path.dirname(data_dir), 'TestingImages')
    
    # If testing directory doesn't exist, create it and split the data
    if not os.path.exists(test_dir):
        print(f"Testing directory not found. Creating {test_dir} and splitting data...")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create emotion subdirectories in testing directory
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
            os.makedirs(os.path.join(test_dir, emotion), exist_ok=True)
            
        # Split data: move 20% of images to testing directory
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
            emotion_dir = os.path.join(train_dir, emotion)
            if not os.path.exists(emotion_dir):
                print(f"Warning: Directory not found: {emotion_dir}")
                continue
                
            # Get list of image files
            image_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                print(f"Warning: No images found in {emotion_dir}")
                continue
                
            # Calculate number of files to move (20%)
            num_files = len(image_files)
            num_test = max(1, int(num_files * 0.2))  # At least 1 file for testing
            
            # Randomly select files to move
            import random
            test_files = random.sample(image_files, num_test)
            
            # Move files to testing directory
            for file in test_files:
                src = os.path.join(emotion_dir, file)
                dst = os.path.join(test_dir, emotion, file)
                import shutil
                shutil.copy2(src, dst)  # Copy instead of move to preserve original data
                
            print(f"Moved {len(test_files)} files from {emotion} to testing set")
    
    print(f"Loading data from:")
    print(f"Training directory: {train_dir}")
    print(f"Testing directory: {test_dir}")
    
    # Data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    ])
    
    try:
        # Create datasets
        print("\nLoading training dataset...")
        train_dataset = EmotionDataset(train_dir, transform=train_transform)
        print("\nLoading testing dataset...")
        test_dataset = EmotionDataset(test_dir)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print(f"\nTraining samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
        
        # Initialize models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn_model = ImprovedCNNModel().to(device)
        
        # Transfer Learning model with improved architecture
        transfer_model = models.resnet18(pretrained=True)
        transfer_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = transfer_model.fc.in_features
        transfer_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)
        )
        transfer_model = transfer_model.to(device)
        
        # Loss function and optimizers with learning rate scheduling
        criterion = nn.CrossEntropyLoss()
        cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=0.001, weight_decay=0.01)
        transfer_optimizer = optim.AdamW(transfer_model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Learning rate schedulers
        cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        transfer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(transfer_optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        # Training history
        history = {
            'cnn_train_loss': [], 'cnn_train_acc': [], 'cnn_val_acc': [],
            'transfer_train_loss': [], 'transfer_train_acc': [], 'transfer_val_acc': []
        }
        
        # Train CNN model
        print("Training CNN model...")
        for epoch in range(num_epochs):
            cnn_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                images, labels = images.to(device), labels.to(device)
                
                cnn_optimizer.zero_grad()
                outputs = cnn_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                cnn_optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation
            cnn_model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = cnn_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            
            # Update learning rate
            cnn_scheduler.step(val_acc)
            
            # Save metrics
            history['cnn_train_loss'].append(train_loss)
            history['cnn_train_acc'].append(train_acc)
            history['cnn_val_acc'].append(val_acc)
            
            print(f'CNN Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Train Transfer Learning model
        print("\nTraining Transfer Learning model...")
        for epoch in range(num_epochs):
            transfer_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                images, labels = images.to(device), labels.to(device)
                
                transfer_optimizer.zero_grad()
                outputs = transfer_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                transfer_optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation
            transfer_model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = transfer_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            
            # Update learning rate
            transfer_scheduler.step(val_acc)
            
            # Save metrics
            history['transfer_train_loss'].append(train_loss)
            history['transfer_train_acc'].append(train_acc)
            history['transfer_val_acc'].append(val_acc)
            
            print(f'Transfer Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save models
        torch.save(cnn_model.state_dict(), 'emotion_model_cnn.pth')
        torch.save(transfer_model.state_dict(), 'emotion_model_transfer.pth')
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['cnn_train_loss'], label='Train Loss')
        plt.title('CNN Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['cnn_train_acc'], label='Train Acc')
        plt.plot(history['cnn_val_acc'], label='Val Acc')
        plt.title('CNN Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history_cnn.png')
        plt.close()
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['transfer_train_loss'], label='Train Loss')
        plt.title('Transfer Learning Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['transfer_train_acc'], label='Train Acc')
        plt.plot(history['transfer_val_acc'], label='Val Acc')
        plt.title('Transfer Learning Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history_transfer.png')
        plt.close()
        
        print("\nTraining completed!")
        print(f"Final CNN Model - Train Acc: {history['cnn_train_acc'][-1]:.2f}%, Val Acc: {history['cnn_val_acc'][-1]:.2f}%")
        print(f"Final Transfer Model - Train Acc: {history['transfer_train_acc'][-1]:.2f}%, Val Acc: {history['transfer_val_acc'][-1]:.2f}%")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train_models('TrainingImages') 