import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FacialExpressionRecognition:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load CNN model
        self.cnn_model = None
        if self.load_cnn_model():
            print("CNN model loaded successfully")
        else:
            print("Failed to load CNN model")
            
        # Load Transfer Learning model
        self.transfer_model = None
        if self.load_transfer_model():
            print("Transfer Learning model loaded successfully")
        else:
            print("Failed to load Transfer Learning model")
    
    def load_cnn_model(self):
        try:
            print("Attempting to load CNN model...")
            self.cnn_model = CNNModel().to(self.device)
            print("Loading CNN model weights from emotion_model_cnn.pth...")
            self.cnn_model.load_state_dict(torch.load('emotion_model_cnn.pth', map_location=self.device))
            self.cnn_model.eval()
            print("CNN model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            return False
    
    def load_transfer_model(self):
        try:
            print("Attempting to load Transfer Learning model...")
            self.transfer_model = models.resnet18(pretrained=False)
            print("Configuring Transfer Learning model for emotion detection...")
            self.transfer_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.transfer_model.fc.in_features
            self.transfer_model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 7)
            )
            print("Loading Transfer Learning model weights from emotion_model_transfer.pth...")
            self.transfer_model.load_state_dict(torch.load('emotion_model_transfer.pth', map_location=self.device))
            self.transfer_model = self.transfer_model.to(self.device)
            self.transfer_model.eval()
            print("Transfer Learning model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading Transfer Learning model: {e}")
            return False
    
    def preprocess_image(self, image):
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return None
            
        # Get the first face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        
        # Resize to 48x48
        face = cv2.resize(face, (48, 48))
        
        # Normalize
        face = face / 255.0
        
        # Convert to tensor
        face_tensor = torch.FloatTensor(face).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        face_tensor = face_tensor.to(self.device)
        
        return face_tensor, (x, y, w, h)
    
    def predict_emotion(self, image, use_transfer_model=False):
        # Preprocess the image
        result = self.preprocess_image(image)
        if result is None:
            return None, None, None
            
        face_tensor, face_coords = result
        
        # Select model
        model = self.transfer_model if use_transfer_model else self.cnn_model
        
        # Make prediction
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted = torch.argmax(probabilities, 1)
            
        emotion = self.emotions[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()
        
        return emotion, confidence, face_coords
    
    def process_image(self, image_path, use_transfer_model=False):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
            
        # Make prediction
        result = self.predict_emotion(image, use_transfer_model)
        if result is None or result[0] is None:
            print("No face detected in the image")
            return
            
        emotion, confidence, face_coords = result
        if face_coords is None:
            print("No face detected in the image")
            return
            
        x, y, w, h = face_coords
        
        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add emotion label
        label = f"{emotion} ({confidence:.2f})"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow('Facial Expression Recognition', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_webcam(self, use_transfer_model=False):
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Make prediction
            result = self.predict_emotion(frame, use_transfer_model)
            if result is not None and result[0] is not None and result[2] is not None:
                emotion, confidence, face_coords = result
                x, y, w, h = face_coords
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add emotion label
                label = f"{emotion} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display result
            cv2.imshow('Facial Expression Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    fer = FacialExpressionRecognition()
    
    while True:
        print("\nFacial Expression Recognition")
        print("1. Process image file")
        print("2. Process webcam feed")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            image_path = input("Enter the path to the image file: ")
            model_choice = input("Use transfer learning model? (y/n): ").lower()
            fer.process_image(image_path, use_transfer_model=(model_choice == 'y'))
        elif choice == '2':
            model_choice = input("Use transfer learning model? (y/n): ").lower()
            fer.process_webcam(use_transfer_model=(model_choice == 'y'))
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.") 