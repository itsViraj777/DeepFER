# DeepFER: Facial Emotion Recognition using Deep Learning

DeepFER is a Facial Emotion Recognition system built using Deep Learning. It uses both a custom Convolutional Neural Network (CNN) and a Transfer Learning approach (ResNet18) to classify human facial expressions into various emotional states.

This project supports real-time emotion detection from webcam feed as well as static image files. It has applications in human-computer interaction, mental health monitoring, and smart surveillance.

## 📸 Demo
![Untitled](https://github.com/user-attachments/assets/c478a861-d9dd-4977-af6b-446b24a04042)

![Screenshot (3)](https://github.com/user-attachments/assets/d563a202-2638-4299-99be-4239f992a7b5)

---

## 💡 Features

- Real-time facial emotion recognition via webcam
- Image-based emotion prediction
- Dual-model support: CNN and ResNet18 (Transfer Learning)
- Live confidence score overlay on predictions
- Face detection using OpenCV Haar Cascades
- Emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- Training pipeline with data augmentation and performance visualization

---

## 🧠 Emotion Classes
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😀 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

---

## 🔧 Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Programming language |
| **PyTorch** | Deep learning framework |
| **OpenCV** | Image processing and face detection |
| **Torchvision** | Pre-trained models and transforms |
| **Matplotlib** | Visualization of training history |
| **NumPy & Pandas** | Data manipulation |
| **TQDM** | Progress bars for training |
| **scikit-learn** | Evaluation utilities |

---

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/DeepFER.git
```
2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Install GPU support Ensure you have CUDA-compatible drivers and PyTorch with GPU support installed.

## 🧪 Training the Models
Ensure your training dataset follows this structure:
TrainingImages/
└── sad/
    ├── img1.jpg
    ├── img2.jpg
Then run
```
python train_models.py
```
This will:

1. Train both CNN and Transfer Learning models
2. Save them as emotion_model_cnn.pth and emotion_model_transfer.pth
3. Generate training accuracy/loss graphs

## 📷 Run Inference (Image or Webcam)
```
python facial_recognition.py
```


