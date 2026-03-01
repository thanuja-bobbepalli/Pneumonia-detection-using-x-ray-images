# Pneumonia Detection from Chest X-Ray Images using Deep Learning  

This repository contains the implementation of a deep learning model for **automatic pneumonia detection** from chest X-ray images. The project employs **transfer learning** with a pre-trained **VGG16** architecture using the **PyTorch** framework.  

The model classifies chest X-ray images into two categories:  
- **Normal**  
- **Pneumonia**  

Despite dataset imbalance,Achieves **97%** sensitivity for Pneumonia class and **76%** overall test accuracy.

## 📁 Project Structure
```
.
├── app.py
├── utils.py
├── Dockerfile
├── pnemonia Dectection using Pytorch.ipynb
├── requirements.txt
└── README.md
```
This project is a Deep Learning-based Pneumonia Detection system built using:

- PyTorch (VGG16)
- Streamlit (Web Interface)
- Docker (Containerization)
## Features
- Transfer Learning with **VGG16** pre-trained on ImageNet  
- **Data augmentation** to improve generalization  
- **Cross-entropy loss** and **Adam optimizer**  
- **Early stopping** and **model checkpointing** to prevent overfitting  
- Evaluation metrics: Accuracy, Per-class Accuracy, Confusion Matrix  

## Dataset
- Source: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- Training Set: **5,216 images**  
  - **Normal:** 1,341  
  - **Pneumonia:** 3,875  
- Test Set: **576 images**  

## Methodology
1. **Preprocessing**  
   - Resizing to **224×224** pixels  
   - Normalization using ImageNet statistics  
   - Data augmentation (rotation, flip, color jitter, crop)  

2. **Model Architecture**  
   - Base Model: **VGG16**  
   - Frozen convolutional layers  
   - Custom fully connected classifier (2 classes)  
   - Dropout layers for regularization  

3. **Training Strategy**  
   - Loss: **CrossEntropyLoss**  
   - Optimizer: **Adam** (lr = 0.001)  
   - Batch Size: 64  
   - Epochs: 10  
   - Early stopping (patience = 5)  


## Results
| Class      | Accuracy | Correct Predictions | Total Samples |
|------------|----------|---------------------|---------------|
| Normal     | 41%      | 92                  | 220           |
| Pneumonia  | 97%      | 348                 | 356           |
| **Overall**| **76%**  | 440                 | 576           |

- Training Accuracy: **85.2%**  
- Validation Accuracy: **82.1%**  

Strong performance in pneumonia detection but limited accuracy in normal cases due to dataset imbalance.  

## Deployes it in Streamlit 
Check it out [web_application](https://pneumonia-detection-using-x-ray-images-tj9amadclyqx6tm6tz8hgx.streamlit.app/){:target="_blank"}

## 🏗 Deployment Architecture

User → Streamlit Web App → PyTorch Model → Prediction Output

## Docker Image (Public)

Docker image is available on Docker Hub:
 https://hub.docker.com/r/thanujabobbepalli/pneumonia-app

---

## How to Run Using Docker ,Run Using Prebuilt Docker Image 
### 1️⃣ Install Docker

Make sure Docker is installed:

```bash
docker --version
```
### 2️⃣ Pull the Image

```bash
docker pull thanujabobbepalli/pneumonia-app
```
### 3️⃣ Run the Container

```bash
docker run -p 8501:8501 thanujabobbepalli/pneumonia-app
                ( or )
docker run --network host pneumonia-app
```
### 4️⃣ Open in Browser

Go to:

```
http://localhost:8501
```

You will see the Streamlit application.

---
## Run This Project Using Docker (Build from Source)
## Clone Repository and Build Docker Image Yourself

If you want to run the project directly from this GitHub repository:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/thanuja-bobbepalli/pneumonia-detection.git
cd pneumonia-detection
```

### 2️ Build the Docker Image

Make sure you are inside the project folder (where Dockerfile exists), then run:

```bash
docker build -t pneumonia-app .
```
This command:
- Reads the Dockerfile
- Installs dependencies
- Creates a Docker image named pneumonia-app

### 3️⃣ Run the Docker Container
```bash
docker run -p 8501:8501 pneumonia-app
```

### 4️⃣ Open in Browser
```bash
http://localhost:8501
```
Note : The trained model weights are automatically downloaded during container startup.

## Future Work
- Handle **class imbalance** (weighted loss, SMOTE, balanced sampling)  
- Explore **ensemble methods** with other CNN architectures (ResNet, DenseNet)  
- Improve **interpretability** using Grad-CAM or attention mechanisms  
- Extend to **multi-class pneumonia classification**  
- Validate across **multiple clinical datasets**  

