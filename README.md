# Pneumonia Detection from Chest X-Ray Images using Deep Learning  

This repository contains the implementation of a deep learning model for **automatic pneumonia detection** from chest X-ray images. The project employs **transfer learning** with a pre-trained **VGG16** architecture using the **PyTorch** framework.  

The model classifies chest X-ray images into two categories:  
- **Normal**  
- **Pneumonia**  

Despite dataset imbalance, the model achieves **97% accuracy in pneumonia detection** and an overall test accuracy of **76%**.  


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


## Future Work
- Handle **class imbalance** (weighted loss, SMOTE, balanced sampling)  
- Explore **ensemble methods** with other CNN architectures (ResNet, DenseNet)  
- Improve **interpretability** using Grad-CAM or attention mechanisms  
- Extend to **multi-class pneumonia classification**  
- Validate across **multiple clinical datasets**  

