import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# -----------------------------------
# Class Names
# -----------------------------------
CLASSES = ["Normal", "Pneumonia"]

# -----------------------------------
# Load Model
# -----------------------------------
import os
import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path=None):

    if model_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "vgg16-chest-4.pth")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    #  Initialize model architecture
    model = models.vgg16(pretrained=False)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    return model


# -----------------------------------
# Image Preprocessing
# -----------------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)

    return image


# -----------------------------------
# Prediction Function
# -----------------------------------
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.exp(output)
        confidence, predicted = torch.max(probabilities, 1)

    return CLASSES[predicted.item()], confidence.item()