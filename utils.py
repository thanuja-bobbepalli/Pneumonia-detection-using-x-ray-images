import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import vgg16
import gdown

# -----------------------------------
# Class Names
# -----------------------------------
CLASSES = ["Normal", "Pneumonia"]

# -----------------------------------
# Google Drive Model Download
# -----------------------------------
MODEL_URL = "https://drive.google.com/uc?id=16aoMiOSh6WPkWB5sIAfO0g-pWajzc1t0"
MODEL_FILE = "vgg16-chest-4.pth"

def download_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# -----------------------------------
# Load Model
# -----------------------------------
def load_model():

    download_model()

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    model = vgg16(weights=None)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']

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