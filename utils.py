import os
import torch
import torch.nn as nn
from torchvision import models, transforms
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

    model = models.vgg16(pretrained=False)

    n_inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(n_inputs, 2)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    print("Checkpoint type:", type(checkpoint))

    model.load_state_dict(checkpoint)

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