# utils.py
from torchvision import transforms
from PIL import Image
import torch
from model import BrainTumorCNN

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def load_model(model_path='model/model.pth'):
    model = BrainTumorCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image_path, model):
    transform = get_transforms()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    output = model(image_tensor)
    prediction = (output > 0.5).item()
    return "Tumor" if prediction == 1 else "No Tumor"
