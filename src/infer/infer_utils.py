# -*- coding: utf-8 -*-
import torch, json
from torchvision import transforms
from pathlib import Path
from src.models.factory import get_model


def load_model(model_name: str, ckpt_path: Path, classes_json: Path, device: str = None):
    """Charge le modèle et la correspondance classes ↔ indices."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Charger classes
    with open(classes_json, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    n_classes = len(idx_to_class)

    # Charger modèle
    model = get_model(model_name, n_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    return model, idx_to_class, device


def get_transform(image_size=96):
    """Transformations d'entrée pour l'inférence."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])



def predict(model, img, device, idx_to_class):
    """Retourne la classe prédite et la probabilité."""
    model.eval()
    with torch.no_grad():
        out = model(img.to(device))
        probs = torch.softmax(out, dim=1)
        conf, pred = probs.max(1)
        label = idx_to_class[pred.item()]
        return label, conf.item()
