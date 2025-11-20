# -*- coding: utf-8 -*-
import argparse, json, yaml, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F

from src.models.factory import get_model
from src.preprocess.augment import mixup, cutmix   # <-- à créer

def main():
    # ----- Arguments -----
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ----- Config -----
    cfg = yaml.safe_load(open("src/config/training.yaml"))
    EPOCHS = args.epochs or cfg["epochs"]
    BATCH = args.batch or cfg["batch_size"]
    LR = args.lr or cfg["lr"]
    DEVICE = args.device

    # ----- Transformations -----
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
        ),
        transforms.RandomRotation(25),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.8, 1.2)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # ----- Dataset -----
    train_ds = datasets.ImageFolder(args.data / "train", transform=train_transform)
    val_ds   = datasets.ImageFolder(args.data / "val",   transform=val_transform)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    n_classes = len(train_ds.classes)

    # Mapping classes
    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/class_to_idx.json", "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    # ----- Modèle -----
    model = get_model(args.model, n_classes).to(DEVICE)

    # MixUp / CutMix active ou pas ?
    USE_MIXUP = True
    USE_CUTMIX = False   # ne JAMAIS activer les deux en même temps

    # Très important : BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ----- Logs -----
    run_dir = Path("runs") / f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("checkpoints") / f"{args.model}_best.pt"

    best_val_acc = 0.0

    # ----- Training Loop -----
    for epoch in range(EPOCHS):

        # ---- TRAIN ----
        model.train()
        train_correct, total_loss, total = 0, 0, 0

        for X, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            X, y = X.to(DEVICE), y.to(DEVICE)

            # One-hot
            y_onehot = F.one_hot(y, num_classes=n_classes).float()

            # MixUp / CutMix
            if USE_MIXUP:
                idx = torch.randperm(X.size(0))
                X, y_onehot = mixup(X, y_onehot, X[idx], y_onehot[idx])

            if USE_CUTMIX:
                idx = torch.randperm(X.size(0))
                X, y_onehot = cutmix(X, y_onehot, X[idx], y_onehot[idx])

            optimizer.zero_grad()
            out = model(X)

            loss = criterion(out, y_onehot)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += y.size(0)
            train_correct += (out.argmax(1) == y).sum().item()

        train_acc = 100 * train_correct / total

        # ---- VALIDATION ----
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for X, y in tqdm(val_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}: train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] meilleur modèle sauvegardé ({val_acc:.2f}%)")

    print(f"[OK] Meilleure val_acc = {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
