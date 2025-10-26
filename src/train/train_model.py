# -*- coding: utf-8 -*-
# Usage :
#   python -m src.train.train_model --data data/split --model CNN --epochs 15

import argparse, json, yaml, torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.models.factory import get_model


def main():
    # ---------- Arguments ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True, help="Nom du modèle : CNN, HybridCNNTransformer")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---------- Configuration ----------
    cfg = yaml.safe_load(open("src/config/training.yaml"))
    EPOCHS = args.epochs or cfg["epochs"]
    BATCH = args.batch or cfg["batch_size"]
    LR = args.lr or cfg["lr"]
    DEVICE = args.device

    # ---------- Transformations ----------
    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # ---------- Dataset ----------
    train_ds = datasets.ImageFolder(args.data / "train", transform=transform)
    val_ds = datasets.ImageFolder(args.data / "val", transform=transform)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    n_classes = len(train_ds.classes)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("runs").mkdir(exist_ok=True)

    # Sauvegarde mapping classes
    with open("checkpoints/class_to_idx.json", "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    # ---------- Modèle ----------
    model = get_model(args.model, n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ---------- Log setup ----------
    run_name = f"{args.model}_lr{LR}_b{BATCH}_ep{EPOCHS}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("checkpoints") / f"{args.model}_best.pt"

    print(f"[INFO] Entraînement du modèle {args.model} ({n_classes} classes)")
    print(f"[INFO] Logs → {run_dir}")
    print(f"[INFO] Checkpoints → {ckpt_path}")

    # ---------- Boucle d'entraînement ----------
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        # ---- Entraînement ----
        model.train()
        train_correct, train_total, train_loss_sum = 0, 0, 0
        for X, y in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / train_total * 100
        train_loss = train_loss_sum / len(train_dl)

        # ---- Validation ----
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X, y in tqdm(val_dl, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total * 100

        print(f"Epoch {epoch+1}/{EPOCHS} | loss={train_loss:.4f} | train_acc={train_acc:.2f}% | val_acc={val_acc:.2f}%")

        # Sauvegarde meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Nouveau meilleur modèle sauvegardé (val_acc={best_val_acc:.2f}%)")

        # Écrit un log simple
        with open(run_dir / "log.txt", "a") as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.2f},{val_acc:.2f}\n")

    print(f"[OK] Entraînement terminé. Meilleure val_acc={best_val_acc:.2f}%")
    print(f"→ Checkpoint final : {ckpt_path}")

if __name__ == "__main__":
    main()
