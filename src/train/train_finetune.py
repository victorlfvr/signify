# -*- coding: utf-8 -*-
"""
Fine-tuning d'un modèle CNN pré-entraîné sur un nouveau dataset.
Usage :
python -m src.train.train_finetune --data data/split_2 --pretrained checkpoints/CNN_dataset1.pt --output cnn_finetuned.pt
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.factory import get_model


def load_pretrained(model, pretrained_path, device):
    state = torch.load(pretrained_path, map_location=device)

    keys_to_remove = [k for k in state.keys() if k.startswith("classifier.5")]
    for k in keys_to_remove:
        del state[k]

    model.load_state_dict(state, strict=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pretrained", type=str, required=True)
    ap.add_argument("--output", type=str, default="cnn_finetuned.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    DEVICE = args.device

    train_t = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


    val_t = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_ds = datasets.ImageFolder(args.data / "train", train_t)
    val_ds   = datasets.ImageFolder(args.data / "val", val_t)

    n_classes = len(train_ds.classes)
    print(f"[INFO] Fine-tuning sur {n_classes} classes")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = get_model("cnn", n_classes).to(DEVICE)
    model = load_pretrained(model, args.pretrained, DEVICE)

    for p in model.features.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(
        list(model.features.parameters()) + list(model.classifier.parameters()),
        lr=args.lr
    )

    criterion = nn.CrossEntropyLoss()
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):

        model.train()
        train_correct = train_total = 0

        for X, y in tqdm(train_dl, desc=f"[Train E{epoch}]"):
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_correct += (out.argmax(1) == y).sum().item()
            train_total += len(y)

        train_acc = 100 * train_correct / train_total
        model.eval()
        val_correct = val_total = 0

        with torch.no_grad():
            for X, y in tqdm(val_dl, desc=f"[Val E{epoch}]"):
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += len(y)

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch}/{args.epochs} | Train={train_acc:.2f}% | Val={val_acc:.2f}%")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"[INFO] Nouveau meilleur modèle → {args.output}")

    print(f"[OK] Fine-tuning terminé | Best Val Acc = {best_val:.2f}%")


if __name__ == "__main__":
    main()
