# -*- coding: utf-8 -*-
# Usage:
#   python -m src.train.evaluate_model --data data/split --model CNN --ckpt checkpoints/CNN_best.pt --split test

import argparse, json, torch
from pathlib import Path
from torchvision import datasets, transforms
from src.models.factory import get_model
from src.train.utils_train import evaluate, compute_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    ds = datasets.ImageFolder(args.data / args.split, transform=transform)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    n_classes = len(ds.classes)

    # Chargement du modèle et des poids
    model = get_model(args.model, n_classes).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, preds, gts = evaluate(model, dl, criterion, args.device)

    cm, report = compute_metrics(preds, gts, ds.classes)

    print(f"\n✅ Évaluation terminée ({args.split})")
    print(f"Accuracy : {val_acc:.2f}%  |  Loss : {val_loss:.4f}")
    print("\n--- Rapport détaillé ---")
    print(report)

    # Sauvegarde des résultats
    out_dir = Path("runs") / f"eval_{args.model}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"confusion_matrix": cm, "accuracy": val_acc}, out_dir / "metrics.pt")
    with open(out_dir / "report.txt", "w") as f:
        f.write(report)
    print(f"\nRésultats sauvegardés dans {out_dir}")

if __name__ == "__main__":
    main()
