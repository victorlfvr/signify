# -*- coding: utf-8 -*-
# Usage:
#   python -m src.train.evaluate_model --data data/split --model CNN --ckpt checkpoints/CNN_best.pt --split test

import argparse, json, torch
from pathlib import Path
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.models.factory import get_model
from src.train.utils_train import evaluate, compute_metrics


def save_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Vraies classes")
    plt.xlabel("Prédictions")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_f1_barplot(report_dict, out_path):
    classes = list(report_dict.keys())[:-3]  
    f1_scores = [report_dict[c]["f1-score"] for c in classes]

    plt.figure(figsize=(14, 5))
    sns.barplot(x=classes, y=f1_scores)
    plt.ylim(0, 1)
    plt.ylabel("F1-score")
    plt.title("F1-score par classe")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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

    model = get_model(args.model, n_classes).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc, preds, gts = evaluate(model, dl, criterion, args.device)

    cm, report = compute_metrics(preds, gts, ds.classes)

    print(f"\nÉvaluation terminée ({args.split})")
    print(f"Accuracy : {val_acc:.2f}%  |  Loss : {val_loss:.4f}")
    print("\n--- Rapport détaillé ---")
    print(report)

    out_dir = Path("runs") / f"eval_{args.model}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_df = pd.DataFrame(cm, index=ds.classes, columns=ds.classes)

    from sklearn.metrics import classification_report
    report_dict = classification_report(gts, preds, target_names=ds.classes, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    cm_df.to_csv(out_dir / "confusion_matrix.csv", index=True)
    report_df.to_csv(out_dir / "classification_report.csv", index=True)

    summary_df = pd.DataFrame({
        "accuracy": [val_acc],
        "loss": [val_loss]
    })
    summary_df.to_csv(out_dir / "summary.csv", index=False)



    save_confusion_matrix(cm, ds.classes, out_dir / "confusion_matrix.png")
    save_f1_barplot(report_dict, out_dir / "f1_scores.png")

    torch.save({"confusion_matrix": cm, "accuracy": val_acc}, out_dir / "metrics.pt")
    with open(out_dir / "report.txt", "w") as f:
        f.write(report)

    print(f"\nRésultats sauvegardés dans : {out_dir}")
    print("Fichiers générés :")
    print("   - confusion_matrix.png")
    print("   - f1_scores.png")
    print("   - evaluation.xlsx")
    print("   - report.txt")
    print("   - metrics.pt")


if __name__ == "__main__":
    main()
