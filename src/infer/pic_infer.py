# -*- coding: utf-8 -*-
import argparse, cv2
from pathlib import Path
import numpy as np
from PIL import Image

from src.infer.infer_utils import load_model, get_transform, predict
from src.preprocess.hand_crop import crop_hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Nom du modèle : CNN, HybridCNNTransformer")
    ap.add_argument("--ckpt", type=str, required=True, help="Chemin du checkpoint")
    ap.add_argument("--img", type=str, required=True, help="Image à prédire")
    ap.add_argument("--classes", type=str, default="checkpoints/class_to_idx.json")
    ap.add_argument("--image_size", type=int, default=96)
    args = ap.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        raise FileNotFoundError(f"Image introuvable : {img_path}")

    model, idx_to_class, device = load_model(args.model, args.ckpt, args.classes)
    transform = get_transform()

    pil_img = Image.open(img_path).convert("RGB")

    pil_img = Image.open(img_path).convert("RGB")

    cropped = crop_hand(pil_img)
    if cropped == pil_img:
        print("[WARN] Aucune main détectée → prédiction incertaine.")

    tensor = transform(cropped).unsqueeze(0)

    label, conf = predict(model, tensor, device, idx_to_class)
    print(f"\n=== PRÉDICTION ===")
    print(f"Lettre : {label}")
    print(f"Confiance : {conf*100:.1f}%")
    print("==================\n")

    out_dir = Path("runs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
    cv2.putText(
        annotated,
        f"{label} ({conf*100:.1f}%)",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    out_file = out_dir / f"{img_path.stem}_pred.jpg"
    cv2.imwrite(str(out_file), annotated)

    print(f"Résultat sauvegardé dans : {out_file}")


if __name__ == "__main__":
    main()
