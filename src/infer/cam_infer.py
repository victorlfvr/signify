# -*- coding: utf-8 -*-
import argparse, cv2, numpy as np
from src.infer.infer_utils import load_model, get_transform, predict
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--classes", type=str, default="checkpoints/class_to_idx.json")
    ap.add_argument("--image_size", type=int, default=96)
    args = ap.parse_args()

    model, idx_to_class, device = load_model(args.model, args.ckpt, args.classes)
    transform = get_transform(args.image_size)



    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la caméra.")

    print("[INFO] Appuie sur 'q' pour quitter.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
    

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = transform(img_pil).unsqueeze(0)


        label, conf = predict(model, tensor, device, idx_to_class)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)
        cv2.imshow("Signify - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
