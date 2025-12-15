# -*- coding: utf-8 -*-
# Usage:
#   python -m src.infer.cam_infer --model CNN --ckpt checkpoints/cnn.pt --cam 0
import argparse, cv2, numpy as np, torch
from pathlib import Path
from PIL import Image

from src.preprocess.hand_crop import crop_hand, equalize
from src.infer.infer_utils import load_model, get_transform, predict


def smooth_prediction(history, new_value, max_len=5):
    history.append(new_value)
    if len(history) > max_len:
        history.pop(0)
    return max(set(history), key=history.count)


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
    history = []  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cropped = crop_hand(img_pil)   

        
        if cropped == img_pil:
            cv2.putText(frame, "No hand detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
            cv2.imshow("Signify - Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        
        cropped = equalize(cropped)

        
        tensor = transform(cropped).unsqueeze(0)

        
        label, conf = predict(model, tensor, device, idx_to_class)

        
        smoothed_label = smooth_prediction(history, label)

        
        crop_np = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        crop_np = cv2.resize(crop_np, (128, 128))
        frame[10:138, 10:138] = crop_np  

        
        text = f"{smoothed_label} ({conf*100:.1f}%)"
        cv2.putText(frame, text, (160, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3, cv2.LINE_AA)

        cv2.imshow("Signify - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
