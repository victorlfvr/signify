# -*- coding: utf-8 -*-
# Usage:
#   python -m src.cam_infer --ckpt checkpoints/best_small_cnn.pt --classes checkpoints/class_to_idx.json --cam 0 --model small_cnn
from pathlib import Path
import argparse, json, time, collections
import numpy as np
import cv2
import torch
import torch.nn as nn


try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("MediaPipe non installé. `pip install mediapipe`") from e

# Le modèle est maintenant importé depuis src.models

def load_classes(p: Path, ckpt_dict=None):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            idx2cls = json.load(f)
        return [idx2cls[str(i)] for i in range(len(idx2cls))]
    if ckpt_dict is not None and "classes" in ckpt_dict: return ckpt_dict["classes"]
    raise RuntimeError("Impossible de charger les classes.")

def clamp(v, lo, hi): return max(lo, min(hi, v))

def square_box(x0,y0,x1,y1,w,h, margin=0.20, min_side=140):
    cx, cy = (x0+x1)/2.0, (y0+y1)/2.0
    side = max(1, x1-x0, y1-y0)
    side = int(max(side, min_side) + 0.5)
    x0n, y0n = int(cx-side/2), int(cy-side/2)
    x1n, y1n = x0n+side, y0n+side
    # marge relative
    m = int(margin * side)
    x0n -= m; y0n -= m; x1n += m; y1n += m
    x0n, y0n = clamp(x0n,0,w-1), clamp(y0n,0,h-1)
    x1n, y1n = clamp(x1n,1,w), clamp(y1n,1,h)
    if x1n<=x0n: x1n=min(w,x0n+1)
    if y1n<=y0n: y1n=min(h,y0n+1)
    # recarré
    side = max(x1n-x0n, y1n-y0n)
    cx, cy = (x0n+x1n)//2, (y0n+y1n)//2
    x0n, y0n = cx - side//2, cy - side//2
    x1n, y1n = x0n + side, y0n + side
    x0n, y0n = clamp(x0n,0,w-1), clamp(y0n,0,h-1)
    x1n, y1n = clamp(x1n,1,w), clamp(y1n,1,h)
    return x0n,y0n,x1n,y1n

def to_tensor_96(rgb):
    dbg = cv2.resize(rgb, (96,96), interpolation=cv2.INTER_LINEAR)
    arr = dbg.astype(np.float32)/255.0
    arr = np.transpose(arr, (2,0,1))
    return torch.from_numpy(arr).unsqueeze(0), dbg  # [1,3,96,96], RGB

def draw_hud(img, lines, x=10, y=12, w=280, h_step=24, alpha=0.35):
    h = h_step*(len(lines)+1)
    overlay = img.copy()
    cv2.rectangle(overlay, (x-8, y-8), (x-8+w, y-8+h), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    yy = y
    for txt, color in lines:
        cv2.putText(img, txt, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        yy += h_step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/CNN_best.pt"))
    ap.add_argument("--classes", type=Path, default=Path("checkpoints/class_to_idx.json"))
    ap.add_argument("--cam", type=int, default=0)
    # stabilité
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--smooth", type=int, default=15)
    ap.add_argument("--lock_k", type=int, default=7)
    ap.add_argument("--gap", type=float, default=0.20)
    # cadrage
    ap.add_argument("--margin", type=float, default=0.18)
    ap.add_argument("--minside", type=int, default=180)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    classes = load_classes(args.classes, ckpt if isinstance(ckpt, dict) else None)
    n = len(classes)

    # Créer le modèle avec l'architecture spécifiée
    model = create_model(args.model, n_classes=n, p=0.3).to(device)
    print(f"Modèle utilisé: {args.model}")
    print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
    
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state); model.eval()
    softmax = torch.nn.Softmax(dim=1)

    buf_preds = collections.deque(maxlen=max(1,args.smooth))
    locked_label = None

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(False, max_num_hands=1, model_complexity=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened(): raise SystemExit("Caméra introuvable.")

    fps_t0=time.time(); fps_n=0; fps=0.0
    dbg_patch = np.zeros((96,96,3), np.uint8)

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            H,W = frame.shape[:2]
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb_full)

            pred_text="—"; conf=0.0; box=None; top3=[]

            if res.multi_hand_landmarks:
                # choisir la main avec boîte la plus large
                best=None; best_area=0; best_lm=None
                for lm in res.multi_hand_landmarks:
                    xs=[int(p.x*W) for p in lm.landmark]; ys=[int(p.y*H) for p in lm.landmark]
                    x0,x1=min(xs),max(xs); y0,y1=min(ys),max(ys)
                    bx0,by0,bx1,by1 = square_box(x0,y0,x1,y1,W,H, margin=args.margin, min_side=args.minside)
                    area=(bx1-bx0)*(by1-by0)
                    if area>best_area: best_area=area; best=(bx0,by0,bx1,by1); best_lm=lm

                if best:
                    x0,y0,x1,y1 = best
                    crop_bgr = frame[y0:y1, x0:x1].copy()

                    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    inp, dbg = to_tensor_96(rgb)
                    dbg_patch = cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR)

                    inp = inp.to(device, non_blocking=True)
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=="cuda" else torch.bfloat16):
                        probs = softmax(model(inp))
                    vals, inds = torch.topk(probs, k=min(3,n), dim=1)
                    idx=int(inds[0,0]); conf=float(vals[0,0]); box=best
                    top3=[(classes[int(inds[0,i])], float(vals[0,i])) for i in range(vals.size(1))]

                    buf_preds.append(idx)
                    maj_idx, maj_count = Counter(buf_preds).most_common(1)[0]
                    second=float(vals[0,1]) if vals.size(1)>1 else 0.0
                    if locked_label is None: locked_label=maj_idx
                    elif maj_idx!=locked_label and (maj_count>=args.lock_k or (conf-second)>=args.gap):
                        locked_label=maj_idx
                    pred_text = classes[locked_label] if conf>=args.conf else "nothing"
            else:
                pred_text="nothing"; conf=1.0; buf_preds.clear(); locked_label=None

            # overlays
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            if box is not None:
                x0,y0,x1,y1 = box
                cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,255),2)

            fps_n+=1
            if fps_n>=10:
                t=time.time(); fps=fps_n/(t-fps_t0+1e-9); fps_t0=t; fps_n=0

            hud = [
                (f"Pred: {pred_text}", (0,255,0) if pred_text not in ["?","—","nothing"] else (0,0,255)),
                (f"Conf: {conf:.2f}", (255,255,255)),
            ]
            for lbl,pr in top3: hud.append((f"{lbl}: {pr:.2f}", (255,255,0)))
            hud.append((f"FPS: {fps:.1f}", (255,255,0)))
            draw_hud(frame, hud)

            cv2.imshow("Signify - Crop 96x96", dbg_patch)
            cv2.imshow("Signify - Webcam Inference", frame)
            k=cv2.waitKey(1)&0xFF
            if k==27 or k==ord('q'): break
    finally:
        cap.release(); hands.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
