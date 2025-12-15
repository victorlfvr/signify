# -*- coding: utf-8 -*-
# Usage:
#   python -m script.preview_batch --root data/split --split train --batch 64 --seed 42
from pathlib import Path
import argparse, random
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_classes(root: Path):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    return classes

class SimpleImageDataset(Dataset):
    def __init__(self, split_dir: Path, classes: list[str], is_train: bool):
        self.split_dir = split_dir
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.samples = []
        for c in classes:
            for p in (split_dir / c).iterdir():
                if p.is_file() and p.suffix.lower() in EXTS:
                    self.samples.append((p, self.class_to_idx[c]))
        self.is_train = is_train
        self.rng = random.Random(42)

    def _augment(self, img: Image.Image) -> Image.Image:
        # flip H p=0.5
        if self.rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # rotation ±10°
        angle = self.rng.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0,0,0))
        # brightness / contrast ±15%
        b = 1.0 + self.rng.uniform(-0.15, 0.15)
        c = 1.0 + self.rng.uniform(-0.15, 0.15)
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        return img

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.is_train:
            img = self._augment(img)
        
        img = img.resize((96, 96), resample=Image.BILINEAR)
        
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2,0,1))  
        x = torch.from_numpy(arr)
        return x, y

def make_grid_bgr(batch_imgs: torch.Tensor, batch_labels: torch.Tensor, classes: list[str], ncols=8):
    B = batch_imgs.shape[0]
    nrows = int(np.ceil(B / ncols))
    cell_h, cell_w = 96, 96
    pad = 2
    grid = np.zeros((nrows*(cell_h+pad)+pad, ncols*(cell_w+pad)+pad, 3), dtype=np.uint8)
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i >= B: break
            img = (batch_imgs[i].numpy().transpose(1,2,0) * 255.0).clip(0,255).astype(np.uint8)  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            y = int(batch_labels[i].item())
            ytext = classes[y]
            y0 = r*(cell_h+pad)+pad
            x0 = c*(cell_w+pad)+pad
            grid[y0:y0+cell_h, x0:x0+cell_w] = img
            cv2.rectangle(grid, (x0, y0), (x0+cell_w, y0+cell_h), (255,255,255), 1)
            cv2.putText(grid, ytext, (x0+4, y0+14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)
            i += 1
    return grid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/split"))
    ap.add_argument("--split", type=str, choices=["train","val","test"], default="train")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split_path = args.root / args.split
    classes = list_classes(split_path)
    print("Classes ({}):".format(len(classes)), classes)

    ds = SimpleImageDataset(split_path, classes, is_train=(args.split=="train"))
    print(f"{args.split} samples:", len(ds))

    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    X, y = next(iter(dl))  # one batch
    print("Batch X:", tuple(X.shape), "dtype:", X.dtype, "range:", (float(X.min()), float(X.max())))
    print("Batch y:", y.shape, "min/max:", int(y.min()), int(y.max()))

    grid = make_grid_bgr(X, y, classes, ncols=8)
    cv2.imshow(f"Signify batch [{args.split}]", grid)
    print("Press ESC to close window.")
    while True:
        k = cv2.waitKey(10) & 0xFF
        if k == 27: break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
