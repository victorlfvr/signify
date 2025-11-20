import numpy as np
import torch

def mixup(x1, y1, x2, y2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y


def rand_bbox(size, lam):
    _, H, W = size
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def cutmix(x1, y1, x2, y2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    x1 = x1.clone()
    batch, _, _, _ = x1.size()

    for i in range(batch):
        x2_img = x2[i]
        y2_lab = y2[i]
        x1_img = x1[i]

        x1_, y1_, x2_, y2_ = rand_bbox(x1_img.size(), lam)
        x1_img[:, y1_:y2_, x1_:x2_] = x2_img[:, y1_:y2_, x1_:x2_]

        lam = 1 - ((x2_ - x1_) * (y2_ - y1_) / (x1_img.shape[1] * x1_img.shape[2]))

    y = lam * y1 + (1 - lam) * y2
    return x1, y
