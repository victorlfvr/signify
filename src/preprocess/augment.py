import numpy as np

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
    batch = x1.size(0)

    for i in range(batch):
        bbx1, bby1, bbx2, bby2 = rand_bbox(x1[i].size(), lam)
        x1[i, :, bby1:bby2, bbx1:bbx2] = x2[i, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1[i].shape[1] * x1[i].shape[2]))
    y = lam * y1 + (1 - lam) * y2
    return x1, y
