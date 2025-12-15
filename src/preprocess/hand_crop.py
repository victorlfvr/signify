import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

mp_hands = mp.solutions.hands


def crop_hand(pil_img, enlarge_ratio=1.6):
    img = np.array(pil_img)
    h, w = img.shape[:2]

    img_rgb = img.copy()

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return pil_img 

        hand = results.multi_hand_landmarks[0]

        xs = [lm.x for lm in hand.landmark]
        ys = [lm.y for lm in hand.landmark]

        xmin = int(min(xs) * w)
        xmax = int(max(xs) * w)
        ymin = int(min(ys) * h)
        ymax = int(max(ys) * h)

        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        bw = int((xmax - xmin) * enlarge_ratio)
        bh = int((ymax - ymin) * enlarge_ratio)

        side = max(bw, bh)
        half = side // 2

        xmin = max(0, cx - half)
        xmax = min(w, cx + half)
        ymin = max(0, cy - half)
        ymax = min(h, cy + half)

        crop = img[ymin:ymax, xmin:xmax]

        if crop.size == 0:
            return pil_img

        return Image.fromarray(crop)

def equalize(pil_img):
    img = np.array(pil_img)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    img_eq = img_eq.astype(np.float32)
    img_eq = 0.90 * img_eq + 0.10 * img.astype(np.float32) 
    img_eq = np.clip(img_eq, 0, 255).astype(np.uint8)

    return Image.fromarray(img_eq)
