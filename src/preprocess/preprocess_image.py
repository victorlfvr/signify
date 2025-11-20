# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

mp_hands = mp.solutions.hands

# ---------------------------------------------------------
# 1) Génération du masque main via Mediapipe Hands
# ---------------------------------------------------------
def hand_mask(img_cv):
    """
    Retourne un masque binaire (255 = main, 0 = fond)
    basé sur les landmarks MediaPipe.
    """
    h, w, _ = img_cv.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.15
    ) as hands:

        res = hands.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None  # pas de main détectée

        hand = res.multi_hand_landmarks[0]

        # Polygone approximé = convex hull des landmarks
        points = np.array([
            [int(lm.x * w), int(lm.y * h)]
            for lm in hand.landmark
        ])

        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

    return mask


# ---------------------------------------------------------
# 2) Applique le masque → fond noir
# ---------------------------------------------------------
def apply_mask(img_cv, mask):
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img_cv, mask_3)


# ---------------------------------------------------------
# 3) Crop autour de la main
# ---------------------------------------------------------
def crop_to_hand(mask, img_cv, margin=20):
    ys, xs = np.where(mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        return img_cv  # fallback

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img_cv.shape[1], x2 + margin)
    y2 = min(img_cv.shape[0], y2 + margin)

    crop = img_cv[y1:y2, x1:x2]
    return crop


# ---------------------------------------------------------
# 4) Pipeline complet pour dataset et inference
# ---------------------------------------------------------
def preprocess_image(img_pil, size=96):
    """
    Entrée: PIL Image
    Sortie: PIL Image 96×96, fond noir, main centrée
    """

    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    mask = hand_mask(img_cv)

    if mask is None:
        # fallback : resize simple (évite de jeter l'image)
        img_cv = cv2.resize(img_cv, (size, size))
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # Fond noir
    masked = apply_mask(img_cv, mask)
    cropped = crop_to_hand(mask, masked)

    # Redimension
    resized = cv2.resize(cropped, (size, size))

    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
