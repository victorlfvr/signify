# -*- coding: utf-8 -*-
import cv2
from PIL import Image
from src.preprocess.preprocess_image import preprocess_image
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Caméra introuvable")

    print("Appuie sur Q pour quitter")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # PIL → preprocess
        img_pil = Image.fromarray(frame_rgb)
        out_pil = preprocess_image(img_pil)

        # Retour CV2
        out_cv = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Original", frame)
        cv2.imshow("Processed Hand", out_cv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
