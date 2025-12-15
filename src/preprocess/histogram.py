import cv2
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image

class HistogramEqualization:
    def __init__(self, clip=2.0, tile=(8, 8)):
        self.clip = clip
        self.tile = tile

    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip, tileGridSize=self.tile)

        arr = np.array(img)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)

        L, A, B = cv2.split(arr)
        L = clahe.apply(L)

        arr = cv2.merge([L, A, B])
        arr = cv2.cvtColor(arr, cv2.COLOR_LAB2RGB)

        return Image.fromarray(arr)

