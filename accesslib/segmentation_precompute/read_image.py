import numpy as np
import cv2


def read_image(path):
    """
    Reads and converts the image.
    path: the full complete path to the .png file
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
    # Scale to [0, 255]
    if np.max(img) != np.min(img):
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img


def img_rescale(img):
    return (img * (1. / 255)).astype('float32')
