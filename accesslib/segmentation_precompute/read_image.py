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


def read_img_from_disk(img_paths: list, mask_paths: list) -> tuple:
    """ Read images and move to RAM """
    assert len(img_paths) == len(mask_paths)

    img_arr = []
    mask_arr = []
    for pos in range(len(img_paths)):
        img_arr.append(read_image(img_paths[pos]))
        mask_arr.append(read_image(mask_paths[pos]))
    return img_arr, mask_arr
