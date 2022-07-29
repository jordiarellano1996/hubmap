import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap


def read_image(path):
    """
    Reads and converts the image.
    path: the full complete path to the .png file
    """

    # Read image in a corresponding manner
    # convert int16 -> float32
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')
    # Scale to [0, 255]
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = image.astype(np.uint8)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    return image


def show_img(img, mask=None, mask_labels=None):
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img = clahe.apply(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        labels_len = len(mask_labels)
        colors_arr = [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        cmap1 = LinearSegmentedColormap.from_list("", ["red", "red"])
        cmap2 = LinearSegmentedColormap.from_list("", ["green", "green"])
        cmap3 = LinearSegmentedColormap.from_list("", ["blue", "blue"])
        cmap = [cmap1, cmap2, cmap3]

        for pos in range(labels_len):
            plt.imshow(mask[:, :, pos], alpha=0.5 * (mask[:, :, pos] / 255), cmap=cmap[pos])

        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in colors_arr[0:labels_len]]
        plt.legend(handles, mask_labels)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
