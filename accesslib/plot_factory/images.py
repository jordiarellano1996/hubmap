import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


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


def show_img(img, mask=None, mask_labels: list = None, fig_size=(10, 10)):
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img = clahe.apply(img)
    plt.figure(figsize=fig_size)
    plt.imshow(img, cmap='bone', aspect='auto')

    if mask is not None:
        labels_len = len(mask_labels)
        colors_arr = [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        cmap1 = LinearSegmentedColormap.from_list("", ["red", "red"])
        cmap2 = LinearSegmentedColormap.from_list("", ["green", "green"])
        cmap3 = LinearSegmentedColormap.from_list("", ["blue", "blue"])
        cmap = [cmap1, cmap2, cmap3]

        for pos in range(labels_len):
            plt.imshow(mask[:, :, pos], alpha=0.5 * (mask[:, :, pos] / 255), cmap=cmap[pos], aspect='auto')

        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in colors_arr[0:labels_len]]
        plt.legend(handles, mask_labels, fontsize=fig_size[0])

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_to_img_dec(func):
    """ Retrieve a view on the renderer buffer to numpy rgba array. """

    def wrapper(*args, **kwargs):
        fig = func(*args, **kwargs)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()  # draw the canvas, cache the renderer
        # convert to a NumPy array
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.clf()
        return img

    return wrapper


@plot_to_img_dec
def show_img_canvas(img, mask=None, mask_labels: list = None, fig_size=(10, 10)):
    fig = plt.figure(figsize=fig_size)
    plt.imshow(img, cmap='bone', aspect='auto')

    if mask is not None:
        labels_len = len(mask_labels)
        colors_arr = [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        cmap1 = LinearSegmentedColormap.from_list("", ["red", "red"])
        cmap2 = LinearSegmentedColormap.from_list("", ["green", "green"])
        cmap3 = LinearSegmentedColormap.from_list("", ["blue", "blue"])
        cmap = [cmap1, cmap2, cmap3]

        for pos in range(labels_len):
            plt.imshow(mask[:, :, pos], alpha=0.5 * (mask[:, :, pos] / 255), cmap=cmap[pos], aspect='auto')

        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in colors_arr[0:labels_len]]
        plt.legend(handles, mask_labels, fontsize=fig_size[0])

    plt.axis('off')
    plt.tight_layout()
    return fig
