import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from accesslib.segmentation_precompute.read_image import read_image


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


def check_patches(patches_paths, mask_patches_paths, xy_shape=(6, 6), fig_size=(20, 20)):
    cmap = LinearSegmentedColormap.from_list("", ["red", "red"])

    f, axarr = plt.subplots(xy_shape[0], xy_shape[1], figsize=fig_size)
    i = 0
    for x in range(xy_shape[0]):
        for y in range(xy_shape[1]):
            img = read_image(patches_paths[i])
            mask = read_image(mask_patches_paths[i])
            axarr[x, y].imshow(img)
            axarr[x, y].imshow(mask[:, :, 0], alpha=0.5 * (mask[:, :, 0] / 255), cmap=cmap, aspect='auto')
            axarr[x, y].axis('off')
            i += 1
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_img_result(img, mask, pred_mask, bce_dice_loss, bce_coef, iou_coef, dice_coef, fig_size=(20, 10)):
    fig, axs = plt.subplots(1, 3, figsize=fig_size)
    axs = axs.flatten()
    # Config title
    print(bce_dice_loss, bce_coef, iou_coef, dice_coef)
    title_arr = ["Input image", "Image & True_Mask", "Image & Predicted_Mask"]
    axs[0].set_title(title_arr[0], fontsize=14, weight='bold')
    axs[1].set_title(title_arr[1], fontsize=14, weight='bold')
    axs[2].set_title(title_arr[
                         2] + f"\nbce_dice_loss: {bce_dice_loss:.4}, bce_coef: {bce_coef:.4}\niou_coef: {iou_coef:.4},dice_coef: {dice_coef:.4}",
                     fontsize=14, weight='bold')

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # img = clahe.apply(img)
    axs[0].imshow(img, cmap='bone')
    axs[0].axis('off')

    # True mask plot
    i = 1
    for m in [mask, pred_mask]:
        axs[i].imshow(img, cmap='bone')
        axs[i].axis('off')

        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["mask", ]
        axs[i].legend(handles, labels)
        axs[i].imshow(m, alpha=0.5 * (m[:, :, 0] / 255), cmap="bwr")
        i += 1

    plt.tight_layout()
    plt.show()
