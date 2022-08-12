import os
import pandas as pd
import json
from accesslib import CFG
from importlib import reload
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np
from accesslib.plot_factory import images
from accesslib.plot_factory import wandb as wdb
from accesslib.segmentation_precompute import patches

reload(images)
reload(wdb)
reload(patches)


@images.plot_to_img_dec
def histogram(df, col_name, hue, legend=False):
    fig = plt.figure(figsize=(10, 10))
    ax = sns.histplot(df, x=col_name, hue=hue, legend=legend)
    for i in ax.containers:
        ax.bar_label(i, )

    return fig


@images.plot_to_img_dec
def scatter():
    fig = plt.figure(figsize=(20, 20))
    sns.scatterplot(data=train, x="img_width", y="img_height", size="size_count", hue="size_count", palette="tab10",
                    legend=False, alpha=0.5, sizes=(20, 2000))
    return fig


def plot_image(in_target_row):
    img = images.read_image(in_target_row["img_path"].values[0])
    mask = images.read_image(in_target_row["mask_path"].values[0])
    images.show_img(img, mask, [in_target_row["organ"].values[0]])


def get_contours(img, mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 0), 2)
        cv2.drawContours(img, [cnt], 0, (36, 255, 12), 4)

    cv2.imwrite('contours_cv.png', img)


# Get image cropped
def load_json_to_dict(json_path):
    """ tbd """
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data


def flatten_l_o_l(nested_list):
    """ Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]


def crop_ftu_polygons(img, cnts, pad=5):
    def get_crop_from_cnt(_img, _cnt, _pad):
        x1, y1, w, h = cv2.boundingRect(np.array(_cnt))
        x2, y2 = x1 + w + _pad, y1 + h + _pad
        x1, y1 = max(0, x1 - _pad), max(0, y1 - _pad)
        return _img[y1:y2, x1:x2]

    ftu_crops = [get_crop_from_cnt(img, cnt, pad) for cnt in cnts]
    return ftu_crops


@images.plot_to_img_dec
def plot_ftus(ftu_crops, organ="", n_cols=4, height_per_row=7, max_crops=50):
    ftu_crops = ftu_crops[:max_crops]
    n_rows = int(np.ceil(len(ftu_crops) / n_cols))

    fig = plt.figure(figsize=(20, n_rows * height_per_row))
    for i, ftu_crop in enumerate(ftu_crops):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(ftu_crop)
        plt.title(f"Organ: {organ}, FTU #{i + 1}  ––  SHAPE={ftu_crop.shape[:-1]}", fontweight="bold")
        plt.axis(False)

    plt.tight_layout()
    return fig
    # plt.show()


if __name__ == "__main__":
    cfg = CFG()

    """ 🤫 Load data set """
    train = pd.read_csv(os.path.join(cfg.base_path, "train_precompute.csv"))

    """ 🤫 Number of Segmentation Masks Per Organ Type """
    hist_img_org = histogram(train, "organ", "organ", legend=True)
    images.show_img(hist_img_org)
    hist_img_age = histogram(train, "age", "organ", legend=True)
    images.show_img(hist_img_age)

    """ 🤫 Pixel distribution """
    train["size_count"] = train.groupby(by="img_width")["id"].transform("count")
    train["size_count"].where(train["size_count"].values == 326, other=train["size_count"].values * 20, inplace=True)
    scatter_img = scatter()
    images.show_img(scatter_img)

    """ Tiff images show."""
    organ = 'lung'
    sub_df = train[train.organ == organ].sample(1)
    img = images.read_image(sub_df.iloc[[0, ]]["img_path"].values[0])
    mask = images.read_image(sub_df.iloc[[0, ]]["mask_path"].values[0])
    # get_contours(img, mask)
    images.show_img(img, mask=mask, mask_labels=[organ, ], fig_size=(20, 20))

    """ Tiff images show."""
    MAX_N_FTU = 8
    N_EX = 10
    ORGANS = ['kidney', 'largeintestine', 'lung', 'prostate', 'spleen']

    out_img = []
    for _organ in ORGANS:
        sub_df = train[train.organ == _organ].sample(N_EX)
        imgs = [images.read_image(sub_df.iloc[[pos, ]]["img_path"].values[0]) for pos in range(N_EX)]
        cnts = [load_json_to_dict(sub_df.iloc[[pos, ]]["lbl_path"].values[0]) for pos in range(N_EX)]  # image contour
        ftu_crops = flatten_l_o_l([crop_ftu_polygons(img, _cnts, pad=0) for img, _cnts in zip(imgs, cnts)])[:MAX_N_FTU]
        print(f"\n\n\n... DISPLAYING {len(ftu_crops)} {_organ} FTU CROPS ...\n")
        _img = plot_ftus(ftu_crops, organ=_organ)
        images.show_img(_img, fig_size=(30, 20))
        out_img.append(_img)

    del out_img, imgs, cnts, sub_df, ftu_crops, MAX_N_FTU, N_EX

    """ Check patches consistency."""
    patcher_img = patches.Patcher(img, verbose=True)
    out_patch_img = patcher_img.extract_patches(256, 256, step=256)
    reconstructed_img = patcher_img.reconstruct_from_patches(out_patch_img)
    patcher_mask = patches.Patcher(mask, verbose=True)
    out_patch_mask = patcher_mask.extract_patches(256, 256, step=256)
    reconstructed_mask = patcher_mask.reconstruct_from_patches(out_patch_mask)
