import os
from glob import glob
import numpy as np
import pandas as pd
from accesslib import CFG
from tqdm import tqdm
from accesslib.segmentation_precompute.rle import RLE

if __name__ == "__main__":
    cfg = CFG()
    TRAIN_IMAGES_DIR = os.path.join(cfg.base_path, "train_images")
    TRAIN_ANNOTATIONS_DIR = os.path.join(cfg.base_path, "train_annotations")
    TEST_IMAGES_DIR = os.path.join(cfg.base_path, "test_images")

    """ 🤫 Get img, annotations paths. """
    train = pd.read_csv(os.path.join(cfg.base_path, "train.csv"))
    test = pd.read_csv(os.path.join(cfg.base_path, "test.csv"))

    all_train_images = glob(os.path.join(TRAIN_IMAGES_DIR, "*.tiff"), recursive=True)
    all_train_labels = glob(os.path.join(TRAIN_ANNOTATIONS_DIR, "*.json"), recursive=True)
    all_test_images = glob(os.path.join(TEST_IMAGES_DIR, "*.tiff"), recursive=True)
    train_img_map = {int(x[:-5].rsplit("/", 1)[-1]): x for x in all_train_images}
    train_lbl_map = {int(x[:-5].rsplit("/", 1)[-1]): x for x in all_train_labels}
    test_img_map = {int(x[:-5].rsplit("/", 1)[-1]): x for x in all_test_images}
    train.insert(3, "img_path", train["id"].map(train_img_map))
    train.insert(4, "lbl_path", train["id"].map(train_lbl_map))
    test.insert(3, "img_path", test["id"].map(test_img_map))

    """ 🤫 Create mask paths. """
    # Train
    train.insert(4, "mask_path", train["img_path"])
    train["mask_path"].replace(["/train_images", ".tiff"], ["/train_masks", ".png"], regex=True, inplace=True)

    """ 🤫 Sex & age transformation in only train set."""
    sex_2_int = {"Male": 0, "Female": 1}
    age_divisor = train.age.max()

    train["sex"] = train["sex"].map(sex_2_int)
    train["age"] = train["age"] / age_divisor

    """ 🤫 Rle mask"""
    # rle = RLE(verbose=False)
    # for row_pos in tqdm(range(train.shape[0])):
    #     target_row = train.iloc[[row_pos, ]]
    #     segmentation = target_row["rle"].squeeze()
    #     img_height = target_row["img_height"].values[0]
    #     img_width = target_row["img_width"].values[0]
    #     mask = rle.mask_from_segmentation(segmentation, (img_height, img_width))
    #     # save mask
    #     rle.save_image_on_disk(np.uint8(mask * 255), target_row["mask_path"].values[0])

    """ 🤫 Save custom train dataset """
    train.to_csv(os.path.join(cfg.base_path, "train_precompute.csv"), index=False)
