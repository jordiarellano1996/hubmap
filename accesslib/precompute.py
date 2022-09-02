import os
from glob import glob
import pandas as pd
import cv2
import numpy as np
from accesslib import CFG
from accesslib.segmentation_precompute.rle import RLE
from accesslib.segmentation_precompute.patches import ExtractPatchesFlat
from accesslib.segmentation_precompute.read_image import read_image
from tqdm import tqdm


def save_image_on_disk(image, path_scaled):
    dir_path = "/" + os.path.join(*path_scaled.split("/")[:-1])
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass
        # print(f"File exist")
    cv2.imwrite(path_scaled, image)


def patches_on_disk(path_values: list, path_patches_values: list, patch_size, mask_consistency_flag=False):
    patch_shapes = []
    new_shapes = []
    patches_path = []
    mask_consistency_index = []
    for pos in tqdm(range(len(path_values))):
        img = read_image(path_values[pos])
        patcher = ExtractPatchesFlat(img)
        img_patch, patch_shape = patcher.extract(patch_size[0], patch_size[1], step=patch_size[0])

        # Save shapes
        new_shape = patcher.img.shape
        patch_shapes.append(patch_shape)
        new_shapes.append(new_shape)

        # Save patches on disk
        i = 0
        patches_path_list = []
        mask_consistency_list = []
        for img_pos in range(len(img_patch)):
            path = path_patches_values[pos][:-5] + f"_{i}.png"
            patches_path_list.append(path)
            img = img_patch[img_pos]
            save_image_on_disk(img, path)
            if mask_consistency_flag:
                """ This logic is use to check if in the new tiles there is any segmentation marked."""
                flatten_mask = img.flatten()
                pct_up = (len(flatten_mask[flatten_mask == 255])/len(flatten_mask))*100  # percentage of pixels at 255.
                if pct_up > 5:
                    mask_consistency_list.append(img_pos)
            i += 1

        patches_path.append(patches_path_list)
        mask_consistency_index.append(mask_consistency_list)

    return patches_path, new_shapes, patch_shapes, mask_consistency_index


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

    """ 🤫 Generate patches for images """
    # Train images data
    patch_size = (500, 500)
    train.insert(5, "patches_path", train["img_path"])
    train["patches_path"].replace(["/train_images", ], ["/train_patches", ], regex=True, inplace=True)
    patches_path_out, new_shapes_out, patch_shapes_out, _ = patches_on_disk(train["img_path"].values,
                                                                            train["patches_path"].values, patch_size)
    train["patches_path"] = patches_path_out
    train["new_shape"] = new_shapes_out
    train["patch_shape"] = patch_shapes_out

    # Train masks data
    train.insert(5, "pmask_path", train["img_path"])
    train["pmask_path"].replace(["/train_images", ], ["/train_pmask", ], regex=True, inplace=True)
    pmask_path_out, _, _, mask_consistency_index = patches_on_disk(train["mask_path"].values,
                                                                   train["pmask_path"].values, patch_size,
                                                                   mask_consistency_flag=True)
    train["pmask_path"] = pmask_path_out
    train["pmask_consistency"] = mask_consistency_index

    # Test images data
    test.insert(5, "patches_path", test["img_path"])
    test["patches_path"].replace(["/test_images", ], ["/test_patches", ], regex=True, inplace=True)
    patches_path_out, new_shapes_out, patch_shapes_out, _ = patches_on_disk(test["img_path"].values,
                                                                            test["patches_path"].values, patch_size)
    test["patches_path"] = patches_path_out
    test["new_shape"] = new_shapes_out
    test["patch_shape"] = patch_shapes_out

    """ 🤫 Save custom train dataset """
    # train.to_csv(os.path.join(cfg.base_path, "train_precompute.csv"), index=False)
    # test.to_csv(os.path.join(cfg.base_path, "test_precompute.csv"), index=False)
    train.to_pickle(os.path.join(cfg.base_path, "train_precompute.csv"))
    test.to_pickle(os.path.join(cfg.base_path, "test_precompute.csv"))
