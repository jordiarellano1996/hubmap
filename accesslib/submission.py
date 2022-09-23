import os
from glob import glob
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
from accesslib import CFG
from accesslib.segmentation_precompute.read_image import read_image, img_rescale
from accesslib.segmentation_precompute.patches import reconstruct_from_patches, ExtractPatchesFlat
from accesslib.model.custom_metrics import bce_dice_loss, dice_coef, iou_coef, jacard_coef, bce_coef
from keras.models import load_model


def rle_encode(img):
    """ TBD
    Args:
        img (np.array):
            - 1 indicating mask
            - 0 indicating background

    Returns:
        run length as string formated
    """

    img = img.T
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def replace_path(str_path_list, old_paths, new_path):
    return [path.replace(old_paths, new_path) for path in str_path_list]


def save_image_on_disk(image, path_scaled):
    dir_path = "/" + os.path.join(*path_scaled.split("/")[:-1])
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass
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
                pct_up = (len(flatten_mask[flatten_mask == 255]) / len(
                    flatten_mask)) * 100  # percentage of pixels at 255.
                if pct_up > 5:
                    mask_consistency_list.append(img_pos)
            i += 1

        patches_path.append(patches_path_list)
        mask_consistency_index.append(mask_consistency_list)

    return patches_path, new_shapes, patch_shapes, mask_consistency_index


if __name__ == "__main__":
    cfg = CFG()
    # df = pd.read_pickle(os.path.join(cfg.base_path, "train_precompute.csv"))
    # v = np.random.randint(0, 300, size=(15,))
    # df = df.iloc[v]

    """ 🤫 Generate patches for images """
    DATA_DIR = cfg.base_path
    KAGGLE_WORKING_DIR = cfg.epochs_path
    MODEL_PATH = "/home/titoare/Documents/ds/hubmap/models/2_bce_dice_640x640_Aug_unet_f16_0.36033_0.32320"

    # Open the training dataframe and display the initial dataframe
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    test_df = pd.read_csv(TEST_CSV)

    SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
    ss_df = pd.read_csv(SS_CSV)

    # Get images
    all_test_images = glob(os.path.join(TEST_IMAGES_DIR, "*.tiff"), recursive=True)
    test_img_map = {int(x[:-5].rsplit("/", 1)[-1]): x for x in all_test_images}
    test_df.insert(3, "img_path", test_df["id"].map(test_img_map))

    """ 🤫 Generate patches for images """
    # Train images data
    patch_size = (640, 640)

    # Test images data

    test_df.insert(5, "patches_path", test_df["img_path"])
    test_df["patches_path"].replace([os.path.join(cfg.base_path, "test_images"), ],
                                    [os.path.join(KAGGLE_WORKING_DIR, "test_patches"), ], regex=True, inplace=True)
    patches_path_out, new_shapes_out, patch_shapes_out, _ = patches_on_disk(test_df["img_path"].values,
                                                                            test_df["patches_path"].values, patch_size)
    test_df["patches_path"] = patches_path_out
    test_df["new_shape"] = new_shapes_out
    test_df["patch_shape"] = patch_shapes_out

    """ 🤫 Save custom train dataset """
    test_df.to_pickle(os.path.join(KAGGLE_WORKING_DIR, "test_precompute.csv"))

    ### 🚀 Load model
    """ if you use custom metrics to evaluate the model, when you loaded you must pass
     custom objects instead you will get an error."""
    custom_objects = custom_objects = {
        'bce_dice_loss': bce_dice_loss,
        'dice_coef': dice_coef,
        'iou_coef': iou_coef,
        "jacard_coef": jacard_coef,
        "bce_coef": bce_coef,
    }
    out_model = load_model(
        MODEL_PATH,
        custom_objects=custom_objects)

    # 🚀 Start loop
    IMAGE_SIZE = (640, 640, 3)
    t = []
    rles = []
    for pos in tqdm(range(len(test_df))):
        df_line = test_df.iloc[[pos, ]]

        # 🚀 Change directory
        # old_path = "/home/titoare/Documents/ds/hubmap/kaggle/input/hubmap-organ-segmentation"
        # patch_paths = replace_path(df_line.patches_path.values[0], old_path, cfg.base_path)
        patch_paths = df_line.patches_path.values[0]
        complete_img = read_image(df_line.img_path.values[0])

        # 🚀 Generate data
        x = np.zeros((len(patch_paths), IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]), dtype='float32')

        k = 0
        for index in range(len(patch_paths)):
            # Read
            img = read_image(patch_paths[index])
            # Rescale
            img = img_rescale(img)
            # Save
            x[k] = img
            k += 1

        # 🚀 Prediction
        pred_masks = out_model.predict(x)
        pred_masks = np.uint8(pred_masks > 0.5)
        print(pred_masks.shape)

        # 🚀 Com
        """ Replace last tuple value with 1"""
        patch_shape = tuple(list(df_line.patch_shape.values[0])[:-1] + [1])
        new_shape = tuple(list(df_line.new_shape.values[0])[:-1] + [1])
        print(patch_shape)
        print(new_shape)
        mask_margins = reconstruct_from_patches(pred_masks.reshape(patch_shape), new_shape)
        mask = mask_margins[:df_line.img_height.values[0], :df_line.img_width.values[0], :]  # y,x, z
        rles.append(rle_encode(mask))

    from accesslib.plot_factory.images import show_img
    show_img((complete_img).astype('uint8'), (mask * 255).astype('uint8'), mask_labels=[f"a"])


    # 5. Reduce number of columns (just in case) and save
    # ss_df = ss_df[["id", "rle"]]
    # ss_df.to_csv("submission.csv", index=False)
