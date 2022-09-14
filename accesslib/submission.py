import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from accesslib import CFG
from accesslib.segmentation_precompute.read_image import read_image, img_rescale
from accesslib.segmentation_precompute.patches import reconstruct_from_patches


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


if __name__ == "__main__":
    cfg = CFG()
    df = pd.read_pickle(os.path.join(cfg.base_path, "train_precompute.csv"))
    v = np.random.randint(0, 300, size=(15,))
    df = df.iloc[v]

    # 🚀 Start loop
    IMAGE_SIZE = (512, 512, 3)
    rles = []
    for pos in tqdm(range(len(df))):
        df_line = df.iloc[[pos, ]]

        # 🚀 Change directory
        old_path = "/home/titoare/Documents/ds/hubmap/kaggle/input/hubmap-organ-segmentation"
        patch_paths = replace_path(df_line.patches_path.values[0], old_path, cfg.base_path)

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
        y = x

        # 🚀 Com
        """ Replace last tuple value with 1"""
        # patch_shape = tuple(list(df_line.patch_shape.values[0])[:-1] + [1])
        # new_shape = tuple(list(df_line.new_shape.values[0])[:-1] + [1])
        patch_shape = df_line.patch_shape.values[0]
        new_shape = df_line.new_shape.values[0]
        mask_margins = reconstruct_from_patches(y.reshape(patch_shape), new_shape)
        mask = mask_margins[:df_line.img_height.values[0], :df_line.img_width.values[0], :]     # y,x, z
        rles.append(rle_encode(mask))





