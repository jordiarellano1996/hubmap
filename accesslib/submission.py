import os
import pandas as pd
import numpy as np
import cv2


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


if __name__ == "__main__":
    # Open the training dataframe and display the initial dataframe
    DATA_DIR = "/kaggle/input/hubmap-organ-segmentation"
    DATA_DIR = "/home/titoare/Documents/ds/hubmap/kaggle/input/hubmap-organ-segmentation"

    # Open the training dataframe and display the initial dataframe
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    test_df = pd.read_csv(TEST_CSV)

    SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
    ss_df = pd.read_csv(SS_CSV)

    # Rando circle data
    SMART_CIRCLE_FRAC = 0.8
    rles = []
    for i, (img_w, img_h) in enumerate(zip(test_df["img_width"], test_df["img_height"])):
        tmp_img = np.zeros((img_w, img_h))
        tmp_img = cv2.circle(tmp_img, (int(np.round(img_w / 2)), int(np.round(img_h / 2))),
                             int(np.round((img_w / 2) * SMART_CIRCLE_FRAC)), 1, -1)
        rles.append(rle_encode(tmp_img))

    ss_df["rle"] = rles

    # 5. Reduce number of columns (just in case) and save
    ss_df = ss_df[["id", "rle"]]
    ss_df.to_csv("submission.csv", index=False)

