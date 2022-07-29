import os
import pandas as pd
from accesslib import CFG
from importlib import reload
import matplotlib.pyplot as plt
import seaborn as sns
from accesslib.plot_factory import images

reload(images)

if __name__ == "__main__":
    cfg = CFG()

    """ 🤫 Load data set """
    train = pd.read_csv(os.path.join(cfg.base_path, "train_precompute.csv"))

    """ 🤫 Number of Segmentation Masks Per Organ Type """
    ax = sns.histplot(train, x="organ", hue="organ", legend=False)
    for i in ax.containers:
        ax.bar_label(i, )
    plt.show()


    """ Tiff images show."""
    # target_row = train.iloc[[27, ]]
    # img = images.read_image(target_row["img_path"].values[0])
    # mask = images.read_image(target_row["mask_path"].values[0])
    # images.show_img(img, mask, [target_row["organ"].values[0]])
