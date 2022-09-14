import os
import pandas as pd
from accesslib import CFG
from accesslib.data_loader import DataGenerator
from accesslib.plot_factory.images import show_img

if __name__ == "__main__":
    cfg = CFG()

    # 🚀 Load data set
    df = pd.read_pickle(os.path.join(cfg.base_path, "train_precompute.csv"))

    # 🚀 Get image and mask paths
    img_paths = df.img_path.values
    mask_paths = df.mask_path.values

    # 🚀 Create data generator
    gen = DataGenerator(img_paths, mask_paths, batch_size=32, shuffle=True, augment=True, crops=100, size=512,
                        size2=512, shrink=1)

    # Evaluate
    x, y = gen[5]
    for i in range(32):
        show_img((x[i] * 255).astype('uint8'), (y[i] * 255).astype('uint8'), mask_labels=[f"{i}"])
