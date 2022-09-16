import os
import pandas as pd
from accesslib import CFG
from accesslib import data_loader
from importlib import reload
from accesslib.plot_factory.images import show_img
from time import time
from accesslib.segmentation_precompute.read_image import read_image
reload(data_loader)

if __name__ == "__main__":
    cfg = CFG()

    # 🚀 Load data set
    df = pd.read_pickle(os.path.join(cfg.base_path, "train_precompute.csv"))

    # 🚀 Get image and mask paths
    img_paths = df.img_path.values
    mask_paths = df.mask_path.values

    #  🚀 Store images and mask on ram.
    t1 = time()
    img_arr = []
    mask_arr = []
    for pos in range(len(img_paths[0:100])):
        img_arr.append(read_image(img_paths[pos]))
        mask_arr.append(read_image(mask_paths[pos]))
    img_arr = img_arr
    mask_arr = mask_arr
    print(time()-t1)

    # 🚀 Create data generator
    gen = data_loader.DataGenerator(img_arr, mask_arr, batch_size=64, shuffle=True, augment=True, crops=100, size=512,
                        size2=512, shrink=1)

    # Evaluate
    t1 = time()
    x, y = gen[5]
    print(time() - t1)
    for i in range(32):
        show_img((x[i] * 255).astype('uint8'), (y[i] * 255).astype('uint8'), mask_labels=[f"{i}"])
