import os
import pandas as pd
from accesslib import CFG
from accesslib import data_loader
from importlib import reload
from accesslib.plot_factory.images import show_img, show_img_result
import numpy as np
from accesslib.segmentation_precompute.read_image import read_image
from accesslib.model.custom_metrics import bce_dice_loss, dice_coef, iou_coef, jacard_coef, bce_loss, bce_coef
from keras.models import load_model

reload(data_loader)

if __name__ == "__main__":
    cfg = CFG()

    # 🚀 Load data set
    df = pd.read_pickle(os.path.join(cfg.base_path, "train_precompute.csv"))
    print(df.organ.unique())
    df = df[df.organ == "largeintestine"]
    df = df.iloc[np.random.randint(len(df), size=(10,))]

    # 🚀 Get image and mask paths
    img_paths = df.img_path.values
    mask_paths = df.mask_path.values

    #  🚀 Store images and mask on ram.
    img_arr = []
    mask_arr = []
    for pos in range(len(img_paths[0:100])):
        img_arr.append(read_image(img_paths[pos]))
        mask_arr.append(read_image(mask_paths[pos]))

    # 🚀 Create data generator
    gen = data_loader.DataGenerator(img_arr, mask_arr, batch_size=1, shuffle=True, augment=False, crops=1,
                                    size=cfg.img_size[0], size2=cfg.img_size[0])

    # 🚀 Load model
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
        "/home/titoare/Documents/ds/hubmap/models/2_bce_dice_640x640_Aug_unet_f16_0.36033_0.32320",
        custom_objects=custom_objects)

    # 🚀 Evaluate
    for i in range(10):
        x, y = gen[i]
        pred_mask = out_model.predict(x)
        bce_coef_v = np.mean(bce_coef(y, pred_mask))
        dice_coef_v = np.mean(dice_coef(y, pred_mask))
        iou_coef_v = np.mean(iou_coef(y, pred_mask))
        bce_dice_loss_v = np.mean(bce_dice_loss(y, pred_mask))
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: ", np.max(pred_mask), np.max(y))
        pred_mask = np.uint8(pred_mask > 0.5)
        # Plot
        show_img_result((x[0] * 255).astype('uint8'),
                        (y[0] * 255).astype('uint8'),
                        (pred_mask[0] * 255).astype('uint8'),
                        bce_dice_loss_v,
                        bce_coef_v,
                        iou_coef_v,
                        dice_coef_v)
