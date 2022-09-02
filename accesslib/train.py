import os
import pandas as pd
from accesslib import CFG
from accesslib.data_loader import DataGenerator
from accesslib.model.unet import create_callbacks, Unet, Model, HalfUnet
from accesslib.model.custom_metrics import bce_dice_loss, dice_coef, iou_coef, jacard_coef, bce_loss

if __name__ == "__main__":
    cfg = CFG()

    """ 🤫 Load data set """
    train = pd.read_pickle(os.path.join(cfg.base_path, "train_precompute.csv"))

    """ 🤫 Get patches paths"""
    patch_paths = []
    pmask_paths = []
    for pos in range(len(train.patches_path)):
        for p_index in train.pmask_consistency.values[pos]:
            patch_paths.append(train.patches_path.values[pos][p_index])
            pmask_paths.append(train.pmask_path.values[pos][p_index])

    """ Getting generators"""
    gen = DataGenerator(patch_paths, pmask_paths, batch_size=cfg.batch_size, shuffle=True, augment=False, )
    #
    # 🚀 Train
    input_layer, output_layer = HalfUnet(img_shape=cfg.img_size).get_layers()
    model = Model(input_layer, output_layer, loss=bce_dice_loss, metrics=[dice_coef, iou_coef, jacard_coef, bce_loss],
                  verbose=True).get_model()

    # callbacks = create_callbacks(cfg.epochs_path, )
    # history = model.fit(
    #     gen,
    #     steps_per_epoch=len(patch_paths) // cfg.batch_size,
    #     epochs=cfg.epochs,
    #     callbacks=callbacks,
    # )
    #
    # model.save(os.path.join(cfg.epochs_path, 'complete_model'))
