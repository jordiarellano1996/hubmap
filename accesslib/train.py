import sys
import os

sys_path = os.path.expanduser('~') + "/project/hubmap"
print(sys_path)
sys.path.append(sys_path)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from accesslib import CFG
from accesslib.data_loader import DataGenerator
from accesslib.model.unet import Unet, Model, HalfUnet
from accesslib.model.callbacks import create_callbacks
from accesslib.model.custom_metrics import bce_dice_loss, dice_coef, iou_coef, jacard_coef, bce_loss
from accesslib.model.gpu import configure_gpu_memory_allocation, print_devices
from accesslib.segmentation_precompute.read_image import read_img_from_disk

"""0: Debug, 1: No Info, 2: No info/warnings, 3: No info/warnings/error logged."""
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def replace_path(str_path_list, old_paths, new_path):
    return [path.replace(old_paths, new_path) for path in str_path_list]


if __name__ == "__main__":
    cfg = CFG()

    # 🚀 Config GPU memory allocation
    print_devices()

    # 🚀 Load data set
    df = pd.read_pickle(os.path.join(cfg.base_path, "train_precompute.csv"))
    if cfg.debug:
        df = df.iloc[np.random.randint(350, size=(cfg.debug_cases,))]

    # 🚀 Cross validation
    """
    - This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by
      preserving the percentage of samples for each class.
    """
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    cross_index = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['organ'], groups=None)):
        cross_index.append((train_idx, val_idx))

    del fold, train_idx, val_idx, skf

    FOLD = 0
    # train = df.iloc[cross_index[0][0]]
    # validation = df.iloc[cross_index[0][1]]
    train_id = df.id.unique()[:-2]
    val_id = df.id.unique()[-2:]
    train = df[df['id'].isin(train_id)]
    validation = df[df['id'].isin(val_id)]

    # 🚀 Get patches paths
    train_img_paths = train.img_path.values
    train_mask_paths = train.mask_path.values
    val_img_paths = validation.img_path.values
    val_mask_paths = validation.mask_path.values

    # 🚀 Change directory
    print(f"The actual path is: {cfg.base_path}")
    old_path = "/home/titoare/Documents/ds/hubmap/kaggle/input/hubmap-organ-segmentation"
    train_img_paths = replace_path(train_img_paths, old_path, cfg.base_path)
    train_mask_paths = replace_path(train_mask_paths, old_path, cfg.base_path)
    val_img_paths = replace_path(val_img_paths, old_path, cfg.base_path)
    val_mask_paths = replace_path(val_mask_paths, old_path, cfg.base_path)

    #  🚀 Store images and mask on ram.
    train_img, train_mask = read_img_from_disk(train_img_paths, train_mask_paths)
    val_img, val_mask = read_img_from_disk(val_img_paths, val_mask_paths)

    # 🚀 Getting generators
    train_gen = DataGenerator(train_img, train_mask, batch_size=cfg.batch_size, shuffle=True, augment=False,
                              crops=cfg.crops, size=cfg.img_size[0], size2=cfg.img_size[0], shrink=1)
    val_gen = DataGenerator(val_img, val_mask, batch_size=cfg.batch_size, shuffle=True, augment=False,
                            crops=cfg.crops, size=cfg.img_size[0], size2=cfg.img_size[0], shrink=1)

    # 🚀 Train
    input_layer, output_layer = Unet(img_shape=cfg.img_size, filters=16, drop_out=0.).get_layers()
    model = Model(input_layer, output_layer, loss="binary_crossentropy",
                  metrics=['accuracy', dice_coef, iou_coef, jacard_coef, bce_loss], verbose=True,
                  learning_rate=cfg.learning_rate).get_model()

    wandb_config = {'competition': "HuBMAP", 'GPU_name': cfg.GPU_name, "batch_size": cfg.batch_size}
    callbacks = create_callbacks(cfg.epochs_path,
                                 wandb_flag=cfg.wandb_callback_flag,
                                 wandb_test_name=cfg.wandb_test_name,
                                 wandb_config=wandb_config)
    history = model.fit(
        train_gen,
        steps_per_epoch=(len(train_img_paths) * cfg.crops) // cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=(len(val_img_paths) * cfg.crops) // cfg.batch_size,
    )

    model.save(os.path.join(cfg.epochs_path, 'complete_model'))
