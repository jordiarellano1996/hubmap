import sys
import os

sys_path = os.path.expanduser('~') + "/project/hubmap"
print(sys_path)
sys.path.append(sys_path)

import pandas as pd
from accesslib import CFG
from accesslib.data_loader import DataGenerator
from accesslib.model.unet import Unet, Model, HalfUnet
from accesslib.model.callbacks import create_callbacks
from accesslib.model.custom_metrics import bce_dice_loss, dice_coef, iou_coef, jacard_coef, bce_loss
from accesslib.model.gpu import configure_gpu_memory_allocation, print_devices
from sklearn.model_selection import StratifiedKFold

"""0: Debug, 1: No Info, 2: No info/warnings, 3: No info/warnings/error logged."""
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def get_patches(df_in):
    patch_paths = []
    pmask_paths = []
    for pos in range(len(df_in.patches_path)):
        for p_index in df_in.pmask_consistency.values[pos]:
            patch_paths.append(df_in.patches_path.values[pos][p_index])
            pmask_paths.append(df_in.pmask_path.values[pos][p_index])

    return patch_paths, pmask_paths


def replace_path(str_path_list, old_paths, new_path):
    return [path.replace(old_paths, new_path) for path in str_path_list]


if __name__ == "__main__":
    cfg = CFG()

    # 🚀 Config GPU memory allocation
    print_devices()

    # 🚀 Load data set
    df = pd.read_pickle(os.path.join(cfg.base_path, "train_precompute.csv"))

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
    train = df.iloc[cross_index[0][0]]
    validation = df.iloc[cross_index[0][1]]

    # 🚀 Get patches paths
    train_patch_paths, train_pmask_paths = get_patches(train)
    val_patch_paths, val_train_pmask_paths = get_patches(validation)

    # 🚀 Change directory
    print(f"The actual path is: {cfg.base_path}")
    old_path = "/home/titoare/Documents/ds/hubmap/kaggle/input/hubmap-organ-segmentation"
    train_patch_paths = replace_path(train_patch_paths, old_path, cfg.base_path)
    train_pmask_paths = replace_path(train_pmask_paths, old_path, cfg.base_path)
    val_patch_paths = replace_path(val_patch_paths, old_path, cfg.base_path)
    val_train_pmask_paths = replace_path(val_train_pmask_paths, old_path, cfg.base_path)

    # 🚀 Getting generators
    train_gen = DataGenerator(train_patch_paths, train_pmask_paths, img_size=cfg.img_size, batch_size=cfg.batch_size,
                              shuffle=True, augment=False, )
    val_gen = DataGenerator(val_patch_paths, val_train_pmask_paths, img_size=cfg.img_size, batch_size=cfg.batch_size,
                            shuffle=True, augment=False, )

    # 🚀 Train
    input_layer, output_layer = HalfUnet(img_shape=cfg.img_size).get_layers()
    model = Model(input_layer, output_layer, loss=bce_dice_loss, metrics=[dice_coef, iou_coef, jacard_coef, bce_loss],
                  verbose=True).get_model()

    callbacks = create_callbacks(cfg.epochs_path,
                                 wandb_flag=cfg.wandb_callback_flag,
                                 wandb_test_name=cfg.wandb_test_name,
                                 wandb_batch_size=cfg.batch_size)
    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_patch_paths) // cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=len(val_patch_paths) // cfg.batch_size,
    )

    model.save(os.path.join(cfg.epochs_path, 'complete_model'))
