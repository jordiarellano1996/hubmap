import os


class CFG:
    seed = 25
    debug = False  # set debug=False for Full Training
    wandb_callback_flag = True
    wandb_test_name = "no_augmentation_unet_filters_16"
    debug_cases = 3
    if len(os.getcwd().split("/")) > 4:
        base_path = "/home/titoare/Documents/ds/hubmap/kaggle/input/hubmap-organ-segmentation"
    else:
        base_path = os.path.expanduser('~') + "/project/hubmap-organ-segmentation"
    print(f"base_path: {base_path}")
    epochs_path = "/tmp/model"
    img_size = (512, 512, 3)
    batch_size = 20
    epochs = 40
    n_fold = 7


__version__ = "unknown"

" It is a list of strings defining what symbols in a module willbe exported when"
" from <module> import * is used on the module."

__all__ = [
    "CFG",
]
