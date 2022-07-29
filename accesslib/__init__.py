import os


class CFG:
    seed = 25
    debug = False  # set debug=False for Full Training
    wandb_test_name = "test_bce_loss"
    debug_cases = 3
    if len(os.getcwd().split("/")) > 4:
        base_path = "/home/titoare/Documents/ds/hubmap/kaggle/input/hubmap-organ-segmentation"
    else:
        base_path = "/mounted/input"
    print(f"base_path: {base_path}")
    epochs_path = base_path + "/models"
    img_size = (320, 384)
    batch_size = 64
    epochs = 60
    n_fold = 5


__version__ = "unknown"

" It is a list of strings defining what symbols in a module willbe exported when"
" from <module> import * is used on the module."

__all__ = [
    "CFG",
]
