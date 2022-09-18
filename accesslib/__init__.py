import os


class CFG:
    seed = 25
    debug = True  # set debug=False for Full Training
    debug_cases = 15
    wandb_callback_flag = True
    wandb_test_name = "try_overtfit_newdataloader_Aug_unet_f16_do0.0"
    GPU_name = '1x NVIDIA RTX A6000'
    if len(os.getcwd().split("/")) > 4:
        base_path = "/home/titoare/Documents/ds/hubmap/kaggle/input/hubmap-organ-segmentation"
    else:
        base_path = os.path.expanduser('~') + "/project/hubmap-organ-segmentation"
    print(f"base_path: {base_path}")
    epochs_path = "/tmp/model"
    img_size = (512, 512, 3)
    crops = 30  # How many random crops for each image.
    batch_size = 32
    epochs = 30
    learning_rate = 0.0001
    n_fold = 2


__version__ = "unknown"

" It is a list of strings defining what symbols in a module willbe exported when"
" from <module> import * is used on the module."

__all__ = [
    "CFG",
]
