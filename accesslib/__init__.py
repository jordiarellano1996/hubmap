import os

"""0: Debug, 1: No Info, 2: No info/warnings, 3: No info/warnings/error logged."""
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class CFG:
    seed = 2021
    debug = True  # set debug=False for Full Training
    debug_cases = False
    wandb_callback_flag = True
    wandb_test_name = "Aug_unet_f16_do0.0"
    GPU_name = '1x NVIDIA RTX A6000'
    if len(os.getcwd().split("/")) > 4:
        base_path = "/home/titoare/Documents/ds/hubmap/kaggle/input/hubmap-organ-segmentation"
    else:
        base_path = os.path.expanduser('~') + "/project/hubmap-organ-segmentation"
    print(f"base_path: {base_path}")
    epochs_path = "/tmp/model"
    img_size = (512, 512, 3)
    crops = 50  # How many random crops for each image.
    batch_size = 64
    epochs = 50
    learning_rate = 0.001


__version__ = "unknown"

" It is a list of strings defining what symbols in a module willbe exported when"
" from <module> import * is used on the module."

__all__ = [
    "CFG",
]
