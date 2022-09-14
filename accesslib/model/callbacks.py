import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

import wandb
from wandb.keras import WandbCallback


def create_callbacks(path, wandb_flag=False, wandb_test_name="NoTestName", wandb_config=None):
    """"""
    if wandb_config is None:
        wandb_config = {'competition': 'AWMadison'}

    filename = "/RNN_Final-{epoch:02d}-{loss:.5f}"
    checkpoint = ModelCheckpoint("{}{}.model".format(path, filename,
                                                     monitor='loss',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     mode='max'))  # saves only the best ones.
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    if wandb_flag:
        wandb.login(relogin=True, key="52fb822f6a358eedc0a801169d4b00b63ffa125f")
        wandb.init(project="HuBMAP", entity="jordiarellano1996", name=wandb_test_name, config=wandb_config)
        return [early_stop, WandbCallback()]
    else:
        return [early_stop]
