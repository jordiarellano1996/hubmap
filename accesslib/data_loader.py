import tensorflow as tf
import numpy as np
from accesslib.segmentation_precompute.patches import ExtractPatchesFlat
from accesslib.segmentation_precompute.read_image import read_image, img_rescale

IMG_SIZE = (500, 500, 3)


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, img_path: list, mask_path: list, batch_size=24, shuffle=False, augment=False
                 , img_size=IMG_SIZE):

        assert len(img_path) == len(mask_path)
        self.img_paths = img_path
        self.mask_paths = mask_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        # Model.fit in each epoch it may create again the class so "self.on_epoch_end()" should be called in each epoch.
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        ct = int(np.ceil(len(self.img_paths) / self.batch_size))
        return ct

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        print("Updating indexes")
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""
        x = np.zeros((len(indexes), self.img_size[0], self.img_size[1], 3), dtype='float32')
        y = np.zeros((len(indexes), self.img_size[0], self.img_size[1], 1), dtype='float32')

        k = 0
        for index in indexes:
            img = read_image(self.img_paths[index])
            mask = read_image(self.mask_paths[index])
            x[k] = img
            y[k] = mask
            k += 1

        return x, y
