import tensorflow as tf
import numpy as np
from accesslib.segmentation_precompute.patches import ExtractPatchesFlat
from accesslib.segmentation_precompute.read_image import read_image, img_rescale
from accesslib.segmentation_precompute.augmentations import random_transform


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, img_path: list, mask_path: list, img_size: tuple, batch_size=24, shuffle=False, augment=False):

        assert len(img_path) == len(mask_path)
        self.img_paths = img_path
        self.mask_paths = mask_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        # Model.fit in each epoch it may create again the class so "self.on_epoch_end()" should be called in each epoch.
        self.indexes = None
        self.path_store = {}
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        ct = int(np.ceil(len(self.img_paths) / self.batch_size))
        return ct

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        img_p = np.array(self.img_paths)[indexes]
        mask_p = np.array(self.mask_paths)[indexes]
        self.path_store[index] = {"img": img_p, "mask": mask_p}
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
            # Read
            img = read_image(self.img_paths[index])
            mask = read_image(self.mask_paths[index])
            # Augment
            if self.augment:
                img, mask = random_transform(img, mask)
            # Rescale
            img = img_rescale(img)
            mask = img_rescale(mask)
            # Save
            x[k] = img
            y[k] = mask
            k += 1

        return x, y
