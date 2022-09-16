import tensorflow as tf
import numpy as np
import albumentations as albu


class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, img_list: list, mask_list: list, batch_size: int = 24, shuffle: bool = False,
                 augment: bool = False, crops: int = 1, size: int = 512, size2: int = 512, shrink: int = 1):
        """
        If augmentation == True
            size: Final crop size, placed in '__random_transform'
            size2: Might be higher or equal than 'size', is the first crop.
        If augmentation == False
            size: Windows "mask up finder" size.
            size2: Final crop size.
        shrink: Reduced image by either 1x, 2x, 3x, or 4x using a Numpy.
        crops: How many random crops with mask data for each image, minimum value 1.
        """

        assert crops >= 1
        assert len(img_list) == len(mask_list)

        self.img_list = img_list
        self.mask_list = mask_list
        self.batch_size = batch_size
        self.crops = crops
        self.size = size
        self.size2 = size2
        self.shrink = shrink
        self.shuffle = shuffle
        self.augment = augment
        # Model.fit in each epoch it may create again the class so "self.on_epoch_end()" should be called in each epoch.
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        ct = int(np.ceil(self.crops * len(self.img_list) / self.batch_size))
        return ct

    def __getitem__(self, index: int):
        """Generate one batch of data"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__data_generation(indexes)

        if self.augment:
            x2, y2 = self.__augment_batch(x, y)
        else:
            x2 = x
            y2 = y

        return x2, y2

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.img_list))
        self.indexes = np.repeat(self.indexes, self.crops)  # Repeat each value n times.

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes: list):
        """Generates data containing batch_size samples"""

        x = np.zeros((len(indexes), self.size2, self.size2, 3), dtype='float32')
        y = np.zeros((len(indexes), self.size2, self.size2, 1), dtype='float32')

        k = 0
        for index in indexes:
            """ Read data from path """
            img = self.img_list[index]
            mask = self.mask_list[index]

            """ Iterate on image to get a crop with valid percentage of 'mask up' values."""
            sm = 0
            ct = 0
            a = 0
            b = 0
            pct = 10  # percentage of pixels up threshold.
            ct_threshold = 25
            while (sm == 0) & (ct < ct_threshold):
                a = np.random.randint(0, img.shape[0] - self.size2 * self.shrink)
                b = np.random.randint(0, img.shape[1] - self.size2 * self.shrink)
                # sm = np.sum()
                flatten_mask = mask[a:a + self.size * self.shrink, b:b + self.size * self.shrink].flatten()
                sm = int((len(flatten_mask[flatten_mask == 255]) / len(flatten_mask)) * 100)
                if sm < pct:
                    sm = 0

                ct += 1

            x[k,] = img[a:a + self.size2 * self.shrink, b:b + self.size2 * self.shrink, ][::self.shrink,
                    ::self.shrink] / 255.
            y[k,] = mask[a:a + self.size2 * self.shrink, b:b + self.size2 * self.shrink][::self.shrink,
                    ::self.shrink] / 255.

            k += 1

        return x, y

    def __random_transform(self, img, msk):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(rotate_limit=25, scale_limit=0.15, shift_limit=0, p=0.75),
            albu.CoarseDropout(max_holes=8, max_height=30, max_width=30, p=0.2),
            albu.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25, p=0.75),
            albu.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, p=0.5),
            albu.RandomCrop(height=self.size, width=self.size, p=1.0)
        ])
        tmp = composition(image=img, mask=msk)
        return tmp['image'], tmp['mask']

    def __augment_batch(self, img_batch, mask_batch):
        img_batch2 = np.zeros((len(img_batch), self.size, self.size, 3), dtype='float32')
        mask_batch2 = np.zeros((len(mask_batch), self.size, self.size, 1), dtype='float32')
        for i in range(img_batch.shape[0]):
            img_batch2[i,], mask_batch2[i,] = self.__random_transform(img_batch[i,], mask_batch[i,])
        return img_batch2, mask_batch2
