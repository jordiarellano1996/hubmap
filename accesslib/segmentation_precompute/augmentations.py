import albumentations as albu
import numpy as np


def random_transform(img, msk):
    composition = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(rotate_limit=25, scale_limit=0.15, shift_limit=0, p=0.75),
        albu.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.5),
        albu.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25, p=0.75),
        albu.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, p=0.5),
        # albu.RandomCrop(height=size, width=size, p=1.0)
    ])
    tmp = composition(image=img, mask=msk)
    return tmp['image'], tmp['mask']


def augment_batch(img_batch, mask_batch, size: list):
    img_batch2 = np.zeros((len(img_batch), size[0], size[1], 3))
    mask_batch2 = np.zeros((len(mask_batch), size[0], size[1], 1))
    for i in range(img_batch.shape[0]):
        img_batch2[i, ], mask_batch2[i, ] = random_transform(img_batch[i, ], mask_batch[i, ])
    return img_batch2, mask_batch2
