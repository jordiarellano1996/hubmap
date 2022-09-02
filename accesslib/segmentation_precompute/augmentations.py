import albumentations as albu


def random_transform(img, msk):
    composition = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(rotate_limit=25, scale_limit=0.15, shift_limit=0, p=0.75),
        albu.CoarseDropout(max_holes=16, max_height=64, max_width=64, p=0.5),
        albu.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25, p=0.75),
        albu.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, p=0.5),
    ])
    tmp = composition(image=img, mask=msk)
    return tmp['image'], tmp['mask']
