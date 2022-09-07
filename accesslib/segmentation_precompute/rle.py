import os
import numpy as np
import cv2


class RLE:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def mask_from_segmentation(self, rle, shape):
        """
        Returns the mask corresponding to the inputed segmentation.
        mask_rle: run-length as string formated (start length)
        max_shape: the shape to be taken by the mask
        return:: a 2D mask
        """

        # Get a string of numbers from the initial segmentation
        segm = np.asarray(rle.split(), dtype=int)

        # Get start point and length between points
        start_point = segm[0::2] - 1
        length_point = segm[1::2]

        # Compute the location of each endpoint
        end_point = start_point + length_point

        # Create an empty list mask the size of the original image
        # take onl
        case_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

        # Change pixels from 0 to 1 that are within the segmentation
        for start, end in zip(start_point, end_point):
            case_mask[start:end] = 1

        case_mask = case_mask.reshape((shape[0], shape[1]))
        if self.verbose:
            print(f"The mask shape is: {case_mask.shape}")

        return case_mask.T

    def save_image_on_disk(self, img, path_scaled):
        dir_path = "/" + os.path.join(*path_scaled.split("/")[:-1])
        try:
            os.makedirs(dir_path)
        except FileExistsError:
            if self.verbose:
                print(f"File exist")
        cv2.imwrite(path_scaled, img)


def rle_encode(img):
    """ TBD
    Args:
        img (np.array):
            - 1 indicating mask
            - 0 indicating background

    Returns:
        run length as string formated
    """

    img = img.T
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)