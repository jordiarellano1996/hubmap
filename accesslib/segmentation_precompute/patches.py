from patchify import patchify, unpatchify
import numpy as np
import cv2
from functools import reduce


class ExtractPatches:
    def __init__(self, img, verbose=False):
        self.verbose = verbose
        self.img = img
        assert len(self.img.shape) == 3
        self.add_height = 0
        self.add_width = 0

    def _check_dim(self, img_dim: tuple, patch_height: int, patch_width: int) -> tuple:
        height_remainder = img_dim[0] % patch_height
        width_remainder = img_dim[1] % patch_width

        if height_remainder != 0:
            add_height = patch_height - height_remainder
        else:
            add_height = 0

        if width_remainder != 0:
            add_width = patch_width - width_remainder
        else:
            add_width = 0

        if self.verbose:
            print(f"Add bottom: {add_height} pixels, add right: {add_width} pixels.")
        return add_height, add_width

    def extract_patches(self, patch_height, patch_width, step=1):
        """
        Reshape a 2D image into a collection of patches.
        The resulting patches are allocated in a dedicated array.
        """
        self.add_height, self.add_width = self._check_dim(self.img.shape, patch_height, patch_width)
        self.img = cv2.copyMakeBorder(self.img, 0, self.add_height, 0, self.add_width,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if len(self.img.shape) != 3:
            self.img = self.img[:, :, np.newaxis]
        patches = patchify(self.img, (patch_height, patch_width, self.img.shape[2]), step=step)
        return patches


class ExtractPatchesFlat(ExtractPatches):
    def __init__(self, img, verbose=False):
        super().__init__(img, verbose)

    def extract(self, patch_height, patch_width, step=1):
        out_patch_img = self.extract_patches(patch_height, patch_width, step)
        s = out_patch_img.shape
        p_len = reduce(lambda x, y: x * y, s[:3])
        reshape_out_patch_img = out_patch_img.reshape((p_len, s[-3:][0], s[-3:][1], s[-3:][2]))
        return reshape_out_patch_img, s


def reconstruct_from_patches(patches, image_shape):
    """
    Reconstruct the image from all of its patches.
    """
    return unpatchify(patches, image_shape)
