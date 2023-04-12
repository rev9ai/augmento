

import os.path
import random
from typing import Optional
import numpy as np
from PIL import Image


class Resizing:

    def __init__(self):
        pass

    @staticmethod
    def cropper(params: dict = {}):

        """
         Create a cropping augmentation for an input image

        Arguments:
            params (dict): A dictionary of parameters for the image cropping. The following key-value pairs are accepted:
                - 'background' The path to the background image to use for cropping. Defaults to None. It will work with keep_within_size=True and set the specified background for the cropped area.
                - 'keep_within_size' (bool, default=True): Whether to keep the output image within the same size as the input image.
                - 'cropping_value' (int, default=None): The amount of cropping to apply to the input image. If None, then a random cropping value will be set selected from a random normal distribution in the range of 0-360.

        Returns:
            An instance of cropper augmentation class
        """

        self = Resizing()

        def apply(background: Optional[str] = None, keep_within_size: Optional[bool] = True, cropping_value: Optional[int] = None):

            if cropping_value is not None:
                crop_area = cropping_value
            else:
                random_scale = round(abs(np.random.randn() / 3.001) * 50 + 10) * 6
                crop_area = int(random_scale / 2)
            self.crop_area = crop_area

            if background is not None:
                assert os.path.exists(background), "invalid path for background image"
                assert os.path.isdir(background), "background path is not a directory"
                assert len(list(filter(lambda x:
                                x.upper().endswith('.PNG') or x.upper().endswith('.JPG') or x.upper().endswith('.JPEG'),
                                os.listdir(background)))), "background directory doesn't contain png or jpg images"
                images_dir = os.listdir(background)
                background = random.choice(images_dir)

            self.background = background
            self.keep_within_size = keep_within_size

        apply(**params)
        return self

    # TODO: Add Zoom Augmentation with options like zoom randomly, center-only, corners-only


    def __call__(self, image: np.ndarray, annotations: Optional[np.ndarray] = None, **kwargs):

        """
        Apply resizing augmentations to the input image and annotations.

        Arguments:
            image (numpy.ndarray): The input image to be resized.
            annotations (numpy.ndarray, optional): An array of annotations for the image. Defaults to None.

        Returns:
            dict: A dictionary containing the resized image and annotations, if any. The dictionary has the following keys:
                - 'image': The resized image as a numpy.ndarray.
                - 'annotations': The resized annotations as a numpy.ndarray, if provided.
        """

        crop_area = self.crop_area
        background = self.background
        keep_within_size = self.keep_within_size

        output = dict()

        h, w = image.shape[:2]
        new_img = image.copy()
        new_img = new_img[crop_area:-crop_area, crop_area:-crop_area]
        nh, nw, _ = new_img.shape

        tw, th = 0, 0
        if keep_within_size:
            tw = np.random.randint(0, crop_area)
            th = np.random.randint(0, crop_area)

        if background is not None:
            with open(background, 'r') as file:
                bg = Image.open(file)
            bg = bg.resize((w, h))
            image_with_pad = np.asarray(bg).astype('uint8')
        elif keep_within_size:
            image_with_pad = np.zeros((h, w, 3)).astype('uint8')
        else:
            image_with_pad = new_img

        image_with_pad[th:nh + th, tw:nw + tw] = new_img
        output['image'] = image_with_pad

        if annotations is not None:

            new_kp = annotations.copy()
            new_kp[:, :2] = new_kp[:, :2] - crop_area
            new_kp[:, 0][new_kp[:, 0] < 0] = 0
            new_kp[:, 1][new_kp[:, 1] < 0] = 0
            new_kp[:, 0][new_kp[:, 0] > nw] = nw
            new_kp[:, 1][new_kp[:, 1] > nh] = nh

            new_kp[:, 0] = new_kp[:, 0] + tw
            new_kp[:, 1] = new_kp[:, 1] + th

            output['annotations'] = new_kp

        return output

