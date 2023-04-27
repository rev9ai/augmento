import cv2
import numpy as np


class Flipper:

    def __init__(self):
        self.flip_type = None

    @staticmethod
    def horizontal_flip():
        flip = Flipper()
        flip.flip_type = 1
        return flip

    @staticmethod
    def vertical_flip():
        flip = Flipper()
        flip.flip_type = 0
        return flip

    def __call__(self, image: np.array, **kwargs):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray for image, but got {type(image)}")
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D image, but got {len(image.shape)}D image")
        if self.flip_type not in [0, 1]:
            raise ValueError(f"Invalid flip type: {self.flip_type}. Expected 0 or 1.")

        augmented = cv2.flip(image, self.flip_type)
        output = {'image': augmented}
        if kwargs.get('annotations') is not None:
            annotations = kwargs['annotations'].copy()
            if self.flip_type == 0:
                annotations[:, 1] = image.shape[0] - annotations[:, 1]
            elif self.flip_type == 1:
                annotations[:, 0] = image.shape[1] - annotations[:, 0]
            output.update({'annotations': annotations})
        return output
