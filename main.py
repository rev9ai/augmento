

from augmentations import Colors, Resizing, Rotations, Flipper
from typing import Union, Optional
import numpy as np


class Augmento:
    """
    Augment an input image along with its associated annotations.

    This class provides a way to apply a sequence of augmentation operations to an input image and its associated
    annotations. There are various augmentations that are available in this library.

    Methods:
        add: Add the augmentation in the pipeline that needs to apply on the image and annotations.

    Example usage:
    >>> augmentor = Augmento()
    >>> augmentor.add(augmentation=Colors.jitter())
    >>> augmentor.add(augmentation=Resizing.cropper({}))
    >>> augmented = augmentor(image, annotations)
    """

    def __init__(self):
        self.augmentations = list()

    def add(self, augmentation: Union[Colors, Resizing, Rotations, Flipper]) -> None:
        """
        Adds an augmentation to the pipeline of the Augmentation instance.

        Arguments:
            augmentation (Union[Colors, Resizing, Rotations]): The augmentation to add. This can be an instance of the
            Colors, Resizing, or Rotations classes.

        Returns:
            None
        """

        self.augmentations.append(augmentation)

    def __call__(self, image: np.ndarray, annotations: Optional[np.ndarray] = None) -> dict:
        """
        Applies the augmentations added in the pipeline to an image and its annotations.

        Args:
            image (np.ndarray): The image to apply augmentations to.
            annotations (Optional[np.ndarray]): Optional annotations to apply augmentations to. If not provided, no
            annotations will be returned.

        Returns:
            A dictionary containing the augmented image and annotations (if any) in the format {"image": image,
            "annotations": annotations}.
        """

        params = {'image': image.copy(),
                  'annotations': annotations}
        for augmentation in self.augmentations:
            params = augmentation(**params)
        return params
