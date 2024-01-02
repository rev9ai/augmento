

import numpy as np
import cv2


class Colors:

    def __init__(self):
        pass

    @staticmethod
    def jitter(params: dict = {}):

        '''

        Create a color jittering augmentation for an input image.

        Arguments:
        params (dict): A dictionary of parameters for the image augmentation. The following key-value pairs are accepted:
            - 'distribution' (str, default='normal'): The type of distribution to use for generating the augmentation values.
            - 'add_brightness' (bool, default=True): Whether or not to add brightness augmentation.
            - 'brightness_range' (int, default=50): The range of brightness values to use for augmentation.
            - 'brightness_value' (int, default=None): A specific brightness value to use for augmentation. If None, then a random value with in the range will be choosen.
            - 'add_contrast' (bool, default=True): Whether or not to add contrast augmentation.
            - 'contrast_range' (int, default=20): The range of contrast values to use for augmentation.
            - 'contrast_value' (float, default=None): A specific contrast value to use for augmentation. If None, then a random value with in the range will be choosen.

        Returns:
        An instance of colors jittering augmentation class

        '''

        self = Colors()

        def apply(distribution: str = 'normal',
                  add_brightness=True, brightness_range=50, brightness_value=None,
                  add_contrast=True, contrast_range=20, contrast_value=None):

            if distribution == 'normal':
                self.brightness = np.random.randn() * brightness_range
                self.contrast = np.random.randn() * contrast_range
            if brightness_value:
                self.brightness = brightness_value
            if contrast_value:
                self.contrast = contrast_value

            if not add_brightness:
                self.brightness = None
            if not add_contrast:
                self.contrast = None

        apply(**params)
        return self

    def __call__(self, image: np.ndarray, **kwargs):

        """
        Apply image augmentation to an input image and return a dictionary containing the augmented image.

        Arguments:
        image (numpy.array): The input image to be augmented

        Returns:
        A dictionary containing the augmented image in the key 'image'.
        """

        augmented = image.copy()

        if self.brightness:
            if self.brightness > 0:
                shadow = self.brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + self.brightness
            alpha_b = (highlight - shadow) / 255
            gamma_b = shadow

            augmented = cv2.addWeighted(augmented, alpha_b, augmented, 0, gamma_b)

        if self.contrast:
            f = 131 * (self.contrast + 127) / (127 * (131 - self.contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)

            augmented = cv2.addWeighted(augmented, alpha_c, augmented, 0, gamma_c)

        output = {'image': augmented}
        if kwargs.get('annotations') is not None:
            output.update({'annotations': kwargs['annotations']})
        return output
