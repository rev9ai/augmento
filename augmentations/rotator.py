

import numpy as np
import cv2
import random
from typing import Optional

class Rotations:

    def __init__(self):
        pass

    @staticmethod
    def rotate(params: dict = {}):

        """
        Rotate image augmentation class generator.

        Arguments:
            params (dict): A dictionary of parameters for the rotation configs. The following key-value pairs are accepted:
                - 'angle' (int, default=None): The angle of rotation in degrees. If None, a random angle between -25 and 25
                    degrees will be selected.
                - 'random_bg' (bool, default=False): Whether to fill the background of the rotated image with a random color.

        Returns:
            An instance of rotation augmentation class
        """

        self = Rotations()

        def apply(angle: int = None, random_bg: bool = False):

            if angle is None:
                angle = int(np.random.randn() / 3.001 * 25)

            self.angle = angle
            self.random_bg = random_bg

        apply(**params)
        return self

    def _rotate_image(self, image: np.ndarray, angle: int, random_bg: bool = False):

        """
        Rotate the input image by a specified angle.

        Arguments:
            image (numpy.ndarray): The input image to be rotated.
            angle (int): The angle of rotation in degrees.
            random_bg (bool, optional): Whether to fill the background of the rotated image with a random color.
                Defaults to False.

        Returns:
            numpy.ndarray: The rotated image.
        """

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        bg_args = {}
        if random_bg:
            border_value = tuple(
                map(lambda x: int(x),
                    random.choice(np.mean(image, axis=0))
                    )
            )
            bg_args = dict(borderMode=cv2.BORDER_CONSTANT,
                           borderValue=border_value
                           )
        rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, **bg_args)

        return rotated_image


    def __call__(self, image: np.array, annotations: Optional[np.ndarray] = None, **kwargs):

        """
        Apply the defined augmentation to the input image and annotations.

        Arguments:
            image (numpy.ndarray): The input image to be augmented.
            annotations (numpy.ndarray, optional): An array of annotations for the image. Defaults to None.

        Returns:
            dict: A dictionary containing the augmented image and annotations, if any. The dictionary has the following keys:
                - 'image': The augmented image as a numpy.ndarray.
                - 'annotations': The augmented annotations as a numpy.ndarray, if provided.
        """

        output = dict()
        rotated_image = self._rotate_image(image, angle=self.angle, random_bg=self.random_bg)
        output['image'] = rotated_image

        if annotations is not None:

            vectors = annotations.copy()
            rotated_vectors = np.zeros(vectors.shape)

            for i, point in enumerate(vectors):
                if len(point) == 4:
                    box = point
                else:
                    box_size = 5
                    box = [point[0], point[1], point[0]+box_size, point[1]+box_size]
                box_img = np.zeros(rotated_image.shape).astype('uint8')
                box_img[box[1]:box[3] + 1, box[0]:box[2] + 1] = 255

                box_img = self._rotate_image(box_img, angle=self.angle)

                if len(np.argwhere(box_img == 255)) == 0:
                    continue

                x1, x2 = np.min(np.argwhere(box_img == 255)[:, 1]), np.max(np.argwhere(box_img == 255)[:, 1])
                y1, y2 = np.min(np.argwhere(box_img == 255)[:, 0]), np.max(np.argwhere(box_img == 255)[:, 0])

                if len(point) == 4:
                    rotated_vectors[i] = [x1, y1, x2, y2]
                else:
                    rotated_vectors[i] = [x1, y1]

            rotated_vectors = rotated_vectors.round().astype(int)
            output['annotations'] = rotated_vectors

        return output














