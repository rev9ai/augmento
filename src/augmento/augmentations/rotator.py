

import numpy as np
import cv2
import random
from typing import Optional, List, Union

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

    def _rotate_p_about_c(self, point: List, angle: int, center_point: List):

        """
        Rotate point by a give angle about center_point

        Arguments:
            point (list): A point would be a list like of length 2 that needs to rotate
            angle (int): The angle in degree to rotate the point.
            center_point (list): Pivot point i.e. the point about which needs to rotate the point by given angle.
        Returns:
            list: Rotated point
        """

        angle_in_rad = np.deg2rad(angle)
        sin_theta = np.sin(angle_in_rad)
        cos_theta = np.cos(angle_in_rad)

        a, b = center_point
        x, y = point

        x = x - a
        y = y - b

        x_prime = round(x * cos_theta - y * sin_theta + a)
        y_prime = round(x * sin_theta + y * cos_theta + b)

        return [x_prime, y_prime]

    def _box_to_polygon(self, bboxes: Union[List[List], np.ndarray]):
        """
        Convert list of bounding boxes into list of polygons coordinates. For each bounding box, it will generate polygon coordinates.
        Arguments:
            bboxes (list or np.ndarray): A list or numpy array of bounding boxes. Each bounding box should follow a format [x1, y1, x2, y2]. Polygon coordinates will be generated for each boudning box.
        Returns:
            np.ndarray: A numpy array will be returned that will contain the polygon coordinates for each respective bounding box.
        """
        polygons = list()
        for box in bboxes:
            x1, y1, x2, y2 = box
            polygons.append([
                [x1, y1],
                [x2, y1],
                [x1, y2],
                [x2, y2]
            ])
        return np.array(polygons)

    def _polygon_to_box(self, polygons: Union[List[List[List]], np.ndarray]):

        """
        Convert list of polygons coordinates into list of bounding boxes. For each polygon coordinates array, it will generate a bounding box.
        Arguments:
            polygons (list or np.ndarray): A list or numpy array containing bounding boxes' polygon coordinates. Each polygon coordinates array should contain the polygon coordinates of a bounding box. An array of bounding boxes will be generated.
        Returns:
            np.ndarray: A numpy array will be returned that will contain the bounding boxes for each respective polygon coordinates array.
        """

        bboxes = list()
        for polygon in polygons:
            x1 = min(p[0] for p in polygon)
            y1 = min(p[1] for p in polygon)
            x2 = max(p[0] for p in polygon)
            y2 = max(p[1] for p in polygon)
            bboxes.append([x1, y1, x2, y2])
        return np.array(bboxes)


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

            h, w, _ = image.shape
            a, b = w / 2, h / 2
            if annotations.shape[1] == 4:
                polygons = self._box_to_polygon(annotations)
                rotated_polygons = [
                    [
                        self._rotate_p_about_c((x, y), - self.angle, (a, b)) \
                        for (x, y) in pol] for pol in polygons
                ]
                rotated_annotations = self._polygon_to_box(rotated_polygons)

                rotated_annotations[rotated_annotations[:, 0] > w, 0] = w - 1
                rotated_annotations[rotated_annotations[:, 2] > w, 2] = w - 1
                rotated_annotations[rotated_annotations[:, 1] > h, 1] = h - 1
                rotated_annotations[rotated_annotations[:, 3] > h, 3] = h - 1
                rotated_annotations[rotated_annotations < 0] = 0
            else:
                rotated_annotations = [
                    self._rotate_p_about_c((x, y), - self.angle, (a, b)) \
                    for (x, y) in annotations[:, :2]
                ]
                rotated_annotations = np.array(rotated_annotations)

                rotated_annotations[rotated_annotations[:, 0] > w, 0] = w - 1
                rotated_annotations[rotated_annotations[:, 1] > h, 1] = h - 1
                rotated_annotations[rotated_annotations < 0] = 0

            output['annotations'] = rotated_annotations

        return output














