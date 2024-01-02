import cv2
import numpy as np
from typing import Optional


class Flipper:

    def __init__(self):
        self.prob = None
        self.flip_type = None

    @staticmethod
    def horizontal_flip(prob: float = 1.0):
        flip = Flipper()
        flip.prob = prob
        flip.flip_type = 1
        return flip

    @staticmethod
    def vertical_flip(prob: float = 1.0):
        flip = Flipper()
        flip.prob = prob
        flip.flip_type = 0
        return flip

    def __call__(self, image: np.ndarray, annotations: Optional[np.ndarray] = None, **kwargs) -> dict:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray for image, but got {type(image)}")
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D image, but got {len(image.shape)}D image")
        if self.flip_type not in [0, 1]:
            raise ValueError(f"Invalid flip type: {self.flip_type}. Expected 0 or 1.")

        output = {"image": image}
        if annotations is not None:
            output["annotations"] = annotations

        # Randomly decide whether to flip the image
        if np.random.uniform(0, 1) > self.prob:

            # Flip the image if needed
            flipped_image = cv2.flip(image, self.flip_type)
            output['image'] = flipped_image

            # Flip the annotations if needed
            if annotations is not None:
                flipped_annotations = annotations.copy()

                box_format = False
                if flipped_annotations.shape[1] == 4:
                    box_format = True
                    flipped_annotations = flipped_annotations.reshape(-1, 2)

                if self.flip_type == 0:
                    flipped_annotations[:, 1] = flipped_image.shape[0] - flipped_annotations[:, 1]
                elif self.flip_type == 1:
                    flipped_annotations[:, 0] = flipped_image.shape[1] - flipped_annotations[:, 0]

                if box_format:
                    flipped_annotations = flipped_annotations.reshape(-1, 4)

                output['annotations'] = flipped_annotations

        return output
