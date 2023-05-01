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

    def __call__(self, image: np.array, annotations: Optional[np.ndarray]) -> dict:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray for image, but got {type(image)}")
        if len(image.shape) != 3:
            raise ValueError(f"Expected 3D image, but got {len(image.shape)}D image")
        if self.flip_type not in [0, 1]:
            raise ValueError(f"Invalid flip type: {self.flip_type}. Expected 0 or 1.")

        # Randomly decide whether to flip the image
        if np.random.uniform(0, 1) > self.prob:

            # Flip the image if needed
            flipped_image = cv2.flip(image, self.flip_type)

            # Flip the annotations if needed
            flipped_annotations = annotations.copy() if annotations is not None else None
            if flipped_annotations is not None:
                if self.flip_type == 0:
                    flipped_annotations[:, 1] = flipped_image.shape[0] - flipped_annotations[:, 1]
                elif self.flip_type == 1:
                    flipped_annotations[:, 0] = flipped_image.shape[1] - flipped_annotations[:, 0]

            # Return the flipped image and annotations (if any)
            return {"image": flipped_image, "annotations": flipped_annotations} \
                if flipped_annotations is not None \
                else {"image": flipped_image}
        else:
            return {"image": image, "annotations": annotations}
