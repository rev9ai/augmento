

import numpy as np
import cv2
from typing import Optional


class AffineTransformer:
    def __init__(self):
        pass

    @staticmethod
    def with_padding(scale_range=(0.8, 1.2), shift_range=(-0.1, 0.1)):
        self = AffineTransformer()
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.pipeline = [self._get_affine_transformation]
        return self

    @staticmethod
    def without_padding(scale_range=(0.8, 1.2), shift_range=(-0.1, 0.1)):
        self = AffineTransformer.with_padding(scale_range=scale_range, shift_range=shift_range)
        self.pipeline.append(self._remove_padding)
        return self

    def _remove_padding(self, image, annotations):
        h, w = image.shape[:2]
        pl = int(max(0, self.tx))
        pr = int(max(0, w - min(w, (w * self.scale)) + abs(min(0, self.tx)) - max(0, self.tx) -
                     max(0, w * self.scale - w)))
        pt = int(max(0, self.ty))
        pb = int(max(0, h - min(h, (h * self.scale)) + abs(min(0, self.ty)) - max(0, self.ty) -
                     max(0, h * self.scale - h)))
        pr = -w if pr == 0 else pr
        pb = -h if pb == 0 else pb
        print(pl, pr, pt, pb)
        image_no_padding = image[pt:-pb, pl:-pr]
        new_annotations = None
        if annotations is not None:
            new_annotations = annotations.copy()
            new_annotations[:, 0] = annotations[:, 0] - pl
            new_annotations[:, 1] = annotations[:, 1] - pt
        return image_no_padding, new_annotations

    def _get_affine_transformation(self, image, annotations):

        h, w = image.shape[:2]

        # Validate input arguments
        if image is None:
            raise ValueError("image cannot be None")
        if annotations is not None and len(annotations) == 0:
            annotations = None

        # Generate a random scale factor and translation values
        self.scale = np.random.uniform(*self.scale_range)
        self.tx = np.random.uniform(*self.shift_range) * w
        self.ty = np.random.uniform(*self.shift_range) * h

        # Generate a transformation matrix
        matrix = np.float32([[self.scale, 0, self.tx], [0, self.scale, self.ty]])
        # Apply the transformation to the image and annotations
        transformed_image = cv2.warpAffine(image, matrix, (w, h))
        # transformed_annotations = annotations.copy()
        transformed_annotations = None
        if annotations is not None:
            transformed_annotations = np.array(annotations.copy())
            transformed_annotations[:, 0] = (transformed_annotations[:, 0] * self.scale) + self.tx
            transformed_annotations[:, 1] = (transformed_annotations[:, 1] * self.scale) + self.ty
            transformed_annotations[transformed_annotations[:, 0] > w, 0] = w - 1
            transformed_annotations[transformed_annotations[:, 1] > h, 1] = h - 1
            transformed_annotations[transformed_annotations < 0] = 0

        return transformed_image, transformed_annotations

    def __call__(self, image: np.ndarray, annotations: Optional[np.ndarray] = None, **kwargs):

        augmented_image, augmented_annotations = image.copy(), annotations.copy() if annotations is not None else None

        box_format = False
        if augmented_annotations is not None:
            if augmented_annotations.shape[1] == 4:
                box_format = True
                augmented_annotations = augmented_annotations.reshape(-1, 2)

        for call in self.pipeline:
            augmented_image, augmented_annotations = call(augmented_image, augmented_annotations)

        output = {"image": augmented_image}
        if augmented_annotations is not None:

            if box_format:
                augmented_annotations = augmented_annotations.reshape(-1, 4)

            output['annotations'] = augmented_annotations

        return output
