import numpy as np
import cv2


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
            new_annotations[:, :, 0] = annotations[:, :, 0] - pl
            new_annotations[:, :, 1] = annotations[:, :, 1] - pt
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
            transformed_annotations = list()
            # TODO: Replace For Loop with Matrix Operations
            for bbox in annotations:
                corners = np.float32([bbox[0],
                                      [bbox[1, 0], bbox[0, 1]],
                                      bbox[1],
                                      [bbox[0, 0], bbox[1, 1]]])
                transformed_corners = cv2.transform(np.array([corners]), matrix)[0]

                # Calculate the new width and height of the bounding box
                x, y, w, h = cv2.boundingRect(transformed_corners.astype(int))
                transformed_bbox = np.array([[x, y], [x + w, y + h]])
                h, w = image.shape[:2]
                transformed_bbox[transformed_bbox[:, 0] > w, 0] = w - 1
                transformed_bbox[transformed_bbox[:, 1] > h, 1] = h - 1
                transformed_bbox[transformed_bbox < 0] = 0
                transformed_annotations.append(transformed_bbox)
            transformed_annotations = np.array(transformed_annotations)

        return transformed_image, transformed_annotations

    def __call__(self, image, annotations):

        augmented_image, augmented_annotations = image.copy(), annotations.copy() if annotations is not None else None
        for call in self.pipeline:
            augmented_image, augmented_annotations = call(augmented_image, augmented_annotations)

        return {"image": augmented_image, "annotations": augmented_annotations}
