# Augmento: An Image Augmentation Library

Augmento is a Python library that provides a collection of image augmentation techniques for machine learning applications. It offers a wide range of augmentation functions that can help improve the accuracy and robustness of computer vision models by introducing variations to the input data. With Augmento, you can easily incorporate different types of image augmentations in your data pipeline and train more effective models.

The library is designed to be flexible and easy-to-use, with a user-friendly API that allows you to chain together different augmentation functions and customize their parameters to fit your specific use case. Augmento supports a variety of image formats and integrates well with popular machine learning frameworks like PyTorch and TensorFlow.

Whether you're working on a computer vision project, training an object detection algorithm, or building a neural network for image classification, Augmento can help you create more diverse and representative training datasets and improve the performance of your models.

## Installation

To install Augmento, you can use pip, a popular Python package manager. First, ensure that you have pip installed on your system. Then, run the following command:

```shell
pip install augmento
```

This will download and install the latest version of Augmento and its dependencies. Once installation is complete, you can begin using the library in your Python projects.

## Usage

To use the `augmento` library, first, you need to install it using pip or any other package manager. After installation, you can import the `Augmento` class and the desired augmentation classes from the `augmento.augmentations` module.

Here's an example of how to use the `Augmento` class to apply multiple augmentations to an image:

```python
from augmento.augmentations import Colors, Resizing, Rotations
from augmento import Augmento
import cv2

# Load the image
image = cv2.imread("image.jpg")

# Create an instance of the Augmento class
augmentor = Augmento()

# Add augmentations to the pipeline
augmentor.add(augmentation=Colors.jitter())
augmentor.add(augmentation=Resizing.cropper())
augmentor.add(augmentation=Rotations.rotate())

# Apply the augmentations to the image
augmented = augmentor(image)

# Display the augmented image
cv2.imshow("Augmented Image", augmented["image"])
cv2.waitKey(0)
cv2.destroyAllWindows()
```

In the above example, we first loaded an image using OpenCV's `imread()` function. Then we created an instance of the `Augmento` class and added three augmentations to the pipeline using the `add()` method. Finally, we applied the augmentations to the image using the `__call__()` method of the `Augmento` class and displayed the augmented image using OpenCV's `imshow()` function.

You can also apply augmentations to annotations, if provided, by passing them as an additional argument to the `__call__()` method:

```python
# Load the image and annotations
image = cv2.imread("image.jpg")
annotations = load_annotations("annotations.json")

# Create an instance of the Augmento class
augmentor = Augmento()

# Add augmentations to the pipeline
augmentor.add(augmentation=Colors.jitter())
augmentor.add(augmentation=Resizing.cropper())
augmentor.add(augmentation=Rotations.rotate())

# Apply the augmentations to the image and annotations
augmented = augmentor(image, annotations)

# Display the augmented image and annotations
cv2.imshow("Augmented Image", augmented["image"])
print(augmented["annotations"])
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Classes

### `Augmento` class

The `Augmento` class is the main class of the `augmento` library, which is used for image augmentation. It accepts a list of augmentation objects that apply various image transformations to the input image.

#### `__init__(self)` method

The `__init__` method initializes the `Augmento` class and creates an empty list for storing the augmentations. It does not accept any arguments.

#### `add(self, augmentation)` method

The `add` method is used to add an augmentation to the list of augmentations to apply. It accepts a single argument, which is an instance of an augmentation class such as `Colors.jitter()`, `Resizing.cropper({})`, or `Rotations.rotate({})`.

#### `__call__(self, image, annotations=None)` method

The `__call__` method is the main method of the `Augmento` class that applies the list of augmentations to the input image. It accepts two arguments:

- `image`: The input image to be augmented. The image should be a NumPy array with dimensions `(height, width, channels)`.
- `annotations`: Optional. A 2D array of annotations in the shape of `x,y,score`. This can be used to update the annotations if they change due to an augmentation.

The method returns a dictionary with the following keys:

- `'image'`: The augmented image as a NumPy array with dimensions `(height, width, channels)`.
- `'annotations'`: The updated annotations (if any) as a dictionary. If annotations were not passed as an argument, then the value of this key will be `None`.

### `Colors` class

The `Colors` class provides image color augmentation techniques.

#### `jitter(params={})` method

The `jitter` method create color jittering augmentation.

**Arguments:**

- `params`

   (dict): A dictionary of parameters for the image augmentation. The following key-value pairs are accepted:

  - `distribution` (str, default='normal'): The type of distribution to use for generating the augmentation values.
  - `add_brightness` (bool, default=True): Whether or not to add brightness augmentation.
  - `brightness_range` (int, default=50): The range of brightness values to use for augmentation.
  - `brightness_value` (int, default=None): A specific brightness value to use for augmentation. If `None`, then a random value within the range will be chosen.
  - `add_contrast` (bool, default=True): Whether or not to add contrast augmentation.
  - `contrast_range` (int, default=20): The range of contrast values to use for augmentation.
  - `contrast_value` (float, default=None): A specific contrast value to use for augmentation. If `None`, then a random value within the range will be chosen.

**Returns:**

An instance of the `Colors` jittering augmentation class.

### `Resizing` class

#### `cropper(params={})` method

Create a cropping augmentation for an input image.

Arguments:

- `params`

   (dict): A dictionary of parameters for the image cropping. The following key-value pairs are accepted:

  - `background` (str, default=None): The path to the background image to use for cropped area. It will work with `keep_within_size=True` and set the specified background for the cropped area.
  - `keep_within_size` (bool, default=True): Whether to keep the output image within the same size as the input image.
  - `cropping_value` (int, default=None): The amount of cropping to apply to the input image. If `None`, then a random cropping value will be set selected from a random normal distribution in the range of 0-360.

Returns: An instance of the `cropper` augmentation class.

### `Rotations` class

#### `rotate(params={})` method

The `rotate` method creates a rotation augmentation for an input image. It accepts a dictionary of parameters for the rotation configurations. The following key-value pairs are accepted:

- `angle` (int, default=None): The angle of rotation in degrees. If None, a random angle between -25 and 25 degrees will be selected.
- `random_bg` (bool, default=False): Whether to fill the background of the rotated image with a random color.

The `rotate` method returns an instance of the rotation augmentation class, which can be added to an instance of the `Augmento` class for augmenting an image.

## Examples

First, let's import the necessary classes and libraries:

```python
from augmento.augmentations import Augmento, Colors, Rotations
import cv2
import numpy as np
```

Then, let's load the image and its corresponding annotations:

```python
image_path = 'path/to/image.jpg'
annotation_path = 'path/to/annotations.npy'

# Load the image
image = cv2.imread(image_path)

# Load the annotations
annotations = np.load(annotation_path)
```

Next, let's create an instance of the `Augmento` class and add the desired augmentations:

```python
# Create an instance of the Augmento class
augmentor = Augmento()

# Add color jittering augmentation
augmentor.add(augmentation=Colors.jitter())

# Add rotation augmentation
augmentor.add(augmentation=Rotations.rotate(params={'angle': 45}))
```

Now, we can apply the augmentations to the image and annotations using the `__call__` method of the `Augmento` class:

```python
# Apply the augmentations
augmented = augmentor(image, annotations)
```

### Augmentations for Bounding Boxes

If the annotations are in the format of `x1,y1,x2,y2` bounding boxes, we need to reshape them to `x,y` format:

```python
# Reshape annotations from (n, 4) to (2n, 2)
annotations = annotations.reshape(-1, 2)

# Apply the augmentations
augmented = augmentor(image, annotations)

# Reshape annotations back to (n, 4)
annotations = annotations.reshape(-1, 4)
```

Finally, we can save the augmented image and annotations to disk:

```python
# Save the augmented image
cv2.imwrite('path/to/augmented_image.jpg', augmented['image'])

# Save the augmented annotations
np.save('path/to/augmented_annotations.npy', augmented['annotations'])
```

That's it! You have successfully applied color jittering and rotation augmentation to an image with annotations using the `augmento` library.