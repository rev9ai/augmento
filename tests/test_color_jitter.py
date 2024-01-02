
import unittest
from augmento.augmentations import Colors
from augmento import Augmento
import numpy as np

# Load the image
image = np.random.randint(0, 255, size=(244, 244, 3), dtype='uint8')

# Create an instance of the Augmento class
augmentor = Augmento()

# Add augmentations to the pipeline
augmentor.add(augmentation=Colors.jitter())

# Apply the augmentations to the image
augmented = augmentor(image)

if __name__ == '__main__':
    unittest.main()


