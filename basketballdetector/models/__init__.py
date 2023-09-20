"""
This package contains all the implemented models.
Each model has a model_name and a model property: they return the model's name and the actual model respectively.
The model is an instance of the Keras Model class.
More information can be found [here](https://keras.io/api/models/model/)
"""

from .classification import EfficientNet, MobileNet, ResNet152V2
from .segmentation import UNet
