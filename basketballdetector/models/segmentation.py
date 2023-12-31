"""
This module contains all the implemented segmentation models
"""

from tensorflow import keras


class UNet:
    """
    This class represents an implementation of the UNet model.
    It is  based on [this tutorial](https://keras.io/examples/vision/oxford_pets_image_segmentation/#model-architecture)
    """
    def __init__(self, input_shape: (int, int, int) = (512, 1024, 3), number_of_classes: int = 2):
        """
        Class constructor
        :param input_shape: Input shape, (image_height, image_width, channels)
        :param number_of_classes: Number of classes for this segmentor
        """
        inputs = keras.Input(shape=input_shape)
        x = keras.layers.Rescaling(1. / 255, input_shape=input_shape)(inputs)

        # [First half of the network: downsampling inputs] ###

        # Entry block
        x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.UpSampling2D(2)(x)

            # Project residual
            residual = keras.layers.UpSampling2D(2)(previous_block_activation)
            residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
            x = keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = keras.layers.Conv2D(number_of_classes, 3, activation="softmax", padding="same")(x)

        self.__model = keras.Model(inputs, outputs)
        self.__model_name = 'unet'

    @property
    def model(self):
        """
        This method returns the actual model
        :return: The model
        """
        return self.__model

    @property
    def model_name(self):
        """
        This method returns the model's name
        :return: The model's name
        """
        return self.__model_name
