"""
This module contains all the implemented classifiers.
"""

import tensorflow as tf


class SimpleClassifier:
    """
    This class represents a simple classifier used for testing and experiments.
    It is based on this tutorial https://www.tensorflow.org/tutorials/images/classification#a_basic_keras_model
    """
    def __init__(self,
                 number_of_classes: int = 2,
                 image_width: int = 50,
                 image_height: int = 50):
        self.__model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(image_height, image_width, 3)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(number_of_classes)
        ])
        self.__model_name = 'simple-classifier'

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


class ResNet152V2:
    def __init__(self,
                 number_of_classes: int = 2,
                 image_width: int = 50,
                 image_height: int = 50):
        self.__model = tf.keras.applications.ResNet152V2(
            input_shape=(image_height, image_width, 3),
            weights=None,
            classes=number_of_classes
        )
        self.__model_name = 'resnet152v2'

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


class MobileNet:
    # docs at https://keras.io/api/applications/mobilenet/#mobilenetv2-function
    def __init__(self,
                 number_of_classes: int = 2,
                 image_width: int = 50,
                 image_height: int = 50):
        self.__model = tf.keras.applications.MobileNetV2(
            input_shape=(image_height, image_width, 3),
            weights=None,
            classes=number_of_classes
        )
        self.__model_name = 'mobilenetv2'

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


class EfficientNet:
    # docs at https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b0-function
    def __init__(self,
                 number_of_classes: int = 2,
                 image_width: int = 50,
                 image_height: int = 50):
        self.__model = tf.keras.applications.EfficientNetV2B0(
            input_shape=(image_height, image_width, 3),
            weights=None,
            classes=number_of_classes
        )
        self.__model_name = 'efficientnetv2b0'

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
