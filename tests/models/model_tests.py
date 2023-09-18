import unittest

import tensorflow as tf

from basketballdetector.models import MobileNet, ResNet152V2, EfficientNet
from parameterized import parameterized


class ClassificationModelTestCase(unittest.TestCase):
    __NUMBER_OF_CLASSES = 2
    __WIDTH = 112
    __HEIGHT = 112
    __MOBILENET_MODEL = MobileNet(__NUMBER_OF_CLASSES, __WIDTH, __HEIGHT)
    __RESNET_MODEL = ResNet152V2(__NUMBER_OF_CLASSES, __WIDTH, __HEIGHT)
    __EFFICIENTNET_MODEL = EfficientNet(__NUMBER_OF_CLASSES, __WIDTH, __HEIGHT)

    @parameterized.expand([
        (__MOBILENET_MODEL.model_name, 'mobilenetv2'),
        (__RESNET_MODEL.model_name, 'resnet152v2'),
        (__EFFICIENTNET_MODEL.model_name, 'efficientnetv2b0')
    ])
    def test_model_name(self, model_name: str, expected_name: str):
        self.assertEqual(
            model_name,
            expected_name,
            'incorrect model name'
        )

    @parameterized.expand([
        __MOBILENET_MODEL.model,
        __RESNET_MODEL.model,
        __EFFICIENTNET_MODEL.model
    ])
    def test_model(self, model: tf.keras.models.Model):
        self.assertIsInstance(
            model,
            tf.keras.models.Model,
            'incorrect model type'
        )

    def test_model_output(self):
        self.assertTupleEqual(
            self.__MOBILENET_MODEL.model.output_shape,
            (None, self.__NUMBER_OF_CLASSES),
            'incorrect output shape')
        self.assertTupleEqual(
            self.__RESNET_MODEL.model.output_shape,
            (None, self.__NUMBER_OF_CLASSES),
            'incorrect output shape'
        )
        self.assertTupleEqual(
            self.__EFFICIENTNET_MODEL.model.output_shape,
            (None, self.__NUMBER_OF_CLASSES),
            'incorrect output shape'
        )

    def test_model_input(self):
        self.assertTupleEqual(
            self.__MOBILENET_MODEL.model.input_shape,
            (None, self.__HEIGHT, self.__WIDTH, 3),
            'incorrect input shape'
        )
        self.assertTupleEqual(
            self.__RESNET_MODEL.model.input_shape,
            (None, self.__HEIGHT, self.__WIDTH, 3),
            'incorrect input shape'
        )
        self.assertTupleEqual(
            self.__EFFICIENTNET_MODEL.model.input_shape,
            (None, self.__HEIGHT, self.__WIDTH, 3),
            'incorrect input shape'
        )
