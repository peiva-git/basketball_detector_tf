import unittest

import tensorflow as tf

from basketballdetector.models import MobileNet, ResNet152V2, EfficientNet


class ClassificationModelTestCase(unittest.TestCase):
    __NUMBER_OF_CLASSES = 2
    __WIDTH = 112
    __HEIGHT = 112
    __MOBILENET_MODEL = MobileNet(__NUMBER_OF_CLASSES, __WIDTH, __HEIGHT)
    __RESNET_MODEL = ResNet152V2(__NUMBER_OF_CLASSES, __WIDTH, __HEIGHT)
    __EFFICIENTNET_MODEL = EfficientNet(__NUMBER_OF_CLASSES, __WIDTH, __HEIGHT)

    def test_model_name(self):
        self.assertEqual(
            self.__MOBILENET_MODEL.model_name,
            'mobilenetv2',
            'incorrect model name'
        )
        self.assertEqual(
            self.__RESNET_MODEL.model_name,
            'resnet152v2',
            'incorrect model name'
        )
        self.assertEqual(
            self.__EFFICIENTNET_MODEL.model_name,
            'efficientnetv2b0',
            'incorrect model name'
        )

    def test_model(self):
        self.assertIsInstance(
            self.__MOBILENET_MODEL.model,
            tf.keras.models.Model,
            'incorrect model type'
        )
        self.assertIsInstance(
            self.__RESNET_MODEL.model,
            tf.keras.models.Model,
            'incorrect model type'
        )
        self.assertIsInstance(
            self.__EFFICIENTNET_MODEL.model,
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
