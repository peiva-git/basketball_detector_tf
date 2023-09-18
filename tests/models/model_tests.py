import unittest

import tensorflow as tf

from basketballdetector.models import MobileNet


class MobileNetModelTestCase(unittest.TestCase):
    __NUMBER_OF_CLASSES = 2
    __WIDTH = 112
    __HEIGHT = 112
    __MODEL = MobileNet(number_of_classes=__NUMBER_OF_CLASSES, image_width=__WIDTH, image_height=__HEIGHT)

    def test_model_name(self):
        self.assertEqual(
            self.__MODEL.model_name,
            'mobilenetv2',
            'incorrect model name'
        )

    def test_model(self):
        self.assertIsInstance(
            self.__MODEL.model,
            tf.keras.models.Model,
            'incorrect model type'
        )

    def test_model_output(self):
        self.assertTupleEqual(
            self.__MODEL.model.output_shape,
            (None, 2),
            'incorrect output shape')

    def test_model_input(self):
        self.assertTupleEqual(
            self.__MODEL.model.input_shape,
            (None, self.__HEIGHT, self.__WIDTH),
            'incorrect input shape'
        )
