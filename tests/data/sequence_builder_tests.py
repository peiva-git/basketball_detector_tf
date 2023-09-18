import unittest

import tensorflow as tf

from parameterized import parameterized
from basketballdetector.data import ClassificationSequenceBuilder


class SequenceBuilderTestCase(unittest.TestCase):
    __NUMBER_OF_SAMPLES = 18
    __BATCH_SIZE = 3
    __VALIDATION_PERCENTAGE = 0.5
    __BUILDER = ClassificationSequenceBuilder(
        '../../assets/test-sample-data-classification',
        __BATCH_SIZE,
        __VALIDATION_PERCENTAGE
    )

    @parameterized.expand([
        __BUILDER.training_sequence,
        __BUILDER.validation_sequence
    ])
    def test_sequence_length(self, sequence: tf.keras.utils.Sequence):
        samples_divide_factor = 1 // self.__VALIDATION_PERCENTAGE
        self.assertEqual(len(sequence), (self.__NUMBER_OF_SAMPLES // samples_divide_factor) // self.__BATCH_SIZE)

    @parameterized.expand([
        __BUILDER.training_sequence,
        __BUILDER.validation_sequence
    ])
    def test_class_names(self, sequence: tf.keras.utils.Sequence):
        self.assertTrue(
            sequence.class_names == ['ball', 'no_ball'],
            'invalid object classes'
        )

    @parameterized.expand([
        __BUILDER.training_sequence,
        __BUILDER.validation_sequence
    ])
    def test_sequence_labels(self, sequence: tf.keras.utils.Sequence):
        self.assertIn(
            sequence[0].get_single_element()[1].numpy(), [0, 1],
            'invalid class label'
        )
