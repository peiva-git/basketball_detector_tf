import unittest

import numpy as np

from basketballdetector.data import ClassificationSequenceBuilder


class ClassificationSequenceBuilderTestCase(unittest.TestCase):
    __NUMBER_OF_SAMPLES = 18
    __BATCH_SIZE = 3
    __VALIDATION_PERCENTAGE = 0.5
    __BUILDER = ClassificationSequenceBuilder(
        '../assets/test-sample-data-classification',
        __BATCH_SIZE,
        __VALIDATION_PERCENTAGE
    )

    def test_sequence_length(self):
        samples_divide_factor = 1 // self.__VALIDATION_PERCENTAGE
        self.assertEqual(
            len(self.__BUILDER.training_sequence),
            (self.__NUMBER_OF_SAMPLES // samples_divide_factor) // self.__BATCH_SIZE,
            'incorrect training sequence size'
        )
        self.assertEqual(
            len(self.__BUILDER.validation_sequence),
            (self.__NUMBER_OF_SAMPLES // samples_divide_factor) // self.__BATCH_SIZE,
            'incorrect validation sequence size'
        )

    def test_class_names(self):
        self.assertTrue(
            np.all(self.__BUILDER.training_sequence.class_names == ['ball', 'no_ball']),
            'invalid object classes in training sequence'
        )
        self.assertTrue(
            np.all(self.__BUILDER.validation_sequence.class_names == ['ball', 'no_ball']),
            'invalid object classes in validation sequence'
        )
