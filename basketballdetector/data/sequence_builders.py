"""
This module contains all the classes used to build sequence datasets for this project.
Sequence datasets are based on the Sequence Keras class.
More information can be found [here](https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence).
It is assumed that the datasets' structure is as follows:
1. **For classification datasets**: starting from the root directory, the images are organized in the following
directory structure:
    1. [season]/[match]/ball for patches that represent a ball
    2. [season]/[match]/no_ball for patches that don't represent a ball

The same game may be played in different seasons.
2. **For segmentation datasets**: starting from the root directory, the images and masks are organized in the following
directory structure:
    1. [season]/[match]/frames for the video frames
    2. [season]/[match]/masks for the corresponding masks

The same game may be played in different seasons. The correspondences between frames and masks are determined by
the frame index.
"""

import pathlib
import glob
import math
import os

from random import shuffle

import tensorflow as tf
import numpy as np

from .utils import decode_image


class ClassificationSequenceBuilder:
    """
    Class used to build a classification sequence dataset for this project
    """
    def __init__(self,
                 data_directory: str,
                 batch_size: int,
                 image_width: int,
                 image_height: int,
                 validation_percentage: float = 0.2):
        """
        Class constructor
        :param data_directory: Dataset root directory
        :param batch_size: Size of each input batch of this `Sequence`
        :param image_width: Target image width
        :param image_height: Target image height
        :param validation_percentage: Percentage of images to be used as validation data
        """
        self.__image_width = image_width
        self.__image_height = image_height
        data_path = pathlib.Path(data_directory)
        print('Gathering all image paths...')
        image_paths = [
            image_path
            for image_path in glob.iglob(str(data_path / '*/*/*/*.png'))
        ]
        shuffle(image_paths)
        print(f'Found {len(image_paths)} images')
        validation_size = int(len(image_paths) * validation_percentage)
        validation_paths = image_paths[:validation_size]
        training_paths = image_paths[validation_size:]
        print(f'{len(validation_paths)} images in validation dataset')
        print(f'{len(training_paths)} images in training dataset')
        self.__training_sequence = _ClassificationSequence(
            data_directory,
            training_paths,
            batch_size,
            image_width,
            image_height
        )
        self.__validation_sequence = _ClassificationSequence(
            data_directory,
            validation_paths,
            batch_size,
            image_width,
            image_height
        )

    @property
    def training_sequence(self):
        """
        This method returns the training sequence.
        It is a portion of the whole (shuffled) dataset, obtained based on the validation_percentage parameter.
        Instead of loading the whole dataset into memory before training, the Sequence class only loads subsequent
        batches of data.
        :return: The training sequence
        """
        return self.__training_sequence

    @property
    def validation_sequence(self):
        """
        This method returns the validation sequence.
        It is a portion of the whole (shuffled) dataset, obtained based on the validation_percentage parameter.
        Instead of loading the whole dataset into memory before validation, the Sequence class only loads subsequent
        batches of data.
        :return: The validation sequence
        """
        return self.__validation_sequence


class _ClassificationSequence(tf.keras.utils.Sequence):
    def __init__(self,
                 data_directory: str,
                 images_paths: list[str],
                 batch_size: int,
                 image_width: int,
                 image_height: int):
        data_path = pathlib.Path(data_directory)
        self.__image_width = image_width
        self.__image_height = image_height
        self.__batch_size = batch_size
        self.__image_paths = images_paths
        self.__class_names = np.unique(sorted([item.name for item in data_path.glob('*/*/*')]))
        print(f'Found classes {self.__class_names}')

    def __getitem__(self, index):
        low = index * self.__batch_size
        high = min(low + self.__batch_size, len(self.__image_paths))
        batch_paths = self.__image_paths[low:high]
        batch_labels = [
            self.__get_label(tf.constant(image_path))
            for image_path in batch_paths
        ]
        batch_images = [
            self.__get_image(tf.constant(image_path))
            for image_path in batch_paths
        ]
        return tf.stack(batch_images), tf.stack(batch_labels)

    def __len__(self):
        return math.ceil(len(self.__image_paths) / self.__batch_size)

    def __get_label(self, file_path: tf.Tensor) -> int:
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.__class_names
        return tf.one_hot(tf.argmax(one_hot), 2)

    def __get_image(self, file_path: tf.Tensor):
        image_data = tf.io.read_file(file_path)
        return decode_image(image_data, image_width=self.__image_width, image_height=self.__image_height, channels=3)

    @property
    def class_names(self):
        return self.__class_names


class PatchesSequence(tf.keras.utils.Sequence):
    """
    This class is used to build a sequence from the patches obtained from the
    `basketballdetector.tasks.predicting.utils.divide_frame_into_patches` function.
    The sequence can then be used to perform prediction using a Keras model.
    """
    def __init__(self, patches: list[np.ndarray], batch_size: int = 64):
        """
        Class constructor
        :param patches: List of patches' data
        :param batch_size: Size of each input batch of this `Sequence`
        """
        self.__patches = patches
        self.__batch_size = batch_size

    def __getitem__(self, index):
        low = index * self.__batch_size
        high = min(low + self.__batch_size, len(self.__patches))
        patches_batch = self.__patches[low:high]
        return np.array([patch for patch in patches_batch])

    def __len__(self):
        return math.ceil(len(self.__patches) / self.__batch_size)
