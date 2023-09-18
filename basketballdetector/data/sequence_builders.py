import pathlib
import glob
import math
import os

from random import shuffle

import tensorflow as tf
import numpy as np

from basketballdetector.data.utils import decode_image


class ClassificationSequenceBuilder:
    def __init__(self, data_directory: str, batch_size: int, validation_percentage: float = 0.2):
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
        self.__training_sequence = _ClassificationSequence(data_directory, training_paths, batch_size)
        self.__validation_sequence = _ClassificationSequence(data_directory, validation_paths, batch_size)

    @property
    def training_sequence(self):
        return self.__training_sequence

    @property
    def validation_sequence(self):
        return self.__validation_sequence


class _ClassificationSequence(tf.keras.utils.Sequence):
    def __init__(self, data_directory: str, images_paths: list[str], batch_size: int):
        data_path = pathlib.Path(data_directory)
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

    @staticmethod
    def __get_image(file_path: tf.Tensor):
        image_data = tf.io.read_file(file_path)
        return decode_image(image_data, image_width=112, image_height=112, channels=3)

    @property
    def class_names(self):
        return self.__class_names


class PatchesSequence(tf.keras.utils.Sequence):

    def __init__(self, patches: list[np.ndarray], batch_size: int = 64):
        self.__patches = patches
        self.__batch_size = batch_size

    def __getitem__(self, index):
        low = index * self.__batch_size
        high = min(low + self.__batch_size, len(self.__patches))
        patches_batch = self.__patches[low:high]
        return np.array([patch for patch in patches_batch])

    def __len__(self):
        return math.ceil(len(self.__patches) / self.__batch_size)
