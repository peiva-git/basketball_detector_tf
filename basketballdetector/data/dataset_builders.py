"""
This module contains all the classes used to build the datasets for this project.
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

import glob
import pathlib
import os.path
from typing import Any

import tensorflow as tf
import numpy as np

from .utils import decode_image


def _configure_for_performance(dataset: tf.data.Dataset, buffer_size: int, batch_size: int) -> tf.data.Dataset:
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


class ClassificationDatasetBuilder:
    """
    Class used to build a classification dataset for this project
    """
    def __init__(self,
                 data_directory: str,
                 image_width: int,
                 image_height: int,
                 validation_percentage: float = 0.2):
        """
        Class constructor
        :param data_directory: Dataset root directory
        :param image_width: Target image width
        :param image_height: Target image height
        :param validation_percentage: Percentage of images to be used as validation data
        """
        self.__image_width = image_width
        self.__image_height = image_height
        data_path = pathlib.Path(data_directory)
        all_images_dataset = tf.data.Dataset.list_files(str(data_path / '*/*/*/*'), seed=2023)
        self.__image_count = tf.data.experimental.cardinality(all_images_dataset).numpy()
        self.__class_names = np.unique(sorted([item.name for item in data_path.glob('*/*/*')]))
        print('Found the following classes: ', self.__class_names)

        validation_size = int(self.__image_count * validation_percentage)
        self.__train_dataset = all_images_dataset.skip(validation_size)
        self.__validation_dataset = all_images_dataset.take(validation_size)
        print(self.__train_dataset.cardinality().numpy(), 'images in training dataset')
        print(self.__validation_dataset.cardinality().numpy(), 'images in validation dataset')

        self.__train_dataset = self.__train_dataset.map(
            self.__get_image_label_pair_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        self.__validation_dataset = self.__validation_dataset.map(
            self.__get_image_label_pair_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    @property
    def number_of_images(self) -> int:
        """
        This method returns the number of images contained in this dataset
        :return: The number of images loaded into the dataset
        """
        return self.__image_count

    @property
    def class_names(self) -> [str, str]:
        """
        This method returns the classifier's classes obtained from the directory structure
        :return: The classes to be used by the classifier
        """
        return self.__class_names

    @property
    def train_dataset(self) -> tf.data.Dataset:
        """
        This method returns the training dataset.
        It is a portion of the whole (shuffled) dataset, obtained based on the validation_percentage parameter
        :return: The training dataset
        """
        return self.__train_dataset

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        """
        This method returns the validation dataset.
        It is a portion of the whole (shuffled) dataset, obtained based on the validation_percentage parameter
        :return: The validation dataset
        """
        return self.__validation_dataset

    def __get_label(self, file_path: tf.Tensor) -> int:
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.__class_names
        return tf.argmax(one_hot)

    def __get_image_label_pair_from_path(self, file_path: tf.Tensor) -> (Any, int):
        label = self.__get_label(file_path)
        image_data = tf.io.read_file(file_path)
        image = decode_image(image_data, image_width=self.__image_width, image_height=self.__image_height)
        return image, label

    def configure_datasets_for_performance(self, shuffle_buffer_size: int = 10000, input_batch_size: int = 32):
        """
        This method sets some optimizations for the training and validation datasets, in order to improve performance
        :param shuffle_buffer_size: Size of the shuffle buffer to be used with the training dataset
        :param input_batch_size: Input batch size for both datasets
        :return: None
        """
        self.__train_dataset = _configure_for_performance(self.__train_dataset, shuffle_buffer_size, input_batch_size)
        self.__validation_dataset = self.__validation_dataset.cache()
        self.__validation_dataset = self.__validation_dataset.batch(input_batch_size)
        self.__validation_dataset = self.__validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


class SegmentationDatasetBuilder:
    """
    Class used to build a segmentation dataset for this project
    """
    def __init__(self,
                 data_directory: str,
                 image_width: int,
                 image_height: int,
                 validation_percentage: float = 0.2):
        """
        Class constructor
        :param data_directory: Dataset root directory
        :param image_width: Target image width
        :param image_height: Target image height
        :param validation_percentage: Percentage of images to be used as validation data
        """
        self.__image_width = image_width
        self.__image_height = image_height
        data_path = pathlib.Path(data_directory)
        input_image_paths = [
            input_image_path
            for input_image_path in glob.iglob(str(data_path / '*/*/frames/*.png'))
        ]
        mask_image_paths = [
            mask_image_path
            for mask_image_path in glob.iglob(str(data_path / '*/*/masks/*.png'))
        ]
        self.__image_count = len(input_image_paths)
        validation_size = int(self.__image_count * validation_percentage)
        print(len(input_image_paths), 'frames, with', len(mask_image_paths), 'corresponding ground truth masks')
        print(validation_size, 'samples will be set aside as validation data')

        if len(input_image_paths) != len(mask_image_paths):
            raise ValueError('The number of frames is different than the number of ground truth masks, aborting')

        samples_datasets = []
        masks_datasets = []
        for match_directory_path in glob.glob(str(data_path / '*/*/')):
            match_directory = pathlib.Path(match_directory_path)
            match_input_image_paths = [
                match_input_image_path
                for match_input_image_path in glob.iglob(str(match_directory / 'frames/*.png'))
            ]
            match_mask_image_paths = [
                match_mask_image_path
                for match_mask_image_path in glob.iglob(str(match_directory / 'masks/*.png'))
            ]
            match_input_image_paths.sort(key=lambda file_path: int(file_path.split('_')[-1].split('.')[-2]))
            match_mask_image_paths.sort(key=lambda file_path: int(file_path.split('_')[-1].split('.')[-2]))

            samples_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(match_input_image_paths))
            masks_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(match_mask_image_paths))
            samples_datasets.append(samples_dataset)
            masks_datasets.append(masks_dataset)

        samples = samples_datasets[0]
        for dataset in samples_datasets[1:]:
            samples = samples.concatenate(dataset)

        masks = masks_datasets[0]
        for dataset in masks_datasets[1:]:
            masks = masks.concatenate(dataset)

        filenames_dataset = tf.data.Dataset.zip((samples, masks))
        filenames_dataset = filenames_dataset.shuffle(buffer_size=filenames_dataset.cardinality(),
                                                      reshuffle_each_iteration=False)
        dataset = filenames_dataset.map(
            self.__get_frame_and_mask_from_filepaths,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        print(f'Found {dataset.cardinality().numpy()} frames in total')
        self.__train_dataset = dataset.skip(validation_size)
        self.__validation_dataset = dataset.take(validation_size)
        print(f'{self.__train_dataset.cardinality().numpy()} frames in training dataset')
        print(f'{self.__validation_dataset.cardinality().numpy()} frames in validation dataset')

    def __get_frame_and_mask_from_filepaths(self,
                                            frame_filepath: tf.Tensor,
                                            mask_filepath: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        frame_data = tf.io.read_file(frame_filepath)
        mask_data = tf.io.read_file(mask_filepath)
        frame = decode_image(frame_data, image_width=self.__image_width, image_height=self.__image_height)
        mask = decode_image(mask_data, image_width=self.__image_width, image_height=self.__image_height, channels=1)
        # 2 as the number of classes
        mask = tf.one_hot(tf.cast(mask, tf.uint8), 2)
        mask = tf.squeeze(mask)
        mask = tf.cast(mask, tf.float32)
        return frame, mask

    @property
    def train_dataset(self) -> tf.data.Dataset:
        """
        This method returns the training dataset.
        It is a portion of the whole (shuffled) dataset, obtained based on the validation_percentage parameter
        :return: The training dataset
        """
        return self.__train_dataset

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        """
        This method returns the validation dataset.
        It is a portion of the whole (shuffled) dataset, obtained based on the validation_percentage parameter
        :return: The validation dataset
        """
        return self.__validation_dataset

    @property
    def number_of_samples(self) -> int:
        """
        This method returns the number of samples contained in this dataset
        :return: The number of image, mask pairs loaded into the dataset
        """
        return self.__train_dataset.cardinality().numpy() + self.__validation_dataset.cardinality().numpy()

    def configure_datasets_for_performance(self, shuffle_buffer_size: int = 1000, input_batch_size: int = 10):
        """
        This method sets some optimizations for the training and validation datasets, in order to improve performance
        :param shuffle_buffer_size: Size of the shuffle buffer to be used with the training dataset
        :param input_batch_size: Input batch size for both datasets
        :return: None
        """
        self.__train_dataset = _configure_for_performance(self.__train_dataset, shuffle_buffer_size, input_batch_size)
        self.__validation_dataset = self.__validation_dataset.cache()
        self.__validation_dataset = self.__validation_dataset.batch(input_batch_size)
        self.__validation_dataset = self.__validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
