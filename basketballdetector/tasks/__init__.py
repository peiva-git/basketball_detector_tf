"""
This package contains the all the modules used to perform the training and predicting tasks,
with all the additional required postprocessing operations.
"""

from .predicting import write_image_sequence_from_video, write_detections_video
from .training.train_mobilenet_classifier import train_and_save_model as train_mobilenet
from .training.train_efficientnet_classifier import train_and_save_model as train_efficientnet
