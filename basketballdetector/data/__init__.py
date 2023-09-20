"""
This package contains all the modules related to data manipulation and dataset generation.
The `basketballdetector.data.dataset_builders.ClassificationDatasetBuilder` and the
`basketballdetector.data.dataset_builders.SegmentationDatasetBuilder` classes
both load the entire dataset into memory before training,
while the `basketballdetector.data.sequence_builders.ClassificationSequenceBuilder` and the
`basketballdetector.data.sequence_builers.PatchesSequence` rely on the Keras Sequence class.
More information can be found here https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
"""

from .dataset_builders import ClassificationDatasetBuilder, SegmentationDatasetBuilder
from .sequence_builders import ClassificationSequenceBuilder, PatchesSequence
