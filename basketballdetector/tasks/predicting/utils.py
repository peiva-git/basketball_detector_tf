"""
This module contains all the utility functions used by the other modules in this package
"""

import pathlib

import cv2 as cv
import numpy as np


def divide_frame_into_patches(frame: np.ndarray, stride: int = 5, window_size: int = 50) -> [(int, int, np.ndarray)]:
    """
    Divide a frame into square patches of size window_size, with a stride of stride pixels.
    If the stride is smaller than the window_size, the patches are overlapping.
    In case a patch has any pixels that are outside the frame range, it is discarded.
    :param frame: The frame to obtain the patches from
    :param stride: Stride size in pixels
    :param window_size: Window size in pixels
    :return: A list of patches. For each patch, the coordinates of the upper left corner in frame-space
    are also returned. Therefore, each element of the list has the following form:
    (patch_row_coordinate, patch_column_coordinate, patch_data)
    """
    # could try out with a stride of 10 and a window_size of 100 as well
    # the origin of the coordinates' system is in the upper left corner of the image
    # with the x-axis facing to the right, and the y-axis facing down
    height, width, _ = frame.shape
    position_height = 0
    position_width = 0
    number_of_width_windows = int(width / stride) - int(window_size / stride)
    number_of_height_windows = int(height / stride) - int(window_size / stride)

    patches = []
    for window_height_index in range(number_of_height_windows):
        for window_width_index in range(number_of_width_windows):
            current_patch = frame[
                            position_height:position_height + window_size,
                            position_width:position_width + window_size
                            ]
            current_patch_rgb = cv.cvtColor(current_patch, cv.COLOR_BGR2RGB)
            patches.append((position_height, position_width, current_patch_rgb))
            position_width += stride
        position_width = 0
        position_height += stride

    return patches


def write_frame_patches_to_disk(frame: np.ndarray,
                                target_directory: str,
                                stride: int = 5,
                                window_size: int = 50,
                                verbose: bool = False):
    """
    Write all the patches obtained via `divide_frame_into_patches` to disk.
    :param frame: The frame to obtain the patches from
    :param target_directory: Where to write all the obtained patches
    :param stride: Stride size in pixels
    :param window_size: Window size in pixels
    :param verbose: Print additional feedback to stdout
    :return: None
    """
    target = pathlib.Path(target_directory)
    count = 1
    image_patches = divide_frame_into_patches(frame, stride, window_size)
    for position_y, position_x, patch in image_patches:
        patch = cv.cvtColor(patch, cv.COLOR_RGB2BGR)
        cv.imwrite(str(target / f'patch_x{position_x}_y{position_y}.png'), patch)
        if verbose:
            print(f'Written image {count} out of {len(image_patches)}')
        count += 1


def annotate_frame_with_ball_patches(frame: np.ndarray,
                                     patches_with_positions: list[(int, int, np.ndarray)],
                                     predictions: np.ndarray,
                                     window_size: int = 50,
                                     threshold: float = 0.5) -> np.ndarray:
    """
    Annotate a frame with all the patches that qualify as ball candidates, based on the provided predictions.
    Note that this functions assumes that the predictions and the patches have the same ordering.

    :param frame: Frame to annotate.
    :param patches_with_positions: List of patches obtained from `divide_frame_into_patches`
    :param predictions: Predictions obtained from
    `basketballdetector.tasks.predicting.predict_ball_locations.obtain_predictions`
    :param window_size: Window size in pixels
    :param threshold: A prediction score above this threshold will be considered a ball candidate.
    Note that the prediction scores are results of a softmax layers, so they represent a probability distribution.
    :return: The annotated frame.
    """
    annotated_frame = frame.copy()
    for index, (height_coordinate, width_coordinate, image_patch) in enumerate(patches_with_positions):
        prediction = predictions[index]
        if prediction[0] >= threshold:
            # more likely that the patch is a ball
            cv.rectangle(
                annotated_frame,
                (width_coordinate, height_coordinate),
                (width_coordinate + window_size, height_coordinate + window_size),
                color=(0, 255, 0)
            )
        else:
            # more likely that the patch is not a ball
            pass
    return annotated_frame


def annotate_frame(frame: np.ndarray,
                   heatmap: np.ndarray,
                   threshold_delta: int = 10,
                   margin: int = 0) -> ((int, int, int, int), np.ndarray):
    """
    Annotate a frame with the ball location, based on the provided heatmap.
    Note that the frame will be modified by this function
    :param frame: The frame to annotate
    :param heatmap: Detection heatmap
    :param threshold_delta: Coordinates whose value is above the heatmap's max value - threshold_delta in the heatmap
    will be considered as the ball coordinates in the detection process. The area masked by these coordinates,
    with the added margin, will represent the detected Region of Interest
    :param margin: Margin to add to the detected ball area
    :return: The bounding box and annotated frame. The bounding box has the following form:
    (top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y)
    """
    max_pixel = __find_max_pixel(heatmap)
    heatmap_height, heatmap_width = heatmap.shape
    mask = np.zeros((heatmap_height + 2, heatmap_width + 2), np.uint8)
    _, _, _, bounding_box = cv.floodFill(
        image=heatmap, mask=mask, seedPoint=max_pixel, newVal=255, loDiff=threshold_delta,
        flags=8 | (255 << 8) | cv.FLOODFILL_FIXED_RANGE | cv.FLOODFILL_MASK_ONLY
    )
    cv.rectangle(
        frame,
        (bounding_box[0] - margin, bounding_box[1] - margin),
        (bounding_box[0] + bounding_box[2] + margin, bounding_box[1] + bounding_box[3] + margin),
        color=(0, 255, 0)
    )
    return bounding_box, mask


def __find_max_pixel(heatmap: np.ndarray) -> (int, int):
    max_index = heatmap.argmax()
    _, heatmap_width = heatmap.shape
    return max_index - int(max_index / heatmap_width) * heatmap_width, int(max_index / heatmap_width)
