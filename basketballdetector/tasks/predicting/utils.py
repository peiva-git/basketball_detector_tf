import pathlib

import cv2 as cv
import numpy as np

from basketballdetector.tasks.predicting.predict_ball_locations import __find_max_pixel


def divide_frame_into_patches(frame: np.ndarray, stride: int = 5, window_size: int = 50) -> [(int, int, np.ndarray)]:
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
                                     predictions, window_size: int = 50,
                                     threshold: float = 0.5) -> np.ndarray:
    for index, (height_coordinate, width_coordinate, image_patch) in enumerate(patches_with_positions):
        prediction = predictions[index]
        if prediction[0] >= threshold:
            # more likely that the patch is a ball
            cv.rectangle(
                frame,
                (width_coordinate, height_coordinate),
                (width_coordinate + window_size, height_coordinate + window_size),
                color=(0, 255, 0)
            )
        else:
            # more likely that the patch is not a ball
            pass
    return frame


def annotate_frame(frame: np.ndarray,
                   heatmap: np.ndarray,
                   threshold_delta: int = 10,
                   margin: int = 0) -> ((int, int, int, int), np.ndarray):
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
