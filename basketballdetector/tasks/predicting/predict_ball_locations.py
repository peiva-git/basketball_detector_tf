import pathlib
import time
from collections import defaultdict
from itertools import product
from statistics import mean

import cv2 as cv
import tensorflow as tf
import numpy as np

from basketballdetector.data import PatchesSequence
from basketballdetector.tasks.predicting import divide_frame_into_patches, annotate_frame


def obtain_predictions(frame: np.ndarray,
                       model_path: str,
                       stride: int = 5,
                       window_size: int = 50) -> ([int, int, np.ndarray], [int, int]):
    model_path = pathlib.Path(model_path)
    patches_with_positions = divide_frame_into_patches(frame, stride=stride, window_size=window_size)
    patches_only = [element[2] for element in patches_with_positions]
    model = tf.keras.models.load_model(str(model_path))
    patches_sequence = PatchesSequence(patches_only)
    predictions = model.predict(patches_sequence)
    return patches_with_positions, predictions


def obtain_heatmap(frame: np.ndarray,
                   predictions: list[(int, int)],
                   patches_with_positions: list[(int, int, np.ndarray)],
                   window_size: int = 50):
    frame_height, frame_width, _ = frame.shape
    heatmap = np.zeros((frame_height, frame_width), np.float32)
    patch_indexes_by_pixel = defaultdict(set)
    print('Building pixel -> indexes dictionary...')
    __map_pixels_to_patch_indexes(patch_indexes_by_pixel, patches_with_positions, window_size)
    for row, column in product(range(frame_height), range(frame_width)):
        pixel_indexes = patch_indexes_by_pixel[(row, column)]
        print(f'Found indexes for pixel ({row},{column})')
        if len(pixel_indexes) != 0:
            patches_ball_probabilities = [predictions[patch_index][0] for patch_index in pixel_indexes]
            pixel_ball_probability = sum(patches_ball_probabilities) / len(pixel_indexes)
            heatmap[row, column] = pixel_ball_probability
    heatmap_rescaled = heatmap * 255
    return heatmap_rescaled.astype(np.uint8, copy=False)


def __map_pixels_to_patch_indexes(patch_indexes_by_pixel: dict,
                                  patches_with_positions: list[(int, int, np.ndarray)],
                                  window_size: int):
    for index, (patch_position_y, patch_position_x, _) in enumerate(patches_with_positions):
        __iterate_over_patch(index, patch_indexes_by_pixel, patch_position_x, patch_position_y, window_size)


def __iterate_over_patch(index, patch_indexes_by_pixel, patch_position_x, patch_position_y, window_size):
    for row, column in product(range(patch_position_y, patch_position_y + window_size),
                               range(patch_position_x, patch_position_x + window_size)):
        patch_indexes_by_pixel[(row, column)].add(index)


def __find_max_pixel(heatmap: np.ndarray) -> (int, int):
    max_index = heatmap.argmax()
    _, heatmap_width = heatmap.shape
    return max_index - int(max_index / heatmap_width) * heatmap_width, int(max_index / heatmap_width)


def write_detections_video(input_video_path: str,
                           target_video_path: str,
                           model_path: str):
    input_path = pathlib.Path(input_video_path)
    target_path = pathlib.Path(target_video_path)
    model_path = pathlib.Path(model_path)
    capture = cv.VideoCapture(str(input_path))
    out = cv.VideoWriter(str(target_path), fourcc=0, fps=0, frameSize=(1920, 1080))
    if not capture.isOpened():
        print("Can't open video file")
        return
    counter = 1
    frame_processing_times = []
    while counter <= 100:
        ret, image = capture.read()
        if not ret:
            print("Can't read next frame (stream end?). Exiting...")
            break
        print(f'Processing frame {counter} out of {int(capture.get(cv.CAP_PROP_FRAME_COUNT))}')
        start = time.time()
        patches_with_positions, patches_predictions = obtain_predictions(
            image, str(model_path), stride=10, window_size=50
        )
        heatmap = obtain_heatmap(image, patches_predictions, patches_with_positions, window_size=50)
        annotate_frame(image, heatmap)
        out.write(image)
        end = time.time()
        print(f'Took {end - start} seconds to process frame {counter}'
              f' out of {int(capture.get(cv.CAP_PROP_FRAME_COUNT))}')
        frame_processing_times.append(end - start)
        print(f'Average processing speed: {mean(frame_processing_times)} seconds')
        counter += 1
        # cv.imshow(f'frame {counter}', image)
        # if cv.waitKey(1) == ord('q'):
        #     break
    capture.release()
    out.release()
    cv.destroyAllWindows()


def write_image_sequence_from_video(input_video_path: str,
                                    target_directory_path: str,
                                    model_path: str):
    input_path = pathlib.Path(input_video_path)
    target_path = pathlib.Path(target_directory_path)
    model_path = pathlib.Path(model_path)
    capture = cv.VideoCapture(str(input_path))
    if not capture.isOpened():
        print("Can't open video file")
        return
    counter = 1
    frame_processing_times = []
    while counter <= 10:
        ret, image = capture.read()
        if not ret:
            print("Can't read next frame (stream end?). Exiting...")
            break
        print(f'Processing frame {counter} out of {10}')
        start = time.time()
        print('Obtaining predictions...')
        patches_with_positions, patches_predictions = obtain_predictions(
            image, str(model_path), window_size=50, stride=10
        )
        print('Building heatmap from predictions...')
        heatmap = obtain_heatmap(image, patches_predictions, patches_with_positions, window_size=50)
        _, mask = annotate_frame(image, heatmap)
        cv.imwrite(str(target_path / f'frame_{counter}.png'), image)
        cv.imwrite(str(target_path / f'heatmap_{counter}.png'), heatmap)
        cv.imwrite(str(target_path / f'mask_{counter}.png'), mask)
        end = time.time()
        print(f'Took {end - start} seconds to process frame {counter}'
              f' out of {int(capture.get(cv.CAP_PROP_FRAME_COUNT))}')
        frame_processing_times.append(end - start)
        print(f'Average processing speed: {mean(frame_processing_times)} seconds')
        counter += 1
        # cv.imshow(f'frame {counter}', image)
        # if cv.waitKey(1) == ord('q'):
        #     break
    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # with ~4ms inference time on a single patch, a whole image is evaluated in approx. 5 minutes
    # with a window size of 50 and a stride of 5
    # with a window size of 100 and a stride of 10, an image is evaluated in approx. 1 minute
    # these values are estimated based on the mobilenetv2 inference time measurements displayed here
    # https://keras.io/api/applications/#available-models
    write_image_sequence_from_video(input_video_path='/home/peiva/experiments/test_videos/final_cut.mp4',
                                    target_directory_path='/home/peiva/experiments/',
                                    model_path='/home/peiva/mobilenet/first_test/models/Keras_v3/mobilenetv2.keras')
    # write_image_sequence_from_video(input_video_path='/home/ubuntu/test_videos/final_cut.mp4',
    #                                 target_directory_path='/home/ubuntu/test_videos',
    #                                 model_path='/home/ubuntu/basketball_detector/out/models/Keras_v3/mobilenetv2.keras')
    # write_detections_video(input_video_path='/home/ubuntu/test_videos/final_cut.mp4',
    #                        target_video_path='home/ubuntu/test_videos/annotated.mp4',
    #                        model_path='/home/ubuntu/basketball_detector/out/models/Keras_v3/mobilenetv2.keras')
