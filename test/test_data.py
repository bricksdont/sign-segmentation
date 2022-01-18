#! /usr/bin/python3

import os

import numpy as np
import tensorflow as tf

from unittest import TestCase
from typing import List, Optional

from test.common import _create_tmp_working_directory

from sign_language_segmentation.data import DataLoader, random_slice_example, truncate_example


def get_random_raw_example(frames_min: Optional[int] = None,
                           frames_max: Optional[int] = None,
                           num_frames: Optional[int] = None,
                           num_keypoints: int = 137,
                           num_dimensions: int = 2,
                           fps: int = 50) -> dict:
    """

    :param frames_min:
    :param frames_max:
    :param num_frames:
    :param num_keypoints:
    :param num_dimensions:
    :param fps:
    :return:
    """
    if num_frames is None:
        assert None not in [frames_min, frames_max]
        num_frames = np.random.randint(frames_min, frames_max + 1)
    else:
        assert frames_min is None and frames_max is None

    # avoid random keypoints having a mean of zero
    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    tensor = tf.random.normal(shape=(num_frames, 1, num_keypoints, num_dimensions), mean=1.0)

    # (Frames, People, Points) - eg (93, 1, 137)
    confidence = tf.random.uniform((num_frames, 1, num_keypoints), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)

    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    mask = tf.random.uniform((num_frames, 1, num_keypoints, num_dimensions), minval=0, maxval=2, dtype=tf.dtypes.int32)
    mask = tf.cast(mask, tf.bool)

    # (Frames,) - eg (93,)
    tgt = tf.random.uniform((num_frames,), minval=0, maxval=2, dtype=tf.dtypes.int32)

    return {
        "fps": tf.constant(fps, dtype=tf.int32),
        "frames": tf.constant(num_frames, dtype=tf.int32),
        "tgt": tgt,
        "pose_data_tensor": tensor,
        "pose_data_mask": mask,
        "pose_confidence": confidence,
    }


def create_random_raw_dataset(num_examples: int,
                              frames_min: Optional[int] = None,
                              frames_max: Optional[int] = None,
                              num_frames_list: Optional[List[int]] = None,
                              num_keypoints: int = 137,
                              num_dimensions: int = 2,
                              fps: int = 50) -> tf.data.Dataset:
    """

    :param num_examples:
    :param frames_min:
    :param frames_max:
    :param num_frames_list:
    :param num_keypoints:
    :param num_dimensions:
    :param fps:
    :return:
    """
    if num_frames_list is None:
        num_frames_list = [None] * num_examples
    else:
        assert len(num_frames_list) == num_examples

    singleton_datasets = []  # type: List[tf.data.Dataset]

    for index in range(num_examples):
        num_frames = num_frames_list[index]

        example_dict = get_random_raw_example(frames_min=frames_min,
                                              frames_max=frames_max,
                                              num_frames=num_frames,
                                              num_keypoints=num_keypoints,
                                              num_dimensions=num_dimensions,
                                              fps=fps)

        restructured_dict = {}
        for key, value in example_dict.items():
            restructured_dict[key] = [value]

        singleton_dataset = tf.data.Dataset.from_tensor_slices(restructured_dict)
        singleton_datasets.append(singleton_dataset)

    dataset = tf.data.Dataset.from_tensor_slices(singleton_datasets)
    dataset = dataset.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


class TestDataLoader(TestCase):

    def test_data_loader_instance_type_is_correct(self):
        data_loader = DataLoader(data_dir="/tmp",
                                 batch_size=3,
                                 test_batch_size=2,
                                 normalize_pose=False,
                                 frame_dropout=False,
                                 frame_dropout_std=0.0,
                                 scale_pose=False,
                                 min_num_frames=0,
                                 max_num_frames=-1,
                                 max_num_frames_strategy="remove",
                                 num_keypoints=5)

        # dummy test
        self.assertIsInstance(data_loader, DataLoader, "Dataloader object does not have the correct type")

    def test_data_loader_get_datasets_train_example_shape_is_correct(self):
        with _create_tmp_working_directory(num_examples=10,
                                           fps=7,
                                           frames_min=1,
                                           frames_max=6,
                                           num_keypoints=5,
                                           num_dimensions=2) as working_dir:
            data_dir = os.path.join(working_dir, "data")

            data_loader = DataLoader(data_dir=data_dir,
                                     batch_size=3,
                                     test_batch_size=2,
                                     normalize_pose=False,
                                     frame_dropout=False,
                                     frame_dropout_std=0.0,
                                     scale_pose=False,
                                     min_num_frames=0,
                                     max_num_frames=-1,
                                     max_num_frames_strategy="remove",
                                     num_keypoints=5)

            train, _, _ = data_loader.get_datasets()

            expected_batch_size = 3
            expected_num_features = 10

            for index, example_tuple in enumerate(train.as_numpy_iterator()):
                if index == 2:
                    break
                example, label = example_tuple
                actual_batch_size = example.shape[0]
                actual_num_features = example.shape[2]
                self.assertEqual(actual_batch_size, expected_batch_size)
                self.assertEqual(actual_num_features, expected_num_features)

    def test_data_loader_get_datasets_devtest_example_shape_is_correct(self):
        with _create_tmp_working_directory(num_examples=10,
                                           fps=7,
                                           frames_min=1,
                                           frames_max=6,
                                           num_keypoints=5,
                                           num_dimensions=2) as working_dir:
            data_dir = os.path.join(working_dir, "data")

            data_loader = DataLoader(data_dir=data_dir,
                                     batch_size=3,
                                     test_batch_size=1,
                                     normalize_pose=False,
                                     frame_dropout=False,
                                     frame_dropout_std=0.0,
                                     scale_pose=False,
                                     min_num_frames=0,
                                     max_num_frames=-1,
                                     max_num_frames_strategy="remove",
                                     num_keypoints=5)

            _, dev, test = data_loader.get_datasets()

            expected_batch_size = 1
            expected_num_features = 10

            for dataset in [dev, test]:
                for index, example_tuple in enumerate(dataset.as_numpy_iterator()):
                    if index == 2:
                        break
                    example, label = example_tuple
                    actual_batch_size = example.shape[0]
                    actual_num_features = example.shape[2]
                    self.assertEqual(actual_batch_size, expected_batch_size)
                    self.assertEqual(actual_num_features, expected_num_features)

    def test_data_loader_apply_length_constraints_random_slice_examples_dataset_size_correct(self):
        data_loader = DataLoader(data_dir="/tmp",
                                 batch_size=3,
                                 test_batch_size=2,
                                 normalize_pose=False,
                                 frame_dropout=False,
                                 frame_dropout_std=0.0,
                                 scale_pose=False,
                                 min_num_frames=0,
                                 max_num_frames=5,
                                 max_num_frames_strategy="slice",
                                 num_keypoints=137)
        num_examples = 13
        num_frames_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        # create dummy tf.data.Dataset
        dataset = create_random_raw_dataset(num_examples=num_examples,
                                            num_frames_list=num_frames_list)

        # apply length constraints
        filtered_dataset = data_loader.maybe_apply_length_constraints(dataset, dataset_name="dummy")

        # check result size
        actual_size = int(filtered_dataset.reduce(0, lambda x, y: x + 1).numpy())

        self.assertEqual(actual_size, num_examples, "Size of dataset after slicing not as expected")

    def test_data_loader_apply_length_constraints_truncate_examples_dataset_size_correct(self):
        data_loader = DataLoader(data_dir="/tmp",
                                 batch_size=3,
                                 test_batch_size=2,
                                 normalize_pose=False,
                                 frame_dropout=False,
                                 frame_dropout_std=0.0,
                                 scale_pose=False,
                                 min_num_frames=0,
                                 max_num_frames=5,
                                 max_num_frames_strategy="truncate",
                                 num_keypoints=5)
        num_examples = 13
        num_frames_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        # create dummy tf.data.Dataset
        dataset = create_random_raw_dataset(num_examples=num_examples,
                                            num_frames_list=num_frames_list,
                                            num_keypoints=5)

        # apply length constraints
        filtered_dataset = data_loader.maybe_apply_length_constraints(dataset, dataset_name="dummy")

        # check result size
        actual_size = int(filtered_dataset.reduce(0, lambda x, y: x + 1).numpy())

        self.assertEqual(actual_size, num_examples, "Size of dataset after truncating not as expected")


class TestSliceExample(TestCase):

    def test_slice_example_output_num_frames_is_correct(self):

        num_keypoints = 4
        num_dimensions = 2
        fps = 5

        chunk_size = 5

        example = get_random_raw_example(num_frames=17,
                                         num_keypoints=num_keypoints,
                                         num_dimensions=num_dimensions,
                                         fps=fps)

        sliced_example = random_slice_example(example, max_num_frames=chunk_size)

        expected_num_frames = 5

        actual_num_frames = int(sliced_example["frames"].numpy())

        self.assertEqual(actual_num_frames, expected_num_frames, "Size of example after slicing not as expected")

    def test_slice_example_num_frames_lower_than_chunk_size(self):

        num_keypoints = 4
        num_dimensions = 2
        fps = 5

        chunk_size = 5

        example = get_random_raw_example(num_frames=4,
                                         num_keypoints=num_keypoints,
                                         num_dimensions=num_dimensions,
                                         fps=fps)

        sliced_example = random_slice_example(example, max_num_frames=chunk_size)

        expected_num_frames = 4

        actual_num_frames = int(sliced_example["frames"].numpy())

        self.assertEqual(actual_num_frames, expected_num_frames, "Size of example after slicing not as expected")

    def test_slice_example_output_keys_are_correct(self):

        num_keypoints = 4
        num_dimensions = 2
        fps = 5

        chunk_size = 5

        example = get_random_raw_example(num_frames=17,
                                         num_keypoints=num_keypoints,
                                         num_dimensions=num_dimensions,
                                         fps=fps)

        sliced_example = random_slice_example(example, max_num_frames=chunk_size)

        expected_keys = {"fps", "frames", "tgt", "pose_data_tensor", "pose_data_mask", "pose_confidence"}

        actual_keys = set(sliced_example.keys())

        self.assertSetEqual(actual_keys, expected_keys)

    def test_chunk_example_chunk_shapes_are_correct(self):

        num_keypoints = 4
        num_dimensions = 2
        fps = 5

        chunk_size = 5

        example = get_random_raw_example(num_frames=17,
                                         num_keypoints=num_keypoints,
                                         num_dimensions=num_dimensions,
                                         fps=fps)

        sliced_example = random_slice_example(example, max_num_frames=chunk_size)

        expected_shapes = {"tgt": (chunk_size,),
                           "pose_data_tensor": (chunk_size, 1, num_keypoints, num_dimensions),
                           "pose_data_mask": (chunk_size, 1, num_keypoints, num_dimensions),
                           "pose_confidence": (chunk_size, 1, num_keypoints),
                           }

        relevant_keys = expected_shapes.keys()

        for key in relevant_keys:
            expected_shape = expected_shapes[key]
            actual_shape = sliced_example[key].shape
            self.assertEqual(actual_shape, expected_shape)


class TestTruncateExample(TestCase):

    def test_truncate_example_output_num_frames_is_correct(self):

        num_keypoints = 4
        num_dimensions = 2
        fps = 5

        chunk_size = 5

        example = get_random_raw_example(num_frames=17,
                                         num_keypoints=num_keypoints,
                                         num_dimensions=num_dimensions,
                                         fps=fps)

        truncated_example = truncate_example(example, max_num_frames=chunk_size)

        expected_num_frames = 5

        actual_num_frames = int(truncated_example["frames"].numpy())

        self.assertEqual(actual_num_frames, expected_num_frames, "Size of example after truncating not as expected")

    def test_truncate_example_num_frames_lower_than_chunk_size(self):

        num_keypoints = 4
        num_dimensions = 2
        fps = 5

        chunk_size = 5

        example = get_random_raw_example(num_frames=4,
                                         num_keypoints=num_keypoints,
                                         num_dimensions=num_dimensions,
                                         fps=fps)

        truncated_example = truncate_example(example, max_num_frames=chunk_size)

        expected_num_frames = 4

        actual_num_frames = int(truncated_example["frames"].numpy())

        self.assertEqual(actual_num_frames, expected_num_frames, "Size of example after truncating not as expected")

    def test_truncate_example_output_keys_are_correct(self):

        num_keypoints = 4
        num_dimensions = 2
        fps = 5

        chunk_size = 5

        example = get_random_raw_example(num_frames=17,
                                         num_keypoints=num_keypoints,
                                         num_dimensions=num_dimensions,
                                         fps=fps)

        truncated_example = truncate_example(example, max_num_frames=chunk_size)

        expected_keys = {"fps", "frames", "tgt", "pose_data_tensor", "pose_data_mask", "pose_confidence"}

        actual_keys = set(truncated_example.keys())

        self.assertSetEqual(actual_keys, expected_keys)

    def test_truncate_example_chunk_shapes_are_correct(self):

        num_keypoints = 4
        num_dimensions = 2
        fps = 5

        chunk_size = 5

        example = get_random_raw_example(num_frames=17,
                                         num_keypoints=num_keypoints,
                                         num_dimensions=num_dimensions,
                                         fps=fps)

        truncated_example = truncate_example(example, max_num_frames=chunk_size)

        expected_shapes = {"tgt": (chunk_size,),
                           "pose_data_tensor": (chunk_size, 1, num_keypoints, num_dimensions),
                           "pose_data_mask": (chunk_size, 1, num_keypoints, num_dimensions),
                           "pose_confidence": (chunk_size, 1, num_keypoints),
                           }

        relevant_keys = expected_shapes.keys()

        for key in relevant_keys:
            expected_shape = expected_shapes[key]
            actual_shape = truncated_example[key].shape
            self.assertEqual(actual_shape, expected_shape)
