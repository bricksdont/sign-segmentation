#! /usr/bin/python3

import os

import tensorflow as tf
import numpy as np

from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Optional, Tuple


def _get_random_numpy_example(frames_min: Optional[int] = None,
                              frames_max: Optional[int] = None,
                              num_frames: Optional[int] = None,
                              num_keypoints: int = 137,
                              num_dimensions: int = 2) -> Tuple[np.array, np.array, np.array]:
    """

    :param frames_min:
    :param frames_max:
    :param num_frames:
    :param num_keypoints:
    :param num_dimensions:
    :return:
    """
    if num_frames is None:
        assert None not in [frames_min, frames_max]
        num_frames = np.random.randint(frames_min, frames_max + 1)
    else:
        assert frames_min is None and frames_max is None

    # avoid keypoints with a mean of zero
    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    pose_data = np.random.normal(size=(num_frames, 1, num_keypoints, num_dimensions), loc=1.0).astype(np.float32)

    # (Frames, People, Points) - eg (93, 1, 137)
    pose_confidence = np.random.random_sample((num_frames, 1, num_keypoints)).astype(np.float32)

    # (Frames,) - eg (93,)
    bio = np.random.randint(0, 3, size=(num_frames,)).astype(np.int8)

    return pose_data, pose_confidence, bio


def _write_single_example(writer: tf.io.TFRecordWriter,
                          fps: int,
                          pose_data: np.array,
                          pose_confidence: np.array,
                          bio: np.array):
    """

    :param writer:
    :param fps:
    :param pose_data:
    :param pose_confidence:
    :param bio:
    :return:
    """
    pose_data = tf.io.serialize_tensor(pose_data).numpy()
    pose_confidence = tf.io.serialize_tensor(pose_confidence).numpy()

    tags = bio.tobytes()
    fps = np.array(fps, dtype=np.int32)

    features = {
        'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[fps])),
        'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_data])),
        'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_confidence])),
        'tags': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tags]))
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())


def _create_tmp_tfrecord_file(filepath: str,
                              num_examples: int,
                              fps: int,
                              frames_min: Optional[int] = None,
                              frames_max: Optional[int] = None,
                              num_frames: Optional[int] = None,
                              num_keypoints: int = 137,
                              num_dimensions: int = 2):
    """

    :param filepath:
    :param num_examples:
    :param fps:
    :param frames_min:
    :param frames_max:
    :param num_frames:
    :param num_keypoints:
    :param num_dimensions:
    :return:
    """
    with tf.io.TFRecordWriter(filepath) as writer:
        for _ in range(num_examples):
            pose_data, pose_confidence, bio = _get_random_numpy_example(frames_min=frames_min,
                                                                        frames_max=frames_max,
                                                                        num_frames=num_frames,
                                                                        num_keypoints=num_keypoints,
                                                                        num_dimensions=num_dimensions)
            _write_single_example(writer=writer,
                                  fps=fps,
                                  pose_data=pose_data,
                                  pose_confidence=pose_confidence,
                                  bio=bio)


@contextmanager
def _create_tmp_working_directory(num_examples: int,
                                  fps: int,
                                  frames_min: Optional[int] = None,
                                  frames_max: Optional[int] = None,
                                  num_frames: Optional[int] = None,
                                  num_keypoints: int = 137,
                                  num_dimensions: int = 2) -> str:
    """

    :param num_examples:
    :param fps:
    :param frames_min:
    :param frames_max:
    :param num_frames:
    :param num_keypoints:
    :param num_dimensions:
    :return:
    """
    with TemporaryDirectory(prefix="test_train") as working_dir:

        # create "data" subfolder with tfrecord file
        data_dir = os.path.join(working_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        tfrecord_filepath = os.path.join(data_dir, "data.tfrecord")

        _create_tmp_tfrecord_file(filepath=tfrecord_filepath,
                                  num_examples=num_examples,
                                  fps=fps,
                                  frames_min=frames_min,
                                  frames_max=frames_max,
                                  num_frames=num_frames,
                                  num_keypoints=num_keypoints,
                                  num_dimensions=num_dimensions)

        # create "models" subfolder
        models_dir = os.path.join(working_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        yield working_dir
