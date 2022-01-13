#! /usr/bin/python3

import os
import argparse
import logging

import tensorflow as tf
import numpy as np

from unittest import TestCase
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

from sign_language_segmentation.train import main


def get_random_numpy_example(frames_min: Optional[int] = None,
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

    # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
    pose_data = np.random.random_sample((num_frames, 1, num_keypoints, num_dimensions)).astype(np.float32)

    # (Frames, People, Points) - eg (93, 1, 137)
    pose_confidence = np.random.random_sample((num_frames, 1, num_keypoints)).astype(np.float32)

    # (Frames,) - eg (93,)
    bio = np.random.randint(0, 3, size=(num_frames,)).astype(np.int8)

    print(pose_data.shape, pose_confidence.shape, bio.shape)

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
            pose_data, pose_confidence, bio = get_random_numpy_example(frames_min=frames_min,
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
def _create_tmp_working_directory() -> str:
    with TemporaryDirectory(prefix="test_train") as working_dir:

        # create "data" subfolder with tfrecord file
        data_dir = os.path.join(working_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        tfrecord_filepath = os.path.join(data_dir, "data.tfrecord")

        _create_tmp_tfrecord_file(filepath=tfrecord_filepath,
                                  num_examples=10,
                                  fps=7,
                                  frames_min=1,
                                  frames_max=6,
                                  num_keypoints=137,
                                  num_dimensions=2)

        # create "models" subfolder
        models_dir = os.path.join(working_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        yield working_dir


def create_mock_namespace(data_dir: str, model_path: str) -> argparse.Namespace:
    """

    :return:
    """
    input_dict = dict(seed=0,
                      device="/CPU",
                      batch_size=2,
                      test_batch_size=1,
                      learning_rate=0.0001,
                      epochs=2,
                      steps_per_epoch=2,
                      patience=1,
                      min_delta=0.0,
                      input_dropout=0.3,
                      num_encoder_layers=1,
                      encoder_bidirectional=True,
                      hidden_size=8,
                      data_dir=data_dir,
                      frame_dropout_std=0.3,
                      input_size=137,
                      desired_fps=50,
                      max_num_frames=-1,
                      min_num_frames=1,
                      max_num_frames_strategy="slice",
                      pose_type="openpose",
                      normalize_pose=False,
                      scale_pose=False,
                      frame_dropout=False,
                      model_path=model_path,
                      offline=False,
                      log_model_off=False,
                      wandb_run_name=False)

    return argparse.Namespace(**input_dict)


class TestTraining(TestCase):

    def test_train_main_function(self):

        with _create_tmp_working_directory() as working_dir:

            data_dir = os.path.join(working_dir, "data")
            model_path = os.path.join(working_dir, "models", "test_model")

            args = create_mock_namespace(data_dir, model_path)

            main(args)
