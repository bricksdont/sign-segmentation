#! /usr/bin/python3

import os
import argparse

from unittest import TestCase

from test.common import _create_tmp_working_directory

from sign_language_segmentation.train import train


def create_mock_namespace(data_dir: str,
                          model_path: str,
                          scale_pose: bool = False,
                          normalize_pose: bool = False,
                          frame_dropout: bool = False,
                          num_keypoints: int = 137) -> argparse.Namespace:
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
                      num_keypoints=num_keypoints,
                      desired_fps=50,
                      max_num_frames=-1,
                      min_num_frames=1,
                      max_num_frames_strategy="slice",
                      pose_type="openpose",
                      normalize_pose=normalize_pose,
                      scale_pose=scale_pose,
                      frame_dropout=frame_dropout,
                      model_path=model_path,
                      offline=False,
                      log_model_off=False,
                      wandb_run_name=False)

    return argparse.Namespace(**input_dict)


class TestTraining(TestCase):

    def test_train_function(self):

        with _create_tmp_working_directory(num_examples=10,
                                           fps=7,
                                           frames_min=1,
                                           frames_max=6,
                                           num_keypoints=5,
                                           num_dimensions=2) as working_dir:

            data_dir = os.path.join(working_dir, "data")
            model_path = os.path.join(working_dir, "models", "test_model")

            args = create_mock_namespace(data_dir, model_path, num_keypoints=5)

            train(args)

    def test_train_function_scale_pose(self):

        with _create_tmp_working_directory(num_examples=10,
                                           fps=7,
                                           frames_min=1,
                                           frames_max=6,
                                           num_keypoints=5,
                                           num_dimensions=2) as working_dir:

            data_dir = os.path.join(working_dir, "data")
            model_path = os.path.join(working_dir, "models", "test_model")

            args = create_mock_namespace(data_dir, model_path, num_keypoints=5, scale_pose=True)

            train(args)

    def test_train_function_normalize_pose(self):

        # this test needs to run with a full 137 keypoints per frame at the moment, because it
        # expects OpenPose layout

        with _create_tmp_working_directory(num_examples=10,
                                           fps=7,
                                           frames_min=1,
                                           frames_max=6,
                                           num_keypoints=137,
                                           num_dimensions=2) as working_dir:

            data_dir = os.path.join(working_dir, "data")
            model_path = os.path.join(working_dir, "models", "test_model")

            args = create_mock_namespace(data_dir, model_path, num_keypoints=137, normalize_pose=True)

            train(args)

    def test_train_function_frame_dropout(self):

        with _create_tmp_working_directory(num_examples=10,
                                           fps=7,
                                           frames_min=1,
                                           frames_max=6,
                                           num_keypoints=5,
                                           num_dimensions=2) as working_dir:

            data_dir = os.path.join(working_dir, "data")
            model_path = os.path.join(working_dir, "models", "test_model")

            args = create_mock_namespace(data_dir, model_path, num_keypoints=5, frame_dropout=True)

            train(args)
