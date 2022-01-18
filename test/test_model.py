#! /usr/bin/python3

import numpy as np

from unittest import TestCase
from typing import Tuple

from sign_language_segmentation.model import ModelBuilder


def create_mock_training_data(batch_size: int,
                              num_frames: int,
                              num_features: int) -> Tuple[np.array, np.array]:
    """

    :param batch_size:
    :param num_frames:
    :param num_features:
    :return:
    """
    examples = np.random.random((batch_size, num_frames, num_features)).astype(np.float32)
    labels = np.random.randint(0, 3, size=(batch_size, num_frames,)).astype(np.int8)

    return examples, labels


class TestModel(TestCase):

    def test_model_fit(self):

        model_builder = ModelBuilder(input_dropout=0.0,
                                     encoder_bidirectional=True,
                                     hidden_size=16,
                                     num_keypoints=6,
                                     learning_rate=0.001,
                                     num_encoder_layers=1)
        model = model_builder.build_model()

        examples, labels = create_mock_training_data(batch_size=5, num_frames=7, num_features=12)

        model.fit(examples, labels, epochs=1)
