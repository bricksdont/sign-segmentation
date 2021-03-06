#! /usr/bin/python3

"""Training script for sign language segmentation."""

import random
import argparse
import logging

import tensorflow as tf

from .arguments import parse_args
from .data import DataLoader, log_dataset_statistics
from .model import ModelBuilder

from typing import Optional, Tuple

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


def set_seed(seed: Optional[int] = None):
    """
    Set seed for deterministic random number generation.

    :param seed:
    :return:
    """
    if seed is None:
        seed = random.randint(0, 1000)

    tf.random.set_seed(seed)
    random.seed(seed)


def train(args: argparse.Namespace) -> Tuple[tf.keras.callbacks.History, dict]:
    """
    Keras training loop with early-stopping and model checkpoint.

    :param args:
    :return:
    """

    set_seed(args.seed)

    # Initialize Dataset
    logging.debug("###############")
    logging.debug("Loading dataset")
    logging.debug("###############")

    data_loader = DataLoader(data_dir=args.data_dir,
                             batch_size=args.batch_size,
                             test_batch_size=args.test_batch_size,
                             normalize_pose=args.normalize_pose,
                             frame_dropout=args.frame_dropout,
                             frame_dropout_type=args.frame_dropout_type,
                             frame_dropout_std=args.frame_dropout_std,
                             frame_dropout_mean=args.frame_dropout_mean,
                             frame_dropout_min=args.frame_dropout_min,
                             frame_dropout_max=args.frame_dropout_max,
                             scale_pose=args.scale_pose,
                             min_num_frames=args.min_num_frames,
                             max_num_frames=args.max_num_frames,
                             max_num_frames_strategy=args.max_num_frames_strategy,
                             num_keypoints=args.num_keypoints)

    train, dev, test = data_loader.get_datasets()

    log_dataset_statistics(dataset=train, name="train", infinite=True)
    log_dataset_statistics(dataset=dev, name="dev")
    log_dataset_statistics(dataset=test, name="test")

    # Initialize Model
    logging.debug("##############")
    logging.debug("Building model")
    logging.debug("##############")

    model_builder = ModelBuilder(input_dropout=args.input_dropout,
                                 encoder_bidirectional=args.encoder_bidirectional,
                                 hidden_size=args.hidden_size,
                                 num_keypoints=args.num_keypoints,
                                 learning_rate=args.learning_rate,
                                 num_encoder_layers=args.num_encoder_layers)
    model = model_builder.build_model()

    # Train
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=args.patience,
                       min_delta=args.min_delta)
    mc = ModelCheckpoint(args.model_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    logging.debug("##############")
    logging.debug('Start training')
    logging.debug("##############")

    with tf.device(args.device):
        history = model.fit(train,
                            epochs=args.epochs,
                            steps_per_epoch=args.steps_per_epoch,
                            validation_data=dev,
                            callbacks=[es, mc])

    best_model = load_model(args.model_path)

    logging.debug("#############")
    logging.debug('Start testing')
    logging.debug("#############")

    evaluation_results = best_model.evaluate(test, return_dict=True)

    return history, evaluation_results


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    train(args)


if __name__ == '__main__':
    main()
