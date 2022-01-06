#! /usr/bin/python3

"""Training script for sign language segmentation."""

import random
import argparse
import logging

import tensorflow as tf

from arguments import parse_args
from data import DataLoader, log_dataset_statistics
from model import ModelBuilder

from typing import Optional

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


def main(args: argparse.Namespace):
    """
    Keras training loop with early-stopping and model checkpoint.

    :param args:
    :return:
    """

    set_seed(args.seed)

    # Initialize Dataset
    data_loader = DataLoader(data_dir=args.data_dir,
                             batch_size=args.batch_size,
                             test_batch_size=args.test_batch_size,
                             normalize_pose=args.normalize_pose,
                             frame_dropout=args.frame_dropout,
                             frame_dropout_std=args.frame_dropout_std)

    train, dev, test = data_loader.get_datasets()

    log_dataset_statistics(train, "train")
    log_dataset_statistics(dev, "dev")
    log_dataset_statistics(test, "test")

    # Initialize Model
    model_builder = ModelBuilder(input_dropout=args.input_dropout,
                                 encoder_bidirectional=args.encoder_bidirectional,
                                 hidden_size=args.hidden_size,
                                 input_size=args.input_size,
                                 learning_rate=args.learning_rate,
                                 num_encoder_layers=args.num_encoder_layers)
    model = model_builder.build_model()

    # Train
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=args.patience,
                       min_delta=args.min_delta)
    mc = ModelCheckpoint(args.model_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    with tf.device(args.device):
        model.fit(train,
                  epochs=args.epochs,
                  steps_per_epoch=args.steps_per_epoch,
                  validation_data=dev,
                  callbacks=[es, mc])

    best_model = load_model(args.model_path)
    logging.debug('Testing')
    best_model.evaluate(test)


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    main(args)
