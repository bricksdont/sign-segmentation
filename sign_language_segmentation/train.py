#! /usr/bin/python3

"""Training script for sign language segmentation."""

import random
import argparse

import tensorflow as tf

from sign_language_segmentation.args import parse_args
from sign_language_segmentation.data import get_datasets
from sign_language_segmentation.model import build_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


def set_seed(args: argparse.Namespace):
    """Set seed for deterministic random number generation."""
    seed = args.seed if args.seed is not None else random.randint(0, 1000)
    tf.random.set_seed(seed)
    random.seed(seed)


def main(args: argparse.Namespace):
    """Keras training loop with early-stopping and model checkpoint."""

    set_seed(args)

    # Initialize Dataset
    train, dev, test = get_datasets(args)

    # Initialize Model
    model = build_model(args)

    # Train
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=args.stop_patience)
    mc = ModelCheckpoint(args.model_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    with tf.device(args.device):
        model.fit(train,
                  epochs=args.epochs,
                  steps_per_epoch=args.steps_per_epoch,
                  validation_data=dev,
                  callbacks=[es, mc])

    best_model = load_model(args.model_path)
    print('Testing')
    best_model.evaluate(test)


if __name__ == '__main__':
    args = parse_args()
    main(args)
