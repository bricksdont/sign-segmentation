"""Sign language sequence tagging keras model."""

import tensorflow as tf
import argparse


def get_model(args: argparse.Namespace):
    """Create keras sequential model following the hyperparameters."""

    model = tf.keras.Sequential(name='tgt')

    # Random feature dropout
    model.add(tf.keras.layers.Dropout(args.input_dropout))

    # Add LSTM
    for _ in range(args.encoder_layers):
        rnn = tf.keras.layers.LSTM(args.hidden_size, return_sequences=True)
        if args.encoder_bidirectional:
            rnn = tf.keras.layers.Bidirectional(rnn)
        model.add(rnn)

    # Project and normalize to labels space
    model.add(tf.keras.layers.Dense(3))

    return model


def build_model(args: argparse.Namespace):
    """Apply input shape, loss, optimizer, and metric to the model."""
    model = get_model(args)
    model.build(input_shape=(None, None, args.input_size))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        metrics=['accuracy'],
    )
    model.summary()

    return model
