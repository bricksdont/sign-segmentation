#! /usr/bin/python3

"""Sign language sequence tagging keras model."""

import tensorflow as tf


class ModelBuilder:

    def __init__(self, input_dropout: float, num_encoder_layers: int, hidden_size: int,
                 encoder_bidirectional: bool, num_keypoints: int, learning_rate: float) -> None:
        """

        :param input_dropout:
        :param num_encoder_layers:
        :param hidden_size:
        :param encoder_bidirectional:
        :param num_keypoints:
        :param learning_rate:
        """
        self.input_dropout = input_dropout
        self.num_encoder_layers = num_encoder_layers
        self.hidden_size = hidden_size
        self.encoder_bidirectional = encoder_bidirectional
        self.num_keypoints = num_keypoints
        self.learning_rate = learning_rate

    def get_model(self) -> tf.keras.Sequential:
        """
        Create keras sequential model following the hyperparameters.

        :return:
        """

        model = tf.keras.Sequential(name='label_frames_as_bio')

        # Random feature dropout
        model.add(tf.keras.layers.Dropout(self.input_dropout))

        # Add LSTM
        for _ in range(self.num_encoder_layers):
            rnn = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True)
            if self.encoder_bidirectional:
                rnn = tf.keras.layers.Bidirectional(rnn)
            model.add(rnn)

        # Project and normalize to labels space
        model.add(tf.keras.layers.Dense(3))

        return model

    def build_model(self):
        """
        Apply input shape, loss, optimizer, and metric to the model.

        :return:
        """
        model = self.get_model()
        model.build(input_shape=(None, None, self.num_keypoints * 2))
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            metrics=['accuracy'],
        )
        model.summary()

        return model
