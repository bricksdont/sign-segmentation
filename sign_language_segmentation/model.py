"""Sign language sequence tagging keras model."""

import tensorflow as tf

from sign_language_segmentation.args import FLAGS


def get_model():
    """Create keras sequential model following the hyperparameters."""

    model = tf.keras.Sequential(name='tgt')

    # Random feature dropout
    model.add(tf.keras.layers.Dropout(FLAGS.input_dropout))

    # Add LSTM
    for _ in range(FLAGS.encoder_layers):
        rnn = tf.keras.layers.LSTM(FLAGS.hidden_size, return_sequences=True)
        if FLAGS.encoder_bidirectional:
            rnn = tf.keras.layers.Bidirectional(rnn)
        model.add(rnn)

    # Project and normalize to labels space
    model.add(tf.keras.layers.Dense(3))

    return model


def build_model():
    """Apply input shape, loss, optimizer, and metric to the model."""
    model = get_model()
    model.build(input_shape=(None, None, FLAGS.input_size))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        metrics=['accuracy'],
    )
    model.summary()

    return model
