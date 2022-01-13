#! /usr/bin/python3

import tempfile

import numpy as np
import tensorflow as tf

from contextlib import contextmanager


@contextmanager
def _create_tmp_tfrecord_file(num_examples: int) -> str:
    """
    https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset

    :param num_examples:
    :return:
    """
    with tempfile.NamedTemporaryFile(mode="w+") as filehandle:
        with tf.io.TFRecordWriter(filehandle.name) as file_writer:
            for _ in range(num_examples):
                x = np.random.random()

                record_bytes = tf.train.Example(features=tf.train.Features(feature={
                    "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x]))
                })).SerializeToString()
                file_writer.write(record_bytes)

        yield filehandle.name


def read_dummy_tfrecord_dataset(filehandle: str) -> tf.data.TFRecordDataset:
    """

    :param filehandle:
    :return:
    """

    def tfrecord_decode_fn(record_bytes):
        """
        https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset

        :param record_bytes:
        :return:
        """
        return tf.io.parse_single_example(record_bytes, {"x": tf.io.FixedLenFeature([], dtype=tf.float32)})

    return tf.data.TFRecordDataset([filehandle]).map(tfrecord_decode_fn)
