"""Utilities to load and process a sign language segmentation dataset."""

import argparse
import tensorflow as tf

from pose_format.pose import Pose
from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody
from pose_format.tensorflow.pose_body import TF_POSE_RECORD_DESCRIPTION


def distance(src):
    """Calculate the Euclidean distance from x:y coordinates."""
    square = src.square()
    sum_squares = square.sum(dim=-1).fix_nan()
    sqrt = sum_squares.sqrt().zero_filled()
    return sqrt


minimum_fps = tf.constant(1, dtype=tf.float32)


def load_datum(tfrecord_dict):
    """Convert tfrecord dictionary to tensors."""
    pose_body = TensorflowPoseBody.from_tfrecord(tfrecord_dict)
    tgt = tf.io.decode_raw(tfrecord_dict["tags"], out_type=tf.int8)

    fps = pose_body.fps
    frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

    return {
        "fps": fps,
        "frames": frames,
        "tgt": tgt,
        "pose_data_tensor": pose_body.data.tensor,
        "pose_data_mask": pose_body.data.mask,
        "pose_confidence": pose_body.confidence,
    }


# TODO normalize shoulders in every frame
def process_datum(datum, args: argparse.Namespace, augment=False):
    """Prepare every datum to be an input-output pair for training / eval.
    Supports data augmentation only including frames dropout.
    Frame dropout affects the FPS."""
    masked_tensor = MaskedTensor(tensor=datum["pose_data_tensor"], mask=datum["pose_data_mask"])
    pose_body = TensorflowPoseBody(fps=datum["fps"], data=masked_tensor, confidence=datum["pose_confidence"])
    pose = Pose(header=get_openpose_header(), body=pose_body)
    tgt = datum["tgt"]

    fps = pose.body.fps
    frames = datum["frames"]

    if augment:
        pose, selected_indexes = pose.frame_dropout(args.frame_dropout_std)
        tgt = tf.gather(tgt, selected_indexes)

        new_frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

        fps = tf.math.maximum(minimum_fps, (new_frames / frames) * fps)
        frames = new_frames

    return {"src": pose.data.tensor, "tgt": tgt}


def prepare_io(datum):
    """Convert dictionary into input-output tuple for Keras."""
    src = datum["src"]
    tgt = datum["tgt"]

    return src, tgt


def batch_dataset(dataset, batch_size):
    """Batch and pad a dataset."""
    dataset = dataset.padded_batch(
        batch_size, padded_shapes={
            "src": [None, None],
            "tgt": [None]
        })

    return dataset.map(prepare_io)


def train_pipeline(dataset, args: argparse.Namespace):
    """Prepare the training dataset."""
    dataset = dataset.map(load_datum).cache()
    dataset = dataset.repeat()
    dataset = dataset.map(lambda d: process_datum(d, args, True))
    dataset = dataset.shuffle(args.batch_size)
    dataset = batch_dataset(dataset, args.batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def test_pipeline(dataset, args: argparse.Namespace):
    """Prepare the test dataset."""
    dataset = dataset.map(load_datum)
    dataset = dataset.map(process_datum, args)
    dataset = batch_dataset(dataset, args.test_batch_size)
    return dataset.cache()


def split_dataset(dataset, args: argparse.Namespace):
    """Split dataset to train, dev, and test."""

    def is_dev(x, _):
        # Every 7th item
        return x % 8 == 6

    def is_test(x, _):
        # Every 8th item
        return x % 8 == 7

    def is_train(x, y):
        return not is_test(x, y) and not is_dev(x, y)

    def recover(_, y):
        return y

    train = train_pipeline(dataset.enumerate().filter(is_train).map(recover), args)
    dev = test_pipeline(dataset.enumerate().filter(is_dev).map(recover), args)
    test = test_pipeline(dataset.enumerate().filter(is_test).map(recover), args)

    return train, dev, test


def get_datasets(args: argparse.Namespace):
    """Get train, dev, and test datasets."""
    # Set features
    features = {"tags": tf.io.FixedLenFeature([], tf.string)}
    features.update(TF_POSE_RECORD_DESCRIPTION)

    # Dataset iterator
    dataset = tf.data.TFRecordDataset(filenames=[args.dataset_path])
    dataset = dataset.map(
        lambda serialized: tf.io.parse_single_example(serialized, features))

    return split_dataset(dataset, args)
