#! /usr/bin/python3

"""Utilities to load and process a sign language segmentation dataset."""

import functools
import os
import logging
import tensorflow as tf

from typing import Optional, Tuple

from pose_format.pose import Pose
from pose_format.pose_header import PoseHeader
from pose_format.tensorflow.masked.tensor import MaskedTensor
from pose_format.tensorflow.pose_body import TensorflowPoseBody
from pose_format.tensorflow.pose_body import TF_POSE_RECORD_DESCRIPTION
from pose_format.utils.reader import BufferReader


TFRECORD_FILE_NAME = "data.tfrecord"


@functools.lru_cache(maxsize=1)
def get_openpose_header(header_path: Optional[str] = None):
    """
    Get pose header with OpenPose components description.

    :param header_path:
    :return:
    """
    if header_path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        header_path = os.path.join(dir_path, "../assets/openpose.poseheader")

    f = open(header_path, "rb")
    reader = BufferReader(f.read())
    header = PoseHeader.read(reader)
    return header


def differentiate_frames(src: tf.Tensor) -> tf.Tensor:
    """
    Subtract every two consecutive frames.

    :param src:
    :return:
    """
    # Shift data to pre/post frames
    pre_src = src[:-1]
    post_src = src[1:]

    # Differentiate src points
    src = pre_src - post_src

    return src


def distance(src: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Euclidean distance from x:y coordinates.

    :param src:
    :return:
    """
    square = src.square()
    sum_squares = square.sum(dim=-1).fix_nan()
    sqrt = sum_squares.sqrt().zero_filled()
    return sqrt


def optical_flow(src: tf.Tensor, fps: tf.Tensor) -> tf.Tensor:
    """
    Calculate the optical flow norm between frames, normalized by fps.

    :param src:
    :param fps:
    :return:
    """

    # Remove "people" dimension
    src = src.squeeze(1)

    # Differentiate Frames
    src = differentiate_frames(src)

    # Calculate distance
    src = distance(src)

    # Normalize distance by fps
    src = src * fps

    return src


def get_dataset_size(dataset: tf.data.Dataset) -> int:
    """

    :param dataset:
    :return:
    """
    return int(dataset.reduce(0, lambda x, y: x+1).numpy())


def log_raw_datum_examples(dataset: tf.data.Dataset,
                           max_index: int = 2):
    """

    :param dataset:
    :param max_index:
    :return:
    """
    template_string = "\tRaw datum %d: fps=%s, frames=%s, tgt.shape=%s, pose_data_tensor.shape=%s, " \
                      "pose_data_mask.shape=%s, pose_confidence.shape=%s"

    for index, datum in enumerate(dataset.as_numpy_iterator()):
        if index == max_index:
            break
        logging.debug(template_string,
                      index,
                      datum["fps"],
                      datum["frames"],
                      datum["tgt"].shape,
                      datum["pose_data_tensor"].shape,
                      datum["pose_data_mask"].shape,
                      datum["pose_confidence"].shape)


def log_datum_examples(dataset: tf.data.Dataset):
    """

    :param dataset:
    :return:
    """
    for index, datum in enumerate(dataset.as_numpy_iterator()):
        if index == 2:
            break
        example, label = datum
        logging.debug("\tBatch %d: example.shape=%s, label.shape=%s", index, example.shape, label.shape)


def log_dataset_statistics(dataset: tf.data.Dataset,
                           name: str = "data",
                           infinite: bool = False) -> None:
    """
    Log size of dataset and the shapes of the first two examples and labels.

    :param dataset:
    :param name:
    :param infinite:
    :return:
    """
    logging.debug("Statistics of dataset: '%s'", name)

    if not infinite:
        num_batches = get_dataset_size(dataset)

        logging.debug("\tnum_batches: %d", num_batches)
    else:
        logging.debug("\tWill not compute number of batches in dataset since it is infinite.")

    log_datum_examples(dataset)


class DataLoader:

    def __init__(self, data_dir: str, batch_size: int, test_batch_size: int, normalize_pose: bool,
                 frame_dropout: bool, frame_dropout_std: float, scale_pose: bool, min_num_frames: int,
                 max_num_frames: int, max_num_frames_strategy: str):
        """

        :param data_dir:
        :param batch_size:
        :param test_batch_size:
        :param normalize_pose:
        :param frame_dropout:
        :param frame_dropout_std:
        :param scale_pose:
        :param min_num_frames:
        :param max_num_frames:
        :param max_num_frames_strategy:
        """

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.normalize_pose = normalize_pose
        self.frame_dropout = frame_dropout
        self.frame_dropout_std = frame_dropout_std
        self.scale_pose = scale_pose
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.max_num_frames_strategy = max_num_frames_strategy

        self.minimum_fps = tf.constant(1, dtype=tf.float32)

        if self.max_num_frames == -1:
            # set to ridiculously high value that would not run on current hardware
            self.max_num_frames = 10000000

    def load_datum(self, tfrecord_dict: dict) -> dict:
        """
        Convert tfrecord dictionary to tensors.

        :param tfrecord_dict:
        :return:
        """
        pose_body = TensorflowPoseBody.from_tfrecord(tfrecord_dict)
        tgt = tf.io.decode_raw(tfrecord_dict["tags"], out_type=tf.int8)

        fps = pose_body.fps
        frames = tf.cast(tf.size(tgt), dtype=fps.dtype)

        # TODO: Get only relevant input components
        # pose = pose.get_components(FLAGS.input_components)

        return {
            "fps": fps,
            "frames": frames,
            "tgt": tgt,
            "pose_data_tensor": pose_body.data.tensor,
            "pose_data_mask": pose_body.data.mask,
            "pose_confidence": pose_body.confidence,
        }

    def process_datum(self, datum: dict, is_train: bool) -> dict:
        """
        Prepare every datum to be an input-output pair for training / eval.
        Supports frame dropout and different kinds of normalization.
        Frame dropout affects the FPS.

        :param datum:
        :param is_train:
        :return:
        """
        masked_tensor = MaskedTensor(tensor=datum["pose_data_tensor"], mask=datum["pose_data_mask"])
        pose_body = TensorflowPoseBody(fps=datum["fps"], data=masked_tensor, confidence=datum["pose_confidence"])
        pose = Pose(header=get_openpose_header(), body=pose_body)
        tgt = datum["tgt"]

        if self.frame_dropout and is_train:
            pose, selected_indexes = pose.frame_dropout(self.frame_dropout_std)

            # selected_indexes are the ones that are _not_ dropped
            # sub-select frame labels that are not dropped
            tgt = tf.gather(tgt, selected_indexes)

        if self.normalize_pose:
            # normalize by shoulder width
            pose.normalize(pose.header.normalization_info(
                p1=("pose_keypoints_2d", "RShoulder"),
                p2=("pose_keypoints_2d", "LShoulder")
            ))

        if self.scale_pose:
            # TODO: perhaps also average over all keypoints as Bull et al?
            pose.normalize_distribution(axis=(0, 1))

        # (Frames, People, Points, Dims) - eg (93, 1, 137, 2)
        src = pose.body.data

        # (Frames, Points, Dims)
        src = src.squeeze(axis=1)

        # (Frames, Points * Dims)
        src = src.reshape(shape=(-1, 137 * 2))

        # remove Nan values
        src = src.fix_nan().zero_filled()

        return {"src": src, "tgt": tgt}

    def prepare_io(self, datum: dict):
        """
        Convert dictionary into input-output tuple for Keras.

        :param datum:
        :return:
        """
        src = datum["src"]
        tgt = datum["tgt"]

        return src, tgt

    def batch_dataset(self, dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        """
        Batch and pad a dataset.

        :param dataset:
        :param batch_size:
        :return:
        """
        dataset = dataset.padded_batch(
            batch_size, padded_shapes={
                "src": [None, None],
                "tgt": [None]
            })

        return dataset.map(self.prepare_io)

    def maybe_apply_length_constraints(self,
                                       dataset: tf.data.Dataset,
                                       dataset_name: str) -> tf.data.Dataset:
        """
        If a maximum number of frames is defined, applies a specific strategy (removing, splitting
        or truncating) to examples that have too many frames.

        :param dataset:
        :param dataset_name:
        :return:
        """
        def length_is_acceptable(example: dict) -> bool:
            return self.max_num_frames >= example["frames"] and example["frames"] >= self.min_num_frames

        num_examples_before = get_dataset_size(dataset)

        logging.debug("Filtering dataset '%s'...", dataset_name)

        if self.max_num_frames_strategy == "remove":
            dataset = dataset.filter(length_is_acceptable)
        elif self.max_num_frames_strategy == "split":
            #
        else:
            raise NotImplementedError("Length constraint strategy '%s' not implemented." % self.max_num_frames_strategy)

        num_examples_after = get_dataset_size(dataset)

        logging.debug("Number of examples before/after applying length constraints: %d/%d",
                      num_examples_before, num_examples_after)

        return dataset

    def train_pipeline(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Prepare the training dataset.

        :param dataset:
        :return:
        """
        logging.debug("Preparing training pipeline...")

        dataset = dataset.map(self.load_datum).cache()

        logging.debug("AFTER load_datum")
        log_raw_datum_examples(dataset, max_index=2)

        dataset = self.maybe_apply_length_constraints(dataset, dataset_name="train")

        logging.debug("AFTER maybe_apply_length_constraints")
        log_raw_datum_examples(dataset, max_index=2)

        dataset = dataset.map(lambda d: self.process_datum(datum=d, is_train=True),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.repeat().shuffle(self.batch_size)

        dataset = self.batch_dataset(dataset, self.batch_size)

        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def test_pipeline(self,
                      dataset: tf.data.Dataset,
                      dataset_name: str) -> tf.data.Dataset:
        """
        Prepare the test dataset.

        :param dataset:
        :param dataset_name:
        :return:
        """
        logging.debug("Preparing %s pipeline...", dataset_name)

        dataset = dataset.map(self.load_datum)

        logging.debug("AFTER load_datum")
        log_raw_datum_examples(dataset, max_index=2)

        dataset = self.maybe_apply_length_constraints(dataset, dataset_name=dataset_name)

        logging.debug("AFTER maybe_apply_length_constraints")
        log_raw_datum_examples(dataset, max_index=2)

        dataset = dataset.map(lambda d: self.process_datum(datum=d, is_train=False))
        dataset = self.batch_dataset(dataset, self.test_batch_size)

        return dataset.cache()

    def split_dataset(self, dataset: tf.data.Dataset) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Split dataset to train, dev, and test."""

        def is_dev(x, _) -> bool:
            # Every 7th item
            return x % 8 == 6

        def is_test(x, _) -> bool:
            # Every 8th item
            return x % 8 == 7

        def is_train(x, y) -> bool:
            return not is_test(x, y) and not is_dev(x, y)

        def recover(_, y):
            return y

        train = self.train_pipeline(dataset.enumerate().filter(is_train).map(recover))
        dev = self.test_pipeline(dataset.enumerate().filter(is_dev).map(recover), dataset_name="dev")
        test = self.test_pipeline(dataset.enumerate().filter(is_test).map(recover), dataset_name="test")

        return train, dev, test

    def get_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Get train, dev, and test datasets.

        :return:
        """
        # Set features
        features = {"tags": tf.io.FixedLenFeature([], tf.string)}
        features.update(TF_POSE_RECORD_DESCRIPTION)

        tfrecord_path = os.path.join(self.data_dir, TFRECORD_FILE_NAME)

        # Dataset iterator
        dataset = tf.data.TFRecordDataset(filenames=[tfrecord_path])
        dataset = dataset.map(
            lambda serialized: tf.io.parse_single_example(serialized, features))

        return self.split_dataset(dataset)
