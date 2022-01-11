#! /usr/bin/python3

"""Code to create tfrecord for training from The Public DGS Corpus."""

import os
import argparse
import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm

# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences


def parse_args_tfrecord():

    parser = argparse.ArgumentParser()

    # directories and checkpoints
    parser.add_argument('--data_dir', type=str, required=False, metavar='PATH',
                        help="Tensorflow dataset directory. Default: $HOME.")
    parser.add_argument('--download_max_retries', type=int, default=1, help='Retry tfds download N times. (default: 1)')
    parser.add_argument('--max_num_examples', type=int, default=-1, help='Max number of examples to write as tfrecords. '
                                                                         '(default: -1 = all examples)')
    parser.add_argument('--label_type', type=str, default="sentence", help='Whether BIO labels apply to'
                                                                           'individual glosses or entire sentences. '
                                                                           '(default: "sentence")',
                        choices=["gloss", "sentence"])
    parser.add_argument('--pose_type', type=str, default="openpose", help='Type of pose features (default: "openpose")',
                        choices=["openpose", "holistic"])
    parser.add_argument('--skip_if_num_frames_zero', action='store_true', default=False,
                        help='Skip examples if they have zero frames (default: False)')

    args = parser.parse_args()

    return args


def miliseconds_to_frame_index(ms: int, fps: int) -> int:
    """

    :param ms:
    :param fps:
    :return:
    """
    return int(fps * (ms / 1000))


class RecordCReator:

    def __init__(self, pose_type: str, data_dir: str, download_max_retries: int,
                 skip_if_num_frames_zero: bool, label_type: str, max_num_examples: int):
        """

        :param pose_type:
        :param data_dir:
        :param download_max_retries:
        :param skip_if_num_frames_zero:
        :param label_type:
        :param max_num_examples:
        """

        self.pose_type = pose_type
        self.data_dir = data_dir
        self.download_max_retries = download_max_retries
        self.skip_if_num_frames_zero = skip_if_num_frames_zero
        self.label_type = label_type
        self.max_num_examples = max_num_examples

        if self.max_num_examples == -1:
            self.max_num_examples = np.inf

        self.config = SignDatasetConfig(name="annotations-pose",
                                        version="1.0.0",
                                        include_video=False,
                                        include_pose=self.pose_type)
        self.dgs_corpus = None
        self.tfrecord_path = None

    def load_tfds_data(self):

        retries = 0

        while True:
            try:
                self.dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=self.config), data_dir=self.data_dir)
                break
            except tfds.download.download_manager.NonMatchingChecksumError:
                if retries == self.download_max_retries:
                    logging.debug("Reached maximum number of download retries. Download failed at some point.")
                    raise
                else:
                    logging.debug("Download failed at some point. Will retry download.")
                    retries += 1
                    continue

        logging.debug("Finished loading DGS corpus.")

    def create_tfrecord_dataset_if_does_not_exist(self):

        self.tfrecord_path = os.path.join(args.data_dir, "data.tfrecord")

        if os.path.isfile(self.tfrecord_path):
            logging.debug("Tfrecord already exists: '%s'" % self.tfrecord_path)
            logging.debug("Skipping.")
        else:
            self.create_tfrecord_dataset()

    def get_data_for_single_person(self, datum, person, sentences):
        """

        :param datum:
        :param person:
        :param sentences:
        :return:
        """
        fps = int(datum["poses"][person]["fps"].numpy())

        pose_data = datum["poses"][person]["data"].numpy()
        pose_conf = datum["poses"][person]["conf"].numpy()

        pose_num_frames = datum["poses"][person]["data"].shape[0]

        if pose_num_frames == 0 and self.skip_if_num_frames_zero:
            return None

        bio = np.zeros(pose_num_frames, dtype=np.int8)

        for sentence in sentences:
            if sentence["participant"].lower() == person:

                glosses = sentence["glosses"]

                if self.label_type == "gloss":
                    for gloss in glosses:
                        start_frame = miliseconds_to_frame_index(gloss["start"], fps)
                        end_frame = miliseconds_to_frame_index(gloss["end"], fps)

                        bio[start_frame] = 2  # B for beginning
                        bio[start_frame + 1:end_frame + 1] = 1  # I for in
                else:
                    # assume label type is sentence
                    # get start frame of first gloss, end frame of last gloss

                    if len(glosses) == 0:
                        continue

                    first_gloss = glosses[0]
                    last_gloss = glosses[-1]

                    start_frame = miliseconds_to_frame_index(first_gloss["start"], fps)
                    end_frame = miliseconds_to_frame_index(last_gloss["end"], fps)

                    bio[start_frame] = 2  # B for beginning
                    bio[start_frame + 1:end_frame + 1] = 1  # I for in

        return fps, pose_data, pose_conf, bio

    def create_tfrecord_dataset(self):
        """

        :return:
        """

        skipped_because_num_pose_frames_zero = 0
        num_examples = 0

        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for datum in tqdm(self.dgs_corpus["train"]):

                if num_examples == self.max_num_examples:
                    break

                elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')
                sentences = get_elan_sentences(elan_path)

                for person in ["a", "b"]:

                    data_for_single_person = self.get_data_for_single_person(datum, person, sentences)

                    if data_for_single_person is None:
                        skipped_because_num_pose_frames_zero += 1
                        continue
                    else:
                        num_examples += 1

                    fps, pose_data, pose_conf, bio = data_for_single_person

                    if num_examples == 1:
                        logging.debug("fps=%s, pose_data.shape=%s, pose_conf.shape=%s, bio.shape=%s",
                                      fps, pose_data.shape, pose_conf.shape, bio.shape)

                    pose_data = tf.io.serialize_tensor(pose_data).numpy()
                    pose_conf = tf.io.serialize_tensor(pose_conf).numpy()

                    tags = bio.tobytes()

                    features = {
                        'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[fps])),
                        'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_data])),
                        'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_conf])),
                        'tags': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tags]))
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())

        logging.debug("Skipped examples because num_pose_frames was zero:")
        logging.debug(skipped_because_num_pose_frames_zero)

        logging.debug("num_examples:")
        logging.debug(num_examples)

        logging.debug("Finished writing TFRecord data.")


if __name__ == '__main__':
    args = parse_args_tfrecord()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    tfrecord_creator = RecordCReator(pose_type=args.pose_type,
                                     data_dir=args.data_dir,
                                     download_max_retries=args.download_max_retries,
                                     skip_if_num_frames_zero=args.skip_if_num_frames_zero,
                                     label_type=args.label_type,
                                     max_num_examples=args.max_num_examples)

    tfrecord_creator.load_tfds_data()
    tfrecord_creator.create_tfrecord_dataset_if_does_not_exist()
