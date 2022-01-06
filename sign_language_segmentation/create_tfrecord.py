#! /usr/bin/python3

"""Code to create tfrecord for training from The Public DGS Corpus."""

import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf

import argparse
import tensorflow_datasets as tfds
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences


def parse_args_tfrecord():

    parser = argparse.ArgumentParser()

    # directories and checkpoints
    parser.add_argument('--data_dir', type=str, required=False, metavar='PATH',
                        help="Tensorflow dataset directory. Default: $HOME.")
    parser.add_argument('--download_max_retries', type=int, default=1, help='Retry tfds download N times.')
    parser.add_argument('--label_type', type=str, default="sentence", help='Whether BIO labels apply to'
                                                                           'individual glosses or entire sentences',
                        choices=["gloss", "sentence"])

    args = parser.parse_args()

    print(args)

    return args


def miliseconds_to_frame_index(ms: int, fps: int) -> int:
    """

    :param ms:
    :param fps:
    :return:
    """
    return int(fps * (ms / 1000))


def create_tfrecord_dataset(args: argparse.Namespace):

    config = SignDatasetConfig(name="annotations-pose", version="1.0.0", include_video=False, include_pose="openpose")

    retries = 0

    while True:
        try:
            dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=config), data_dir=args.data_dir)
            break
        except tfds.download.download_manager.NonMatchingChecksumError:
            if retries == args.download_max_retries:
                print("Reached maximum number of download retries. Download failed at some point.")
                raise
            else:
                print("Download failed at some point. Will retry download.")
                retries += 1
                continue

    print("Finished loading DGS corpus.")

    tfrecord_path = os.path.join(args.data_dir, "data.tfrecord")

    if os.path.isfile(tfrecord_path):
        print("Tfrecord already exists: '%s'" % tfrecord_path)
        print("Skipping.")
    else:

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for datum in tqdm(dgs_corpus["train"]):

                elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')

                sentences = get_elan_sentences(elan_path)

                for person in ["a", "b"]:

                    fps = int(datum["poses"][person]["fps"].numpy())

                    pose_data = datum["poses"][person]["data"].numpy()
                    pose_conf = datum["poses"][person]["conf"].numpy()

                    pose_num_frames = datum["poses"][person]["data"].shape[0]

                    bio = np.zeros(pose_num_frames, dtype=np.int8)

                    for sentence in sentences:
                        if sentence["participant"].lower() == person:

                            glosses = sentence["glosses"]

                            if args.label_type == "gloss":
                                for gloss in glosses:
                                    start_frame = miliseconds_to_frame_index(gloss["start"], fps)
                                    end_frame = miliseconds_to_frame_index(gloss["end"], fps)

                                    bio[start_frame] = 2  # B for beginning
                                    bio[start_frame + 1:end_frame + 1] = 1  # I for in
                            else:
                                # assume label type is sentence
                                # get start frame of first gloss, end frame of last gloss

                                first_gloss = glosses[0]
                                last_gloss = glosses[-1]

                                start_frame = miliseconds_to_frame_index(first_gloss["start"], fps)
                                end_frame = miliseconds_to_frame_index(last_gloss["end"], fps)

                                bio[start_frame] = 2  # B for beginning
                                bio[start_frame + 1:end_frame + 1] = 1  # I for in

                    tags = bio.tobytes()
                    pose_data = tf.io.serialize_tensor(pose_data).numpy()
                    pose_conf = tf.io.serialize_tensor(pose_conf).numpy()

                    features = {
                        'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[fps])),
                        'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_data])),
                        'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_conf])),
                        'tags': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tags]))
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(example.SerializeToString())


if __name__ == '__main__':
    args = parse_args_tfrecord()
    create_tfrecord_dataset(args)
