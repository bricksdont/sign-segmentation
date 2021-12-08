#! /usr/bin/python3

"""Code to create tfrecord for training from The Public DGS Corpus."""

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from argparse import ArgumentParser

import tensorflow_datasets as tfds
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences

def parse_args_tfrecord():

    parser = ArgumentParser()

    # directories and checkpoints
    parser.add_argument('--data_dir', type=str, required=False, metavar='PATH',
                        help="Tensorflow dataset directory. Default: $HOME.")

    args = parser.parse_args()

    print(args)

    return args

def create_tfrecord_dataset(args: argparse.Namespace):

    config = SignDatasetConfig(name="annotations-pose", version="1.0.0", include_video=False, include_pose="holistic")
    dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=config), data_dir=args.data_dir)


    def time_frame(ms, fps):
        return int(fps * (ms / 1000))


    # Body and two hands, ignoring the face
    body_points = list(range(33)) + list(range(33 + 468, 33 + 468 + 21 * 2))

    with tf.io.TFRecordWriter('data.tfrecord') as writer:
        for datum in tqdm(dgs_corpus["train"]):
            elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')
            sentences = get_elan_sentences(elan_path)

            for person in ["a", "b"]:
                frames = len(datum["poses"][person]["data"])
                fps = int(datum["poses"][person]["fps"].numpy())

                pose_data = datum["poses"][person]["data"].numpy()[:, :, body_points, :]
                pose_conf = datum["poses"][person]["conf"].numpy()[:, :, body_points]

                bio = np.zeros(datum["poses"][person]["data"].shape[0], dtype=np.int8)
                timing = np.full(datum["poses"][person]["data"].shape[0], fill_value=-1, dtype=np.float)

                for sentence in sentences:
                    if sentence["participant"].lower() == person:
                        for gloss in sentence["glosses"]:
                            start_frame = time_frame(gloss["start"], fps)
                            end_frame = time_frame(gloss["end"], fps)

                            bio[start_frame] = 2  # B to beginning
                            bio[start_frame + 1:end_frame + 1] = 1  # I for in

                            timing[start_frame:end_frame] = np.linspace(start=0, stop=gloss["end"] - gloss["start"],
                                                                        num=end_frame - start_frame)

                tags = bio.tobytes()
                timing = tf.io.serialize_tensor(timing).numpy()
                pose_data = tf.io.serialize_tensor(pose_data).numpy()
                pose_conf = tf.io.serialize_tensor(pose_conf).numpy()

                features = {
                    'fps': tf.train.Feature(int64_list=tf.train.Int64List(value=[fps])),
                    'pose_data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_data])),
                    'pose_confidence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pose_conf])),
                    'tags': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tags])),
                    'timing': tf.train.Feature(bytes_list=tf.train.BytesList(value=[timing]))
                }

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())

if __name__ == '__main__':
    args = parse_args_tfrecord()
    create_tfrecord_dataset(args)
