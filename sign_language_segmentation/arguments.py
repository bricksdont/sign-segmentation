#! /usr/bin/python3

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    # Training Arguments
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default="/GPU:0", help='training device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--steps_per_epoch', type=int, default=32, help='Number of batches per epoch')
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--min_delta", type=float, default=0.0,
                        help="Minimum delta for improvement with early stopping."
                             "Default: 0.")

    # sizes and number of layers
    parser.add_argument('--input_dropout', type=float, default=0.3, help='Input dropout rate')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of RNN layers.')
    parser.add_argument('--encoder_bidirectional', type=bool, default=False, help='Use a bidirectional encoder?')
    parser.add_argument('--hidden_size', type=int, default=256, help='RNN hidden state size.')

    # data set details and preprocessing
    parser.add_argument('--data_dir', type=str, default="data", metavar='PATH',
                        help="Where to look for tfrecord dataset.")

    parser.add_argument('--frame_dropout', action='store_true', default=False, help='Whether to apply frame dropout.')
    parser.add_argument('--frame_dropout_type', type=str, default=False, choices=["normal", "uniform"],
                        help='Type of distribution to draw dropout percentage from.')
    parser.add_argument('--frame_dropout_std', type=float, default=0.1,
                        help='If frame dropout is used, and dropout percentage is drawn from a normal distribution,'
                             'this defines the standard deviation.')
    parser.add_argument('--frame_dropout_mean', type=float, default=0.5,
                        help='If frame dropout is used, and dropout percentage is drawn from a normal distribution,'
                             'this defines the mean.')
    parser.add_argument('--frame_dropout_min', type=float, default=0.0,
                        help='If frame dropout is used, and dropout percentage is drawn from a uniform distribution,'
                             'this defines the lower bound.')
    parser.add_argument('--frame_dropout_max', type=float, default=0.5,
                        help='If frame dropout is used, and dropout percentage is drawn from a uniform distribution,'
                             'this defines the upper bound.')

    parser.add_argument('--num_keypoints', type=int, default=137, help='Number of pose points')
    parser.add_argument('--max_num_frames', type=int, default=-1, help='Remove, truncate or split examples with more '
                                                                       'frames (exact behaviour depends '
                                                                       'on --max_num_frames_strategy and --desired_fps)')
    parser.add_argument('--min_num_frames', type=int, default=0, help='Remove examples with fewer '
                                                                      'frames (exact behaviour depends '
                                                                      'on --desired_fps)')
    parser.add_argument('--max_num_frames_strategy', type=str, default="remove", help='What to do with examples that '
                                                                                      'have too many frames',
                        choices=["remove", "truncate", "slice"])

    parser.add_argument('--pose_type', type=str, default="openpose", help='Type of pose features',
                        choices=["openpose", "holistic"])
    parser.add_argument('--normalize_pose', action='store_true', default=False, help='Normalize poses by'
                                                                                     'shoulder width.')
    parser.add_argument('--scale_pose', action='store_true', default=False, help='Scale pose to zero mean and'
                                                                                 'unit variance, individually for'
                                                                                 'x and y dimensions.')

    # directories and checkpoints
    parser.add_argument('--model_path', type=str, default="checkpoints/model.h5", metavar='PATH',
                        help="Where to save model checkpoints.")

    # Logging arguments (not used at the moment, in case I add wandb logging)
    parser.add_argument('--offline', action='store_true', default=False, help='Set wandb to offline mode.')
    parser.add_argument('--log_model_off', action='store_true', default=False, help='Turn off wandb logging.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Name of wandb run.')

    args = parser.parse_args()

    return args
