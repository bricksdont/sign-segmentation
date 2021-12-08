#! /usr/bin/python3

from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser()

    # Training Arguments
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default="/GPU:0", help='training device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--save_top_k', type=int, default=1, help='Keep k best checkpoint')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for.')
    parser.add_argument('--steps_per_epoch', type=int, default=32, help='Number of batches per epoch')
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum delta for improvement with early stopping. Default: 0.")

    # sizes and number of layers
    parser.add_argument('--input_dropout', type=float, default=0.3, help='Input dropout rate')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='Number of RNN layers.')
    parser.add_argument('--encoder_bidirectional', type=bool, default=False, help='Use a bidirectional encoder?')
    parser.add_argument('--hidden_size', type=int, default=256, help='RNN hidden state size.')

    # data set details
    parser.add_argument('--dataset_path', type=str, default="data", metavar='PATH',
                        help="Where to save data.")
    parser.add_argument('--frame_dropout_std', type=float, default=0.3, help='Augmentation drop frames std')
    parser.add_argument('--input_size', type=int, default=75 * 3, help='Number of pose points')

    # directories and checkpoints
    parser.add_argument('--data_dir', type=str, required=False, metavar='PATH',
                        help="Tensorflow dataset directory. Default: $HOME.")
    parser.add_argument('--model_path', type=str, default="checkpoints/model.h5", metavar='PATH',
                        help="Where to save model checkpoints.")
    parser.add_argument('--resume_from_checkpoint', type=str,
                        default=None,
                        required=False, metavar='PATH',
                        help="Resume training from this checkpoint.")

    # Logging arguments
    parser.add_argument('--offline', action='store_true', default=False, help='Set wandb to offline mode.')
    parser.add_argument('--log_model_off', action='store_true', default=False, help='Turn off wandb logging.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Name of wandb run.')

    # Prediction args
    parser.add_argument('--pred_checkpoint', type=str,
                        metavar='PATH', help="Checkpoint path for prediction")
    parser.add_argument('--pred_output', type=str,
                        metavar='PATH', help="Path for saving prediction files")
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for evaluation')

    args = parser.parse_args()

    print(args)

    return args
