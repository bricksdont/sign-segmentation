"""Training command line arguments."""
from absl import flags

FLAGS = flags.FLAGS

# Training flags
flags.DEFINE_integer('seed', 1, 'Random seed')
flags.DEFINE_string('device', '/GPU:0', 'Tensorflow device')
flags.DEFINE_string('model_path', '/tmp/model.h5', 'Path to save trained model')
flags.DEFINE_integer('epochs', 100, 'Maximum number of epochs')
flags.DEFINE_integer('steps_per_epoch', 32, 'Number of batches per epoch')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_integer('stop_patience', 10, 'Patience for early stopping')
flags.DEFINE_integer('batch_size', 8, 'Batch size for training')
flags.DEFINE_integer('test_batch_size', 1, 'Batch size for evaluation')

# Model flags
flags.DEFINE_float('input_dropout', 0.3, 'Input dropout rate')
flags.DEFINE_integer('encoder_layers', 3, 'Number of RNN layers')
flags.DEFINE_bool('encoder_bidirectional', False, 'Use a bidirectional encoder?')
flags.DEFINE_integer('hidden_size', 2 ** 8, 'RNN hidden state size')

# Augmentation flags
flags.DEFINE_float('frame_dropout_std', 0.3, 'Augmentation drop frames std')

# Dataset flags
flags.DEFINE_integer('input_size', 75 * 3, 'Number of pose points')

flags.mark_flag_as_required('input_size')
