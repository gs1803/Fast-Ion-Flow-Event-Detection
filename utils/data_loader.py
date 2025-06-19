import numpy as np
import tensorflow as tf

def generate_timeseries(X, y, n_timesteps, batch_size, start_idx, end_idx, stride):
    """
    Generator function that yields batches of time series data for training or evaluation.

    This function creates overlapping sliding windows of time series sequences and their
    corresponding labels, yielding both timestep-level and sequence-level targets.

    Args:
        X (np.ndarray): Input features of shape (num_samples, num_features).
        y (np.ndarray): Target time series values of shape (num_samples,).
        n_timesteps (int): Number of time steps in each input sequence.
        batch_size (int): Number of sequences per batch.
        start_idx (int): Starting index for the sliding window.
        end_idx (int): Ending index for the sliding window.
        stride (int): Step size for sliding the window.

    Yields:
        tuple: A batch tuple containing:
            - X_batch (tf.Tensor): Shape (batch_size, n_timesteps, num_features)
            - targets (dict): Dictionary with:
                - "time_output": tf.Tensor of shape (batch_size, n_timesteps, 1)
                - "sequence_output": tf.Tensor of shape (batch_size, 1)
    """
    while True:
        X_batch = []
        y_batch = []
        y_seq_batch = []

        for i in range(start_idx + n_timesteps, end_idx, stride):
            X_batch.append(X[i - n_timesteps:i, :])
            y_batch.append(y[i - n_timesteps:i].reshape(-1, 1))
            y_seq_batch.append([np.mean(y[i - n_timesteps:i]).astype(np.float32)])

            if len(X_batch) == batch_size:
                yield (
                    tf.convert_to_tensor(np.array(X_batch), dtype=tf.float32),
                    {
                        "time_output": tf.convert_to_tensor(np.array(y_batch), dtype=tf.float32),
                        "sequence_output": tf.convert_to_tensor(np.array(y_seq_batch), dtype=tf.float32)
                    }
                )
                X_batch, y_batch, y_seq_batch = [], [], []

        if len(X_batch) > 0:
            yield (
                tf.convert_to_tensor(np.array(X_batch), dtype=tf.float32),
                {
                    "time_output": tf.convert_to_tensor(np.array(y_batch), dtype=tf.float32),
                    "sequence_output": tf.convert_to_tensor(np.array(y_seq_batch), dtype=tf.float32)
                }
            )


def create_dataset(X, y, n_timesteps, batch_size, stride, start_idx, end_idx):
    """
    Creates a TensorFlow Dataset from a time series generator for use in model training or evaluation.

    Args:
        X (np.ndarray): Input features of shape (num_samples, num_features).
        y (np.ndarray): Target time series values of shape (num_samples,).
        n_timesteps (int): Number of time steps in each input sequence.
        batch_size (int): Number of sequences per batch.
        stride (int): Step size for sliding the window.
        start_idx (int): Starting index for the sliding window.
        end_idx (int): Ending index for the sliding window.

    Returns:
        tf.data.Dataset: A TensorFlow dataset yielding batches of (X_batch, target_dict),
                         where target_dict contains "time_output" and "sequence_output".
    """
    return tf.data.Dataset.from_generator(
        lambda: generate_timeseries(X, y, n_timesteps=n_timesteps, batch_size=batch_size,
                                    start_idx=start_idx, end_idx=end_idx, stride=stride),
        output_signature=(
            tf.TensorSpec(shape=(None, n_timesteps, X.shape[1]), dtype=tf.float32),
            {
                "time_output": tf.TensorSpec(shape=(None, n_timesteps, 1), dtype=tf.float32),
                "sequence_output": tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            }
        )
    ).prefetch(tf.data.AUTOTUNE)
