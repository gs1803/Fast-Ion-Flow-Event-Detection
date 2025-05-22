import numpy as np
import tensorflow as tf

# Generator function to generate train and test batches without loading all the data into memory
def generate_timeseries(X, y, n_timesteps, batch_size, start_idx, end_idx, stride):
    while True:
        X_batch = []        # Stores input sequences for the batch
        y_batch = []        # Stores target time series values corresponding to each input timestep
        y_seq_batch = []    # Stores sequence-level aggregated target values (e.g., mean over the sequence)

        # Loop through the dataset with a sliding window
        for i in range(start_idx + n_timesteps, end_idx, stride):
            X_batch.append(X[i - n_timesteps:i, :])
            y_batch.append(y[i - n_timesteps:i].reshape(-1, 1))  # Ensure y is shaped as (n_timesteps, 1)
            y_seq_batch.append([np.mean(y[i - n_timesteps:i]).astype(np.float32)])  # Sequence-level label

            # Once a full batch is ready, yield it
            if len(X_batch) == batch_size:
                yield (
                    tf.convert_to_tensor(np.array(X_batch), dtype=tf.float32),
                    {
                        "time_output": tf.convert_to_tensor(np.array(y_batch), dtype=tf.float32),
                        "sequence_output": tf.convert_to_tensor(np.array(y_seq_batch), dtype=tf.float32)
                    }
                )
                # Reset the batch containers
                X_batch, y_batch, y_seq_batch = [], [], []
        
        # Yield the last partial batch if it exists
        if len(X_batch) > 0:
            yield (
                tf.convert_to_tensor(np.array(X_batch), dtype=tf.float32),
                {
                    "time_output": tf.convert_to_tensor(np.array(y_batch), dtype=tf.float32),
                    "sequence_output": tf.convert_to_tensor(np.array(y_seq_batch), dtype=tf.float32)
                }
            )


# Tensorflow dataloader for the generator function to use with tf models
def create_dataset(X, y, n_timesteps, batch_size, stride, start_idx, end_idx):
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