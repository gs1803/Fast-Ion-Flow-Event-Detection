import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from arch import arch_model

import tensorflow as tf
from keras import models, losses


@tf.keras.utils.register_keras_serializable(package='Custom', name='TverskyBCEPerSequence')
class TverskyBCEPerSequence(losses.Loss):
    """
    Custom loss function combining Tversky loss and binary cross-entropy (BCE) with focal loss.
    It calculates loss per sequence and applies event weighting and smoothing.
    """
    
    def __init__(self, alpha_t: float = 0.5, beta_t: float = 0.5, alpha_f: float = 0.5, gamma_f: float = 0.0,
                 event_weight: float = 1.0, smooth: float = 1e-6,
                 reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name: str = "tversky_bce_per_sequence") -> None:
        """
        Initializes the TverskyBCEPerSequence loss function.
        
        Parameters:
            alpha_t: Weight for false positives in the Tversky index.
            beta_t: Weight for false negatives in the Tversky index.
            alpha_f: Focal loss alpha parameter.
            gamma_f: Focal loss gamma parameter.
            event_weight: Weight applied to events in the loss calculation.
            smooth: Small smoothing value to avoid division by zero.
            reduction: Specifies the reduction method for the loss.
            name: The name of the loss function.
        """
        super().__init__(reduction=reduction, name=name)
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.alpha_f = alpha_f
        self.gamma_f = gamma_f
        self.event_weight = event_weight
        self.smooth = smooth


    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the custom loss value combining Tversky and BCE losses.

        Args:
            y_true (tensor): Ground truth values.
            y_pred (tensor): Predicted values.

        Returns:
            tensor: The calculated loss value.
        """
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)

        has_event = tf.cast(tf.reduce_sum(y_true, axis=1) > 0, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred, axis=1)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=1)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=1)

        tversky = (tp + self.smooth) / (tp + self.alpha_t * fp + self.beta_t * fn + self.smooth)
        fbce = losses.binary_focal_crossentropy(y_true, y_pred, alpha=self.alpha_f, gamma=self.gamma_f)
        
        final_loss = has_event * self.event_weight * (1 - tversky) + (1 - has_event) * fbce 

        return final_loss


    def get_config(self) -> dict:
        """
        Get the configuration of the loss function for serialization.

        Returns:
            dict: Config dictionary with loss parameters.
        """
        config = super().get_config()
        config.update({
            "alpha_t": self.alpha_t,
            "beta_t": self.beta_t,
            "alpha_f": self.alpha_f,
            "gamma_f": self.gamma_f,
            "event_weight": self.event_weight,
            "smooth": self.smooth
        })
        
        return config


class DetectEvents:
    """
    Class for event detection based on magnetic field data using a pre-trained model.

    Attributes:
        RE (float): Earth's radius in km, used for filtering magnetic field data.
        batch_size (int): Batch size for model inference.
        n_timesteps (int): Number of timesteps per input sequence.
        stride (int): Stride between sequences when creating input batches.
        threshold (float): Probability threshold for classifying events.
        model_path (str): Path to the pre-trained model.
        model (tensorflow.keras.Model): The loaded model used for predictions.
    """
    def __init__(self, model_path, RE: float=6300, batch_size: int=256, n_timesteps: int=500, stride: int=40, threshold: float=0.5) -> None:
        """
        Initialize the event detection pipeline.

        Args:
            model_path (str): Path to the pre-trained model.
            RE (float): Earth's radius in km.
            batch_size (int): Batch size for model inference.
            n_timesteps (int): Number of timesteps per input sequence.
            stride (int): Stride between sequences when creating input batches.
            threshold (float): Probability threshold for event classification.
            return_df_features (bool): Returns the dataset with generated features
        """
        self.RE = RE
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        self.stride = stride
        self.threshold = threshold
        self.model_path = model_path
        self.model = self._load_model()
        self.return_df_features = True


    def _load_model(self) -> tf.keras.Model:
        """
        Load the pre-trained model from the specified path.

        Returns:
            tensorflow.keras.Model: The loaded model.
        """
        return models.load_model(
            self.model_path,
            custom_objects={"tversky_bce_per_sequence": TverskyBCEPerSequence}
        )


    def tanh_squash(self, x: np.ndarray, scale: float=1.0) -> np.ndarray:
        """
        Apply a hyperbolic tangent (tanh) function with scaling.

        Args:
            x (numpy.array): Input array.
            scale (float): Scaling factor for the tanh function.

        Returns:
            numpy.array: Squashed values.
        """
 
        return np.tanh(x / scale)


    def _nightside_filter(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw dataframe by filtering, dropping NaN, and renaming columns.

        Args:
            raw_df (pandas.DataFrame): Raw dataframe with magnetic field data.

        Returns:
            pandas.DataFrame: Preprocessed dataframe.
        """

        df = (raw_df[
                (raw_df['GSM_x'] < -9 * self.RE) &
                (raw_df['GSM_y'].abs() < raw_df['GSM_x'].abs())
            ]
            .copy()
            .dropna()
            .drop_duplicates(subset=['Time'], keep='last')
            .sort_values(by='Time')
            .drop(columns=['GSM_x', 'GSM_y'])
            .rename(columns={'Time': 'Epoch_time'})
            .reset_index(drop=True)
        )
        return df


    def _segment_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """
        Segment dataframe into sequences based on time differences greater than 300 seconds.

        Args:
            df (pandas.DataFrame): Preprocessed dataframe.

        Returns:
            list: List of index-based sequences.
        """

        return df.groupby((df['Epoch_time'].diff() > 300).cumsum()).apply(lambda x: list(x.index)).values


    def _compute_features(self, df: pd.DataFrame, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute features (volatilities, rolling standard deviations, and scaled values) for each sequence.

        Args:
            df (pandas.DataFrame): Preprocessed dataframe.
            sequences (list): List of index-based sequences.

        Returns:
            tuple: Computed features (scaled values, volatilities, and standard deviations).
        """
        bx_vol, by_vol, bz_vol = [], [], []
        bx_std, by_std, bz_std = [], [], []
        bx_scaled, by_scaled, bz_scaled = [], [], []

        for seq in sequences:
            segment = df.loc[seq[0]:seq[-1]].copy()
            segment = segment[['Bx', 'By', 'Bz']]

            lof = LocalOutlierFactor(n_neighbors=29)
            flags = lof.fit_predict(segment)
            segment.loc[flags == -1] = np.nan
            segment = segment.interpolate().bfill()

            for col, store in zip(['Bx', 'By', 'Bz'], [bx_vol, by_vol, bz_vol]):
                model = arch_model(segment[col] * 10, mean='AR', lags=1, vol='GARCH', p=1, q=1, rescale=False)
                vol = model.fit(disp="off", show_warning=False).conditional_volatility.bfill()
                store.extend(self.tanh_squash(vol, scale=10.0))

            stds = segment.rolling(9, center=True, min_periods=1).std()
            for std, store in zip([stds['Bx'], stds['By'], stds['Bz']], [bx_std, by_std, bz_std]):
                store.extend(self.tanh_squash(std, scale=1.0))

            scaled = MinMaxScaler().fit_transform(segment)
            bx_scaled.extend(scaled[:, 0])
            by_scaled.extend(scaled[:, 1])
            bz_scaled.extend(scaled[:, 2])

        return bx_scaled, by_scaled, bz_scaled, bx_vol, by_vol, bz_vol, bx_std, by_std, bz_std


    def _prepare_model_input(self, df: pd.DataFrame, bx_scaled: np.ndarray, by_scaled: np.ndarray, bz_scaled: np.ndarray,
                             bx_vol: np.ndarray, by_vol: np.ndarray, bz_vol: np.ndarray, bx_std: np.ndarray, 
                             by_std: np.ndarray, bz_std: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Prepares the feature matrix for model input.

        Parameters:
        - df: Preprocessed DataFrame.
        - bx_scaled, by_scaled, bz_scaled, bx_vol, by_vol, bz_vol, bx_std, by_std, bz_std: Computed features.

        Returns:
        - X: Feature matrix.
        - total_len: Total number of rows in the prepared data.
        """
        df_feat = df[['Epoch_time']].copy()
        df_feat['Bx'], df_feat['By'], df_feat['Bz'] = bx_scaled, by_scaled, bz_scaled

        for col in ['Bx', 'By', 'Bz']:
            for lag in range(1, 3):
                df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)

        df_feat['Bx_conditional_vol'] = bx_vol
        df_feat['By_conditional_vol'] = by_vol
        df_feat['Bz_conditional_vol'] = bz_vol
        df_feat['Bx_rolling_stdev'] = bx_std
        df_feat['By_rolling_stdev'] = by_std
        df_feat['Bz_rolling_stdev'] = bz_std

        df_feat = df_feat.dropna().reset_index(drop=True)
        features = [col for col in df_feat.columns if col != 'Epoch_time']
        X = df_feat[features].values

        if self.return_df_features:
            return X, len(df_feat), df_feat
        else:
            return X, len(df_feat)


    def _create_dataset(self, X: np.ndarray) -> tf.data.Dataset:
        """
        Creates a dataset generator for batching input data.

        Parameters:
        - X: Feature matrix.

        Returns:
        - dataset: TensorFlow dataset object.
        """
        def generator():
            X_batch = []

            for i in range(self.n_timesteps, len(X), self.stride):
                X_batch.append(X[i - self.n_timesteps:i, :])

                if len(X_batch) == self.batch_size:
                    yield tf.convert_to_tensor(np.array(X_batch), dtype=tf.float32)
                    X_batch = []

            if len(X_batch) > 0:
                yield tf.convert_to_tensor(np.array(X_batch), dtype=tf.float32)

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tf.TensorSpec(shape=(None, self.n_timesteps, X.shape[1]), dtype=tf.float32)
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset


    def _predict(self, dataset: tf.data.Dataset, total_len: int) -> np.ndarray:
        """
        Predicts event labels using the trained model.

        Parameters:
        - dataset: TensorFlow dataset object for model input.

        Returns:
        - y_pred: Predicted event labels.
        """
        steps_epoch = int(np.ceil((total_len - self.n_timesteps) / (self.stride * self.batch_size)))

        raw_preds = self.model.predict(dataset, steps=steps_epoch, verbose=1)[0].squeeze(-1)
        num_windows, win_len = raw_preds.shape
        output_len = num_windows * self.stride + win_len

        sum_preds = np.zeros(output_len)
        count_preds = np.zeros(output_len)

        for i in range(num_windows):
            start = i * self.stride
            end = start + win_len
            sum_preds[start:end] += raw_preds[i]
            count_preds[start:end] += 1

        avg_preds = np.divide(sum_preds, count_preds, where=count_preds != 0)
        y_pred = (avg_preds >= self.threshold).astype(int)

        return y_pred[:total_len]


    def run(self, raw_df: pd.DataFrame, nightside_filter: bool=False) -> np.ndarray:
        """
        Runs the event detection pipeline on raw data.

        Parameters:
        - raw_df: Raw input DataFrame.

        Returns:
        - y_pred: Predicted event labels.
        """
        if nightside_filter:
            df = self._nightside_filter(raw_df)
        else:
            df = raw_df
        
        sequences = self._segment_sequences(df)
        bx, by, bz, bx_vol, by_vol, bz_vol, bx_std, by_std, bz_std = self._compute_features(df, sequences)
        if self.return_df_features:
            X, total_len, df_features = self._prepare_model_input(df, bx, by, bz, bx_vol, by_vol, bz_vol, bx_std, by_std, bz_std)
        else:
            X, total_len = self._prepare_model_input(df, bx, by, bz, bx_vol, by_vol, bz_vol, bx_std, by_std, bz_std)

        dataset = self._create_dataset(X)
        y_pred = self._predict(dataset, total_len)

        if self.return_df_features:
            return y_pred, df_features
        else:
            return y_pred


raw_df = pd.DataFrame()
detector = DetectEvents(model_path="models/mosrl_80_all_model.keras", return_df_features=True)
y_pred = detector.run(raw_df, nightside_filter=False)
