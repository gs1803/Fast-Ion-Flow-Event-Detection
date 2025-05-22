import tensorflow as tf
import keras
from keras import losses

@tf.keras.utils.register_keras_serializable(package='Custom', name='TverskyBCEPerSequence')
class TverskyBCEPerSequence(losses.Loss):
    """
    Custom loss function combining Tversky loss and binary cross-entropy (BCE) with focal loss.
    It calculates the loss per sequence, applies optional weighting for sequences with events,
    and handles class imbalance using the focal loss formulation.
    """

    def __init__(self, alpha_t: float = 0.5, beta_t: float = 0.5, alpha_f: float = 0.5, gamma_f: float = 0.0,
                 event_weight: float = 1.0, smooth: float = 1e-6,
                 reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name: str = "tversky_bce_per_sequence") -> None:
        """
        Initializes the loss function with Tversky and focal loss parameters.

        Parameters:
            alpha_t: Tversky loss parameter penalizing false positives.
            beta_t: Tversky loss parameter penalizing false negatives.
            alpha_f: Focal loss alpha (class balancing).
            gamma_f: Focal loss gamma (focusing parameter).
            event_weight: Weight multiplier for sequences that contain events.
            smooth: Smoothing term to avoid division by zero.
            reduction: Loss reduction method.
            name: Name of the custom loss.
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
        Compute the custom per-sequence loss combining Tversky and focal BCE.

        Args:
            y_true: Ground truth labels (shape: batch_size, sequence_length).
            y_pred: Predicted probabilities (same shape as y_true).

        Returns:
            Tensor of shape (batch_size,) representing the loss per sequence.
        """
        # Flatten sequence dimension, keeping batch size
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

        # Ensure predictions are within a valid probability range
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)

        # Identify which sequences contain at least one positive label (event)
        has_event = tf.cast(tf.reduce_sum(y_true, axis=1) > 0, tf.float32)

        # Compute true positives, false negatives, and false positives per sequence
        tp = tf.reduce_sum(y_true * y_pred, axis=1)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=1)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=1)

        # Tversky index per sequence
        tversky = (tp + self.smooth) / (tp + self.alpha_t * fp + self.beta_t * fn + self.smooth)

        # Focal binary cross-entropy loss per sequence
        fbce = losses.binary_focal_crossentropy(y_true, y_pred, alpha=self.alpha_f, gamma=self.gamma_f)

        # Combine losses: scale 1 - Tversky index for sequences with events, use BCE for others
        final_loss = has_event * self.event_weight * (1 - tversky) + (1 - has_event) * fbce

        return final_loss


    def get_config(self) -> dict:
        """
        Return the configuration dictionary for serialization.

        Returns:
            A dictionary of the hyperparameters.
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