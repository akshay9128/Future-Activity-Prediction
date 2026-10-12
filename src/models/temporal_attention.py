"""
Temporal Attention Mechanism Module.

Implements a custom Keras Layer that computes temporal attention weights over sequential
hidden representations (e.g. from an LSTM), allowing the network to dynamically weight
the most discriminative frames in the observed video segment.
"""

from typing import Any, Dict, Optional, Tuple, Union
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="CustomLayers", name="TemporalAttention")
class TemporalAttention(tf.keras.layers.Layer):
    """
    Temporal Attention layer for sequential recurrent representations.

    Given an input tensor H of shape (batch_size, sequence_length, hidden_dim):
    1. Computes an attention energy score for each time step:
       u_t = tanh(H_t * W + b)
       e_t = u_t * v
    2. Computes normalized temporal attention weights via softmax across time steps:
       alpha_t = exp(e_t) / sum_k(exp(e_k))
    3. Produces a weighted context vector:
       context = sum_t(alpha_t * H_t)

    Attributes:
        units: Dimensionality of the attention projection space.
        return_attention_scores: If True, call() returns (context_vector, attention_weights).
                                 If False, call() returns only context_vector.
    """

    def __init__(
        self,
        units: int = 64,
        return_attention_scores: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.return_attention_scores = return_attention_scores

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the trainable weight parameters for the attention mechanism.
        """
        hidden_dim = int(input_shape[-1])
        if hidden_dim is None:
            raise ValueError(f"Last dimension of input_shape must be defined, got {input_shape}")

        # Projection weight matrix W: (hidden_dim, units)
        self.W = self.add_weight(
            name="attention_weight",
            shape=(hidden_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Projection bias vector b: (units,)
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        # Context query vector v: (units, 1)
        self.v = self.add_weight(
            name="attention_query_vector",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        return_attention_scores: Optional[bool] = None,
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Forward computation for temporal attention pooling.

        Args:
            inputs: Tensor of shape (batch_size, sequence_length, hidden_dim).
            mask: Optional binary mask of shape (batch_size, sequence_length).
            return_attention_scores: Override for self.return_attention_scores.

        Returns:
            tf.Tensor or Tuple[tf.Tensor, tf.Tensor]:
                - context_vector: Shape (batch_size, hidden_dim)
                - attention_weights: Shape (batch_size, sequence_length, 1) (if requested)
        """
        if return_attention_scores is None:
            return_attention_scores = self.return_attention_scores

        # 1. Project input sequence to attention energy space: (B, T, units)
        # inputs: (B, T, D), W: (D, units) -> (B, T, units)
        u = tf.tanh(tf.matmul(inputs, self.W) + self.b)

        # 2. Score each time step against query vector: (B, T, units) @ (units, 1) -> (B, T, 1)
        scores = tf.matmul(u, self.v)

        # Apply mask if provided
        if mask is not None:
            mask = tf.cast(mask, tf.bool)
            mask = tf.expand_dims(mask, axis=-1)  # (B, T, 1)
            scores = tf.where(mask, scores, tf.constant(-1e9, dtype=scores.dtype))

        # 3. Softmax across time dimension (axis=1) to get normalized attention weights: (B, T, 1)
        attention_weights = tf.nn.softmax(scores, axis=1)

        # 4. Compute weighted context vector across time dimension: (B, D)
        # Broadcast multiply (B, T, D) * (B, T, 1) -> (B, T, D), then reduce_sum over T -> (B, D)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)

        if return_attention_scores:
            return context_vector, attention_weights
        return context_vector

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], Tuple[Optional[int], ...]]]:
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        hidden_dim = input_shape[2]

        context_shape = (batch_size, hidden_dim)
        weights_shape = (batch_size, seq_len, 1)

        if self.return_attention_scores:
            return (context_shape, weights_shape)
        return context_shape

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration dictionary for serialization."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "return_attention_scores": self.return_attention_scores,
            }
        )
        return config
