from tensorflow.keras import layers
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, rate=0.1, trainable=True, **kwargs):

        super(TransformerBlock, self).__init__(trainable=trainable, **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.rate = rate

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.key_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-9)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-9)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

        super(TransformerBlock, self).build(input_shape)



    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)