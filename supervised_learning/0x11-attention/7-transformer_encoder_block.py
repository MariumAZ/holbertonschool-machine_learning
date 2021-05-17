#!/usr/bin/env python3

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, dm, h, hidden, rate=0.1):
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dense_hidden(attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = self.layernorm2(attn_output + ffn_output)  # (batch_size, input_seq_len, d_model)
        return ffn_output
