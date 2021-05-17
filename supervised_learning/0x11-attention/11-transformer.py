#!/usr/bin/env python3

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock

class Decoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()

        self.dm = dm
        self.N = N
        
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        seq_len = x.shape.as_list()[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.convert_to_tensor(self.positional_encoding[:seq_len].reshape((1, seq_len, -1)), dtype=tf.float32)
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training,
                               look_ahead_mask, padding_mask)
        return x
