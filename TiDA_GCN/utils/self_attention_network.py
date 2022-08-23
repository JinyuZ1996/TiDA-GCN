# This code was referred from Transformer.
# coding: utf-8

from __future__ import print_function
import tensorflow as tf
import numpy as np


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos / np.power(10000, 2 * i / dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True,
              l2_reg=0.0, scope="embedding", with_t=False, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        initializer = tf.contrib.layers.xavier_initializer()
        lookup_table = tf.Variable(initializer([vocab_size, num_units]), name='lookup_table')
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0, is_training=True, causality=False,
                        scope="multihead_attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)    (128*2,23,400)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding  作用：禁止Qi和 Kj(j>i)之间所有的连接；
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)    (23,23)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)  (2*128,23,23)
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries
    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

    return outputs
