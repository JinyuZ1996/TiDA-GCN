#!/usr/bin/env python
# coding: utf-8


import os
import random
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from DA_GCN.DA_Settings import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

args = Settings()

class DA_GCN:
    def __init__(self, n_items_A, n_items_B, n_users, A_in, all_h_list, all_t_list, all_v_list):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.n_items_A = n_items_A
        self.n_items_B = n_items_B
        self.n_users = n_users
        self.A_in = A_in
        self.all_h_list = all_h_list
        self.all_t_list = all_t_list
        self.all_v_list = all_v_list
        self.embedding_size = args.embedding_size
        self.n_member = args.n_member
        self.n_layers = args.n_layers
        self.n_fold = args.n_fold
        self.layer_size = args.layer_size
        self.weight_size = eval(self.layer_size)
        self.is_training = True
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.uid, self.seq_A, self.seq_B, self.target_A, self.target_B, self.learning_rate, \
                self.dropout_rate, self.keep_prob = self.get_inputs()

            with tf.name_scope('encoder'):
                self.all_weights = self._init_weights()
                self.i_embeddings_A, self.u_embeddings, self.i_embeddings_B = \
                    self._create_ngcf_embed(self.n_items_A, self.n_users, self.n_items_B, self.A_in,
                                            self.n_member, self.embedding_size)
                self.seq_emb_A_output, self.seq_emb_B_output = self.encoder(self.uid, self.seq_A, self.seq_B,
                                                                            self.dropout_rate, self.i_embeddings_A,
                                                                            self.u_embeddings, self.i_embeddings_B)
                self.h, self.t, self.A_kg_score = self._generate_transE_score()
                self.A_values, self.A_out = self._create_attentive_A_out()

            with tf.name_scope('prediction_A'):
                self.pred_A = self.prediction_A(self.n_items_A, self.seq_emb_B_output, self.seq_emb_A_output,
                                                self.keep_prob)
            with tf.name_scope('prediction_B'):
                self.pred_B = self.prediction_B(self.n_items_B, self.seq_emb_A_output, self.seq_emb_B_output,
                                                self.keep_prob)
            with tf.name_scope('loss'):
                self.loss, self.target_A, self.target_B, self.loss1, self.loss2 = self.cal_loss(self.target_A,
                                                                                                self.pred_A,
                                                                                                self.target_B,
                                                                                                self.pred_B)
            with tf.name_scope('optimizer'):
                self.train_op = self.optimizer(self.loss, self.learning_rate)

    def get_inputs(self):
        uid = tf.placeholder(dtype=tf.int32, shape=[None, ], name='uid')
        seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_A')
        seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_B')
        target_A = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_A')
        target_B = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_B')
        learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
        dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
        keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        return uid, seq_A, seq_B, target_A, target_B, learning_rate, dropout_rate, keep_prob

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.embedding_size * self.n_member]),
                                                    name='user_embedding')
        all_weights['item_embedding_A'] = tf.Variable(
            initializer([self.n_items_A, self.embedding_size * self.n_member]),
            name='item_embedding_A')
        all_weights['item_embedding_B'] = tf.Variable(
            initializer([self.n_items_B, self.embedding_size * self.n_member]),
            name='item_embedding_B')
        self.layers_plus = [self.embedding_size] + self.weight_size
        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.get_variable(
                'W_gc_%d' % k, [self.layers_plus[k] * self.n_member, self.layers_plus[k + 1] * self.n_member],
                tf.float32, initializer)
            all_weights['b_gc_%d' % k] = tf.get_variable(
                'b_gc_%d' % k, [self.layers_plus[k + 1] * self.n_member], tf.float32, tf.zeros_initializer())
            all_weights['W_bi_%d' % k] = tf.get_variable(
                'W_bi_%d' % k, [self.layers_plus[k] * self.n_member, self.layers_plus[k + 1] * self.n_member],
                tf.float32, initializer)
            all_weights['b_bi_%d' % k] = tf.get_variable(
                'b_bi_%d' % k, [self.layers_plus[k + 1] * self.n_member], tf.float32, tf.zeros_initializer())
        return all_weights

    def _generate_transE_score(self):
        embeddings = tf.concat([self.all_weights['item_embedding_A'], self.all_weights['user_embedding'],
                                self.all_weights['item_embedding_B']], axis=0)
        h = tf.placeholder(tf.int32, shape=[None], name='h')
        t = tf.placeholder(tf.int32, shape=[None], name='t')
        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)

        kg_score = self.get_cos_distance(h_e, t_e)
        return h, t, kg_score

    def _create_attentive_A_out(self):
        A_values = tf.placeholder(tf.float32, shape=[len(self.all_v_list)], name='A_values')
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        A = tf.sparse.softmax(tf.SparseTensor(indices, A_values, self.A_in.shape))
        return A_values, A

    def get_cos_distance(self, X1, X2):
        X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
        X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
        X1_X2 = tf.reduce_sum(tf.multiply(X1, X2), axis=1)
        X1_X2_norm = tf.multiply(X1_norm, X2_norm)
        cos = X1_X2 / X1_X2_norm
        return cos

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (X.shape[0]) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = X.shape[0]
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_ngcf_embed(self, n_items_A, n_users, n_items_B, A_in, n_member, embedding_size):
        A_fold_hat = self._split_A_hat(A_in)

        ego_embeddings = tf.concat([self.all_weights['item_embedding_A'], self.all_weights['user_embedding'],
                                    self.all_weights['item_embedding_B']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(args.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            sum_embeddings = tf.nn.leaky_relu(tf.matmul(side_embeddings, self.all_weights['W_gc_%d' % k])
                                              + self.all_weights['b_gc_%d' % k])

            bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
            bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, self.all_weights['W_bi_%d' % k])
                                             + self.all_weights['b_bi_%d' % k])

            ego_embeddings = sum_embeddings + bi_embeddings

            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.dropout_rate)

            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        g_embeddings_A, u_g_embeddings, g_embebddings_B = [], [], []
        for i in range(self.n_layers + 1):
            i_g_embeddings_A, u_g_embeddings_i, i_g_embeddings_B = tf.split(all_embeddings[i],
                                                                            [n_items_A, n_users, n_items_B],
                                                                            0)

            i_g_embeddings_A = tf.reshape(i_g_embeddings_A, [n_items_A, n_member, embedding_size])
            u_g_embeddings_i = tf.reshape(u_g_embeddings_i, [n_users, n_member, embedding_size])
            i_g_embeddings_B = tf.reshape(i_g_embeddings_B, [n_items_B, n_member, embedding_size])

            g_embeddings_A += [i_g_embeddings_A]
            u_g_embeddings += [u_g_embeddings_i]
            g_embebddings_B += [i_g_embeddings_B]

        i_g_embeddings_A = tf.concat(g_embeddings_A, -1)
        u_g_embeddings = tf.concat(u_g_embeddings, -1)
        i_g_embeddings_B = tf.concat(g_embebddings_B, -1)

        i_g_embeddings_A = tf.reduce_mean(i_g_embeddings_A, axis=1)
        u_g_embeddings = tf.reduce_mean(u_g_embeddings, axis=1)
        i_g_embeddings_B = tf.reduce_mean(i_g_embeddings_B, axis=1)
        print(i_g_embeddings_A)
        print(u_g_embeddings)
        print(i_g_embeddings_B)
        return i_g_embeddings_A, u_g_embeddings, i_g_embeddings_B

    def encoder(self, uid, seq_A, seq_B, dropout_rate, i_embeddings_A, u_embeddings, i_embeddings_B):
        with tf.variable_scope('encoder_A'):
            seq_emb_A_output = tf.nn.embedding_lookup(i_embeddings_A, seq_A)
            seq_emb_user_state = tf.nn.embedding_lookup(u_embeddings, uid)
            seq_embed_A_state = tf.reduce_max((seq_emb_A_output), 1)
            seq_emb_A_output = tf.concat([seq_embed_A_state, seq_emb_user_state], axis=1)
            seq_emb_A_output = tf.layers.dropout(seq_emb_A_output, rate=dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))

            seq_emb_B_output = tf.nn.embedding_lookup(i_embeddings_B, seq_B)
            seq_embed_B_state = tf.reduce_max((seq_emb_B_output), 1)
            seq_emb_B_output = tf.concat([seq_embed_B_state, seq_emb_user_state], axis=1)
            seq_emb_B_output = tf.layers.dropout(seq_emb_B_output, rate=dropout_rate,
                                                 training=tf.convert_to_tensor(self.is_training))
            print(seq_emb_A_output)
            print(seq_emb_B_output)

        return seq_emb_A_output, seq_emb_B_output

    def prediction_A(self, n_items_A, seq_emb_B_output, seq_emb_A_output, keep_prob):
        with tf.variable_scope('prediction_A'):
            concat_output = tf.concat([seq_emb_B_output, seq_emb_A_output], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output,
                                          keep_prob)
            pred_A = tf.layers.dense(concat_output, n_items_A, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(
                                         uniform=False))
            print(pred_A)

            return pred_A

    def prediction_B(self, n_items_B, seq_emb_A_output, seq_emb_B_output, keep_prob):

        with tf.variable_scope('prediction_B'):
            concat_output = tf.concat([seq_emb_A_output, seq_emb_B_output], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output,
                                          keep_prob)
            pred_B = tf.layers.dense(concat_output, n_items_B, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(
                                         uniform=False))
            print(pred_B)

            return pred_B

    def cal_loss(self, target_A, pred_A, target_B, pred_B):
        loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_A,
                                                                logits=pred_A)
        loss_A = tf.reduce_mean(loss_1, name='loss_A')
        loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_B, logits=pred_B)
        loss_B = tf.reduce_mean(loss_2, name='loss_B')
        loss = loss_A + loss_B
        return loss, target_A, target_B, pred_A, pred_B

    def optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                            grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op

    def _si_norm_lap(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def update_attentive_A(self, sess):
        fold_len = len(self.all_h_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            feed_dict = {
                self.h: self.all_h_list[start:end],
                self.t: self.all_t_list[start:end]
            }
            A_kg_score = sess.run(self.A_kg_score, feed_dict=feed_dict)
            kg_score += list(A_kg_score)

        kg_score = np.array(kg_score)
        new_A = sess.run(self.A_out, feed_dict={self.A_values: kg_score})
        new_A_values = new_A.values
        new_A_indices = new_A.indices

        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_items_A + self.n_users + self.n_items_B,
                                                                       self.n_items_A + self.n_users + self.n_items_B))

    def train_GCN(self, sess, uid, seq_A, seq_B, target_A, target_B, learning_rate,
                  dropout_rate, keep_prob):

        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B,
                     self.target_A: target_A, self.target_B: target_B, self.learning_rate: learning_rate,
                     self.dropout_rate: dropout_rate, self.keep_prob: keep_prob}

        return sess.run([self.loss, self.train_op], feed_dict)

    def evaluate_GCN(self, sess, uid, seq_A, seq_B, target_A, target_B, learning_rate, dropout_rate, keep_prob):
        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B,
                     self.target_A: target_A, self.target_B: target_B,
                     self.learning_rate: learning_rate, self.dropout_rate: dropout_rate,
                     self.keep_prob: keep_prob}
        return sess.run([self.pred_A, self.pred_B], feed_dict)