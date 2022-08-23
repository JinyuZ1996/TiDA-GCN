# @Author: Jinyu Zhang
# @Time: 2021/11/3 12:28
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

import os
import random
from TiDA_GCN.utils.self_attention_network import embedding, multihead_attention, normalize, feedforward
from TiDA_GCN.TiDA_Settings import *
from utils.TiDA_tools import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

args = Settings()


def get_inputs():
    uid = tf.placeholder(dtype=tf.int32, shape=[None, ], name='uid')
    seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_A')
    seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_B')

    target_A = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_A')
    target_B = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_B')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    dropout_rate = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')
    drop_A = tf.placeholder(dtype=tf.float32, shape=[], name='drop_A')
    drop_B = tf.placeholder(dtype=tf.float32, shape=[], name='drop_B')
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')

    return uid, seq_A, seq_B, target_A, target_B, learning_rate, dropout_rate, drop_A, drop_B, keep_prob


class GCN_Module:
    def __init__(self, n_items_A, n_items_B, n_users, maxlen_A, maxlen_B, matrix_AUB_info, time_matrix, num_relations,
                 processed_row_list,
                 processed_column_list, processed_value_list):

        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.time_matrix = time_matrix
        self.num_relations = num_relations

        self.n_items_A = n_items_A
        self.n_items_B = n_items_B
        self.n_users = n_users
        self.maxlen_A = maxlen_A
        self.maxlen_B = maxlen_B

        self.matrix_AUB_info = matrix_AUB_info

        self.processed_row_list = processed_row_list
        self.processed_column_list = processed_column_list
        self.processed_value_list = processed_value_list

        # hyper params for GCN_net

        self.embedding_size = args.embedding_size
        self.hidden_units = args.hidden_units
        self.dff = args.dff
        self.weight_size = args.weight_size
        self.n_member = args.n_member
        self.n_layers = args.n_layers
        self.n_fold = args.n_fold
        self.layer_size = args.layer_size
        self.num_blocks_A = args.num_blocks_A
        self.num_blocks_B = args.num_blocks_B
        self.num_heads = args.num_heads
        self.l2_emb = args.l2_emb
        self.Beta = args.Beta
        self.Alpha = args.Alpha

        self.w_size = eval(self.layer_size)
        self.is_training = True

        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope('inputs'):
                self.uid, self.seq_A, self.seq_B, self.target_A, self.target_B, \
                self.learning_rate, self.dropout_rate, self.drop_A, self.drop_B, self.keep_prob = get_inputs()

            with tf.name_scope('encoder'):
                self.all_weights = self.init_weights()
                self.item_ebd_A, self.user_ebd, self.item_ebd_B = \
                    self.create_ngcf_embedding(self.n_items_A, self.n_users, self.n_items_B, self.matrix_AUB_info,
                                               self.n_member,
                                               self.embedding_size)
                self.seq_emb_A_output, self.seq_emb_B_output, self.seq_emb_A_output_1, self.seq_emb_B_output_1 = \
                    self.encoder(self.uid, self.seq_A, self.seq_B, self.item_ebd_A, self.user_ebd,
                                 self.item_ebd_B)

                self.h, self.t, self.A_kg_score = self.get_transE_score()
                self.A_values, self.A_out = self.get_attentive_A()

            with tf.name_scope('prediction_A'):
                self.pred_A = self.prediction_A(self.n_items_A, self.seq_emb_B_output, self.seq_emb_A_output,
                                                self.keep_prob)
            with tf.name_scope('prediction_B'):
                self.pred_B = self.prediction_B(self.n_items_B, self.seq_emb_A_output, self.seq_emb_B_output,
                                                self.keep_prob)
            with tf.name_scope('loss'):
                self.loss = self.cal_loss(self.target_A, self.pred_A, self.target_B, self.pred_B)

            with tf.name_scope('optimizer'):
                self.train_op = self.optimizer(self.loss, self.learning_rate)

    def init_weights(self):
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

        self.layers_plus = [self.embedding_size] + self.w_size

        all_weights['padding_embedding'] = tf.Variable(tf.zeros([1, self.n_member * self.embedding_size],
                                                                name='padding_embedding'))
        all_weights['relation_embed'] = tf.Variable(
            initializer([self.num_relations, self.embedding_size * self.n_member]), name='relation_embed')

        all_weights['W_f'] = tf.Variable(
            initializer([self.embedding_size * self.n_member, self.embedding_size * self.n_member]),
            name='W_f')

        all_weights['W_1'] = tf.Variable(initializer([self.embedding_size, self.weight_size]), name='W_1')
        all_weights['b_1'] = tf.Variable(initializer([1, self.weight_size]), name='b_1')
        all_weights['h_1'] = tf.Variable(tf.ones([self.weight_size, 1]), name='h_1')

        all_weights['W_2'] = tf.Variable(initializer([self.embedding_size, self.weight_size]), name='W_2')
        all_weights['b_2'] = tf.Variable(initializer([1, self.weight_size]), name='b_2')
        all_weights['h_2'] = tf.Variable(tf.ones([self.weight_size, 1]), name='h_2')

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
            all_weights['W_ti_%d' % k] = tf.get_variable(
                'W_ti_%d' % k, [self.layers_plus[k] * self.n_member, self.layers_plus[k + 1] * self.n_member],
                tf.float32, initializer)
            all_weights['b_ti_%d' % k] = tf.get_variable(
                'b_ti_%d' % k, [self.layers_plus[k + 1] * self.n_member], tf.float32, tf.zeros_initializer())
        return all_weights

    def get_transE_score(self):
        embeddings = tf.concat([self.all_weights['item_embedding_A'], self.all_weights['user_embedding'],
                                self.all_weights['item_embedding_B']], axis=0)
        h = tf.placeholder(tf.int32, shape=[None], name='h')
        t = tf.placeholder(tf.int32, shape=[None], name='t')
        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)
        kg_score = get_cos_distance(h_e, t_e)
        return h, t, kg_score

    def get_attentive_A(self):
        A_values = tf.placeholder(tf.float32, shape=[len(self.processed_value_list)], name='A_values')
        indices = np.mat([self.processed_row_list, self.processed_column_list]).transpose()
        A = tf.sparse.softmax(tf.SparseTensor(indices, A_values, self.matrix_AUB_info.shape))
        return A_values, A

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

    def create_ngcf_embedding(self, n_items_A, n_users, n_items_B, matrix_AUB_info, n_member, embedding_size):
        A_fold_hat = self._split_A_hat(matrix_AUB_info)
        B_fold_hat = self._convert_sp_mat_to_sp_tensor(self.time_matrix)

        ego_embeddings = tf.concat([self.all_weights['item_embedding_A'], self.all_weights['user_embedding'],
                                    self.all_weights['item_embedding_B']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            temp_ebd = []
            T_embeddings = tf.sparse_tensor_dense_matmul(B_fold_hat, self.all_weights['relation_embed'])
            for f in range(args.n_fold):
                temp_ebd.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
            side_embeddings = tf.concat(temp_ebd, 0)
            mix_embeddings = self.Alpha * side_embeddings + (1 - self.Alpha) * T_embeddings
            sum_embeddings = tf.nn.leaky_relu(tf.matmul(mix_embeddings, self.all_weights['W_gc_%d' % k])
                                              + self.all_weights['b_gc_%d' % k])

            bi_embeddings = tf.multiply(ego_embeddings, mix_embeddings)
            bi_embeddings = tf.nn.leaky_relu(tf.matmul(bi_embeddings, self.all_weights['W_bi_%d' % k])
                                             + self.all_weights['b_bi_%d' % k])

            ego_embeddings = sum_embeddings + bi_embeddings

            ego_embeddings = tf.nn.dropout(ego_embeddings, 1 - self.dropout_rate)

            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)

            all_embeddings += [norm_embeddings]

        temp_ebd_A, temp_ebd_U, temp_ebd_B = [], [], []
        for i in range(self.n_layers + 1):
            split_ebd_A, split_ebd_User, split_ebd_B \
                = tf.split(all_embeddings[i], [n_items_A, n_users, n_items_B], 0)

            temp_ebd_A += [split_ebd_A]
            temp_ebd_U += [split_ebd_User]
            temp_ebd_B += [split_ebd_B]

        item_graph_ebd_A = tf.stack(temp_ebd_A, 1)
        user_graph_ebd = tf.stack(temp_ebd_U, 1)
        item_graph_ebd_B = tf.stack(temp_ebd_B, 1)

        item_graph_ebd_A = tf.reduce_mean(item_graph_ebd_A, axis=1)
        user_graph_ebd = tf.reduce_mean(user_graph_ebd, axis=1)
        item_graph_ebd_B = tf.reduce_mean(item_graph_ebd_B, axis=1)

        item_graph_ebd_A = tf.concat([item_graph_ebd_A, self.all_weights['padding_embedding']], axis=0)
        item_graph_ebd_B = tf.concat([item_graph_ebd_B, self.all_weights['padding_embedding']], axis=0)

        print(item_graph_ebd_A)
        print(user_graph_ebd)
        print(item_graph_ebd_B)
        return item_graph_ebd_A, user_graph_ebd, item_graph_ebd_B

    def encoder(self, uid, seq_A, seq_B, i_embeddings_A, u_embeddings, i_embeddings_B):
        seq_emb_user_state = tf.nn.embedding_lookup(u_embeddings, uid)
        seq_emb_user_state = tf.expand_dims(seq_emb_user_state, 1)
        with tf.variable_scope('SA_A', reuse=tf.AUTO_REUSE):
            masks_A = tf.expand_dims(tf.to_float(tf.not_equal(seq_A, self.n_items_A)), -1)
            seq_emb_A = tf.nn.embedding_lookup(i_embeddings_A, seq_A)

            seq_emb_A_output_1 = seq_emb_A

            seq_emb_user_A = tf.tile(seq_emb_user_state,
                                     [1, tf.shape(seq_emb_A)[1], 1])

            t_A = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(seq_A)[1]), 0), [tf.shape(self.seq_A)[0], 1]),
                vocab_size=self.maxlen_A,
                num_units=self.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=self.l2_emb,
                scope="P_A",
                reuse=None,
                with_t=False
            )

            seq_emb_A_output_1 += t_A
            seq_emb_A_output_1 = tf.layers.dropout(seq_emb_A_output_1, rate=self.drop_A,
                                                   training=tf.convert_to_tensor(self.is_training))
            seq_emb_A_output_1 *= masks_A
            seq_emb_user_A *= masks_A

            for m in range(self.num_blocks_A):  # 0,1,2,3
                with tf.variable_scope("num_blocks_A_%d" % m):
                    seq_emb_A_output_1 = multihead_attention(queries=normalize(seq_emb_user_A),
                                                             keys=seq_emb_A_output_1,
                                                             num_units=self.hidden_units,
                                                             num_heads=self.num_heads,
                                                             dropout_rate=self.drop_A,
                                                             is_training=self.is_training,
                                                             causality=True,
                                                             scope="self_attention_A"
                                                             )
                    seq_emb_A_output_1 = feedforward(normalize(seq_emb_A_output_1),
                                                     num_units=[self.dff, self.hidden_units],
                                                     scope="positionwise_feedforward_A")
            seq_emb_A_output_1 = normalize(seq_emb_A_output_1)

            seq_emb_A1 = normalize(seq_emb_A)

            seq_emb_A1 = tf.reshape(seq_emb_A1,
                                    [tf.shape(seq_emb_A1)[0], tf.shape(seq_emb_A1)[1], self.n_member,
                                     self.embedding_size])
            seq_emb_A1 = tf.reduce_mean(seq_emb_A1, axis=2)
            seq_embed_A = tf.reduce_sum(seq_emb_A1, 1)

            seq_emb_A_output_1 = tf.reshape(seq_emb_A_output_1,
                                            [tf.shape(seq_emb_A_output_1)[0], tf.shape(seq_emb_A_output_1)[1],
                                             self.n_member, self.embedding_size])
            self.seq_emb_A_output_i = tf.reduce_mean(seq_emb_A_output_1, axis=2)

            last_position_A = tf.reshape(tf.reduce_max(masks_A, axis=1), [tf.shape(masks_A)[0]])

            self.last_id = tf.gather_nd(self.seq_emb_A_output_i,
                                        tf.stack([tf.range(tf.shape(self.seq_emb_A_output_i)[0]),
                                                  tf.to_int32(last_position_A) - 1], axis=1))

            last_item_A = tf.expand_dims(self.last_id, 1)
            self.seq_emb_A_output_u = tf.reduce_sum((seq_emb_A_output_1), 1)

            self.e = self.seq_emb_A_output_u * last_item_A

            self.seq_embed_A_state_1 = self.attention_MLP_A(normalize(self.e))

            self.seq_embed_A_state_1 = normalize(self.seq_embed_A_state_1)
            seq_embed_A_state = 0.9 * seq_embed_A + 0.1 * self.seq_embed_A_state_1

            seq_emb_user_A = normalize(seq_emb_user_A)
            seq_emb_userA_output = tf.reshape(seq_emb_user_A,
                                              [tf.shape(seq_emb_A_output_1)[0], tf.shape(seq_emb_A_output_1)[1],
                                               self.n_member, self.embedding_size])
            seq_emb_userA_output = tf.reduce_mean(seq_emb_userA_output, axis=2)
            seq_emb_userA_output = tf.reduce_sum((seq_emb_userA_output), 1)

            seq_emb_A_output = tf.concat([seq_embed_A_state, seq_emb_userA_output], axis=1)
            seq_emb_A_output = tf.layers.dropout(seq_emb_A_output, rate=self.drop_A,
                                                 training=tf.convert_to_tensor(self.is_training))

        with tf.variable_scope('SA_B', reuse=tf.AUTO_REUSE):

            masks_B = tf.expand_dims(tf.to_float(tf.not_equal(seq_B, self.n_items_B)), -1)
            seq_emb_B = tf.nn.embedding_lookup(i_embeddings_B, seq_B)
            seq_emb_B_output_1 = seq_emb_B

            seq_emb_user_B = tf.tile(seq_emb_user_state,
                                     [1, tf.shape(seq_emb_B)[1], 1])

            t_B = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(seq_B)[1]), 0), [tf.shape(self.seq_B)[0], 1]),
                vocab_size=self.maxlen_B,
                num_units=self.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=self.l2_emb,
                scope="P_B",
                reuse=None,
                with_t=False
            )

            seq_emb_B_output_1 += t_B
            seq_emb_B_output_1 = tf.layers.dropout(seq_emb_B_output_1, rate=self.drop_B,
                                                   training=tf.convert_to_tensor(self.is_training))
            seq_emb_B_output_1 *= masks_B
            seq_emb_user_B *= masks_B

            for n in range(self.num_blocks_B):
                with tf.variable_scope("num_blocks_B_%d" % n):
                    seq_emb_B_output_1 = multihead_attention(
                        queries=normalize(seq_emb_user_B),
                        keys=seq_emb_B_output_1,
                        num_units=self.hidden_units,
                        num_heads=self.num_heads,
                        dropout_rate=self.drop_B,
                        is_training=self.is_training,
                        causality=True,
                        scope="self_attention_B"
                    )
                    seq_emb_B_output_1 = feedforward(normalize(seq_emb_B_output_1),
                                                     num_units=[self.dff, self.hidden_units],
                                                     scope="positionwise_feedforward_B")
            seq_emb_B_output_1 = normalize(seq_emb_B_output_1)

            seq_emb_B1 = normalize(seq_emb_B)
            seq_emb_B1 = tf.reshape(seq_emb_B1, [tf.shape(seq_emb_B1)[0], tf.shape(seq_emb_B1)[1], self.n_member,
                                                 self.embedding_size])
            seq_emb_B1 = tf.reduce_mean(seq_emb_B1, axis=2)
            seq_embed_B = tf.reduce_sum((seq_emb_B1), 1)

            seq_emb_B_output_1 = tf.reshape(seq_emb_B_output_1,
                                            [tf.shape(seq_emb_B_output_1)[0], tf.shape(seq_emb_B_output_1)[1],
                                             self.n_member, self.embedding_size])
            self.seq_emb_B_output_i = tf.reduce_mean(seq_emb_B_output_1, axis=2)

            last_position_B = tf.reshape(tf.reduce_sum(masks_B, axis=1), [tf.shape(masks_B)[0]])

            self.last_id_B = tf.gather_nd(self.seq_emb_B_output_i,
                                          tf.stack(
                                              [tf.range(tf.shape(self.seq_emb_B_output_i)[0]),
                                               tf.to_int32(last_position_B) - 1],
                                              axis=1))
            last_item_B = tf.expand_dims(self.last_id_B, 1)
            self.seq_emb_B_output_u = tf.reduce_sum((seq_emb_B_output_1), 1)
            self.e1 = self.seq_emb_B_output_u * last_item_B

            self.seq_embed_B_state_1 = self.attention_MLP_B(normalize(self.e1))
            self.seq_embed_B_state_1 = normalize(self.seq_embed_B_state_1)
            seq_embed_B_state = 0.9 * seq_embed_B + 0.1 * self.seq_embed_B_state_1

            seq_emb_user_B = normalize(seq_emb_user_B)
            seq_emb_userB_output = tf.reshape(seq_emb_user_B,
                                              [tf.shape(seq_emb_B_output_1)[0], tf.shape(seq_emb_B_output_1)[1],
                                               self.n_member, self.embedding_size])
            seq_emb_userB_output = tf.reduce_mean(seq_emb_userB_output, axis=2)
            seq_emb_userB_output = tf.reduce_sum((seq_emb_userB_output), 1)

            seq_emb_B_output = tf.concat([seq_embed_B_state, seq_emb_userB_output], axis=1)
            seq_emb_B_output = tf.layers.dropout(seq_emb_B_output, rate=self.drop_B,
                                                 training=tf.convert_to_tensor(self.is_training))

        print(seq_emb_A_output)
        print(seq_emb_B_output)
        print(seq_emb_A_output_1)
        print(seq_emb_B_output_1)

        return seq_emb_A_output, seq_emb_B_output, seq_emb_A_output_1, seq_emb_B_output_1

    def attention_MLP_A(self, p_):
        with tf.name_scope("attention_MLP_A"):
            b = tf.shape(p_)[0]
            n = tf.shape(p_)[1]
            r = self.embedding_size

            self.MLP_output_1 = tf.matmul(tf.reshape(p_, [-1, r]), self.all_weights['W_1']) + self.all_weights[
                'b_1']

            MLP_output = tf.nn.tanh(self.MLP_output_1)

            self.A_ = tf.reshape(tf.matmul(MLP_output, self.all_weights['h_1']),
                                 [b, n])

            self.exp_A_ = tf.exp(self.A_)
            self.exp_sum_1 = tf.reduce_sum(self.exp_A_, 1, keepdims=True)
            self.exp_sum = tf.pow(self.exp_sum_1, tf.constant(self.Beta, tf.float32, [1]))
            self.A = tf.expand_dims(tf.div(self.exp_A_, self.exp_sum), 2)

            return tf.reduce_sum(self.A * self.seq_emb_A_output_u, 1)

    def attention_MLP_B(self, q_):
        with tf.name_scope("attention_MLP_B"):
            b1 = tf.shape(q_)[0]
            n1 = tf.shape(q_)[1]
            r1 = self.embedding_size

            self.MLP_output_B_1 = tf.matmul(tf.reshape(q_, [-1, r1]), self.all_weights['W_2']) + self.all_weights[
                'b_2']

            self.MLP_output_B = tf.nn.tanh(self.MLP_output_B_1)

            self.B_ = tf.reshape(tf.matmul(self.MLP_output_B, self.all_weights['h_2']),
                                 [b1, n1])

            self.exp_B_ = tf.exp(self.B_)
            self.exp_sum_B_1 = tf.reduce_sum(self.exp_B_, 1, keepdims=True)
            self.exp_sum_B = tf.pow(self.exp_sum_B_1, tf.constant(self.Beta, tf.float32, [1]))
            self.B = tf.expand_dims(tf.div(self.exp_B_, self.exp_sum_B), 2)

            return tf.reduce_sum(self.B * self.seq_emb_B_output_u, 1)

    def prediction_A(self, n_items_A, seq_emb_B_output, seq_emb_A_output, keep_prob):
        with tf.variable_scope('prediction_A'):
            concat_output = tf.concat([seq_emb_B_output, seq_emb_A_output], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, keep_prob)
            pred_A = tf.layers.dense(concat_output, n_items_A, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(pred_A)
            return pred_A

    def prediction_B(self, n_items_B, seq_emb_A_output, seq_emb_B_output, keep_prob):
        with tf.variable_scope('prediction_B'):
            concat_output = tf.concat([seq_emb_A_output, seq_emb_B_output], axis=-1)
            print(concat_output)
            concat_output = tf.nn.dropout(concat_output, keep_prob)
            pred_B = tf.layers.dense(concat_output, n_items_B, activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            print(pred_B)
            return pred_B

    def cal_loss(self, target_A, pred_A, target_B, pred_B):
        cross_entropy_loss_A = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_A, logits=pred_A)
        loss_A = tf.reduce_mean(cross_entropy_loss_A, name='loss_A')
        cross_entropy_loss_B = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_B, logits=pred_B)
        loss_B = tf.reduce_mean(cross_entropy_loss_B, name='loss_B')
        loss = loss_A + loss_B
        return loss

    def optimizer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op

    def update_attentive_A(self, sess):
        fold_len = len(self.processed_row_list) // self.n_fold
        kg_score = []

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.processed_row_list)
            else:
                end = (i_fold + 1) * fold_len

            feed_dict = {
                self.h: self.processed_row_list[start:end],
                self.t: self.processed_column_list[start:end]
            }
            A_kg_score = sess.run(self.A_kg_score, feed_dict=feed_dict)
            kg_score += list(A_kg_score)

        kg_score = np.array(kg_score)

        new_A = sess.run(self.A_out, feed_dict={self.A_values: kg_score})
        new_A_values = new_A.values
        new_A_indices = new_A.indices

        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.matrix_AUB_info = sp.coo_matrix((new_A_values, (rows, cols)),
                                             shape=(self.n_items_A + self.n_users + self.n_items_B,
                                                    self.n_items_A + self.n_users + self.n_items_B))

    def train_GCN(self, sess, uid, seq_A, seq_B, target_A, target_B, learning_rate,
                  dropout_rate, drop_A, drop_B, keep_prob):

        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B,
                     self.target_A: target_A, self.target_B: target_B, self.learning_rate: learning_rate,
                     self.dropout_rate: dropout_rate, self.drop_A: drop_A, self.drop_B: drop_B,
                     self.keep_prob: keep_prob}

        return sess.run([self.loss, self.train_op], feed_dict)

    def evaluate_GCN(self, sess, uid, seq_A, seq_B, target_A, target_B, learning_rate, dropout_rate,
                     drop_A, drop_B, keep_prob):
        feed_dict = {self.uid: uid, self.seq_A: seq_A, self.seq_B: seq_B,
                     self.target_A: target_A, self.target_B: target_B,
                     self.learning_rate: learning_rate, self.dropout_rate: dropout_rate,
                     self.drop_A: drop_A, self.drop_B: drop_B,
                     self.keep_prob: keep_prob}
        return sess.run([self.pred_A, self.pred_B], feed_dict)
