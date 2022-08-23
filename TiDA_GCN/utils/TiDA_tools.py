# @Author: Jinyu Zhang
# @Time: 2021/11/8 12:51
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.sparse as sp



def get_cos_distance(X1, X2):
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
    X1_X2 = tf.reduce_sum(tf.multiply(X1, X2), axis=1)
    X1_X2_norm = tf.multiply(X1_norm, X2_norm)
    cos = X1_X2 / X1_X2_norm
    return cos

def normalize_laplace_matrix(matrix_input):
    row_sum = np.array(matrix_input.sum(1))
    d_inv = np.power(row_sum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(matrix_input)
    return norm_adj.tocoo()


def dumplicate_matrix(matrix_in):
    frame_temp = pd.DataFrame(matrix_in, columns=['row', 'column'])
    frame_temp.duplicated()
    frame_temp.drop_duplicates(inplace=True)

    return frame_temp.values.tolist()
