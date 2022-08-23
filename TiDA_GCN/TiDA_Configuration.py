# @Author: Jinyu Zhang
# @Time: 2021/11/11 9:14
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

import os
import random
import collections
import math
import time
import itertools
from TiDA_GCN.utils.TiDA_tools import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

random.seed(1)
np.random.seed(1)

def load_dict(dict_path):
    itemdict = {}
    with open(dict_path, 'r') as file_object:
        items = file_object.readlines()
    for item in items:
        item = item.strip().split('\t')
        itemdict[item[1]] = int(item[0])
    return itemdict


def load_tdict(all_data_path, dict_A, dataset):
    tstamp_A = set()
    tstamp_B = set()
    with open(all_data_path, 'r') as file_object:
        lines = file_object.readlines()
        for line in lines:
            line = line.strip().split('\t')
            for item in line[1:]:
                item_info = item.split('|')
                item_id = item_info[0]
                if dataset is "Hvideo":
                    item_tstamp = float(item_info[3])
                else:
                    item_tstamp = float(item_info[1])
                if item_id in dict_A:
                    tstamp_A.add(item_tstamp)
                else:
                    tstamp_B.add(item_tstamp)
    tdict_A = generate_tdict(tstamp_A)
    tdict_B = generate_tdict(tstamp_B)
    return tdict_A, tdict_B


def generate_tdict(time_set):
    time_dict = dict()
    value = 0
    for time in time_set:
        if time in time_dict:
            continue
        else:
            time_dict[time] = value
            value += 1
    return time_dict


def get_data(data_path, dict_A, dict_B, dict_U, tdict_A, tdict_B, dataset):
    with open(data_path, 'r') as file_object:
        mixed_data = []
        lines = file_object.readlines()
        jump_time = 0
        for line in lines:
            sign_A, sign_B = 0, 0
            temp_sequence = []
            line = line.strip().split('\t')
            if len(line) <= 3:
                print("Jump the short line.")
                continue
            else:
                sequence_all = []
                tstamp_all = []
                user = line[0]
                sequence_all.append(dict_U[user])
                for item in line[1:]:
                    item_info = item.split('|')
                    item_id = item_info[0]
                    if dataset is "Hvideo":
                        item_tstamp = float(item_info[3])
                    else:
                        item_tstamp = float(item_info[1])
                    if item_id in dict_A:
                        sequence_all.append(dict_A[item_id])
                        tstamp_all.append(tdict_A[item_tstamp])
                        sign_A+=1
                    else:
                        sequence_all.append(dict_B[item_id] + len(dict_A))
                        tstamp_all.append(tdict_B[item_tstamp] + len(tdict_A))
                        sign_B+=1
                if sign_A<=1 or sign_B<=1:
                    jump_time+=1
                    print("Jump the short_sequence for {} times.".format(jump_time))
                    continue
                else:
                    temp_sequence.append(sequence_all)
                    temp_sequence.append(tstamp_all)
                    mixed_data.append(temp_sequence)
    return mixed_data


def process_data(mixed_data, dict_A, tdict_A):
    data_outputs = []
    maxlen_A, maxlen_B = 0, 0
    for index in mixed_data:
        temp = []
        seq_A, seq_B = [], []
        t_stamp_A, t_stamp_B = [], []
        len_A, len_B = 0, 0
        mixed_sequence = index[0]
        mixed_tstamp = index[1]
        seq_A.append(mixed_sequence[0])
        seq_B.append(mixed_sequence[0])
        for item_id in mixed_sequence[1:-2]:
            if item_id < len(dict_A):
                seq_A.append(item_id)
                len_A += 1
            else:
                seq_B.append(item_id - len(dict_A))
                len_B += 1
        for stamp_id in mixed_tstamp[0:-2]:
            if stamp_id < len(tdict_A):
                t_stamp_A.append(stamp_id)
            else:
                t_stamp_B.append(stamp_id - len(tdict_A))
        temp.append(seq_A)
        temp.append(seq_B)
        temp.append(t_stamp_A)
        temp.append(t_stamp_B)
        temp.append(len_A)
        temp.append(len_B)
        if maxlen_A <= len_A:
            maxlen_A = len_A
        if maxlen_B <= len_B:
            maxlen_B = len_B
        temp.append(mixed_sequence[-2])
        temp.append(mixed_sequence[-1] - len(dict_A))

        data_outputs.append(temp)
    return data_outputs, maxlen_A, maxlen_B


def generate_time_matrix(processed_data, dict_A, dict_B, dict_U):
    n_items_A = len(dict_A)
    n_items_B = len(dict_B)
    n_users = len(dict_U)

    time_matrix_A = sp.dok_matrix((n_items_A, n_items_A), dtype=np.int64)
    time_matrix_B = sp.dok_matrix((n_items_B, n_items_B), dtype=np.int64)
    time_location_x_A = []
    time_location_y_A = []
    time_location_x_B = []
    time_location_y_B = []

    start_init_matrix_locations = time.time()

    for index in processed_data:
        seq_A = index[0]
        seq_B = index[1]
        t_stamp_A = index[2]
        t_stamp_B = index[3]

        items_A = [int(i) for i in seq_A[1:]]
        items_B = [int(j) for j in seq_B[1:]]

        for row_A in range(0, len(items_A) - 1):
            column_A = row_A + 1
            spanA = math.ceil(abs(t_stamp_A[row_A] - t_stamp_A[column_A]) / 86400)
            temp_location_x = items_A[row_A]
            temp_location_y = items_A[column_A]
            time_matrix_A[temp_location_x, temp_location_y] += spanA
            time_location_x_A.append(temp_location_x)
            time_location_y_A.append(temp_location_y)

        for row_B in range(0, len(items_B) - 1):
            column_B = row_B + 1
            spanB = math.ceil(abs(t_stamp_B[row_B] - t_stamp_B[column_B]) / 86400)
            temp_location_x = items_B[row_B]
            temp_location_y = items_B[column_B]
            time_matrix_B[temp_location_x, temp_location_y] += spanB
            time_location_x_B.append(temp_location_x)
            time_location_y_B.append(temp_location_y)

    print("Already get the relations in {:.3f} secs.".format(time.time() - start_init_matrix_locations))

    relation_nums = len(time_location_x_A) + len(time_location_x_B)
    print("Relations num is: {}.".format(relation_nums))

    time_matrix = sp.dok_matrix((n_items_A + n_users + n_items_B, relation_nums), dtype=np.float32)

    start_adj_matrix_A = time.time()
    for index in range(len(time_location_x_A)):
        time_matrix[time_location_x_A[index],
                    time_matrix_A[time_location_x_A[index], time_location_y_A[index]]] += 1.0
    print("Already add the Domain-A's relations to matrix in {:.3f} secs.".format(time.time() - start_adj_matrix_A))

    start_adj_matrix_B = time.time()
    for index in range(len(time_location_x_B)):
        time_matrix[n_items_A + n_users + time_location_x_B[index],
                    time_matrix_B[time_location_x_B[index], time_location_y_B[index]]] += 1.0
    print("Already add the Domain-B's relations to matrix in {:.3f} secs.".format(time.time() - start_adj_matrix_B))

    time_coo_matrix = normalize_laplace_matrix(time_matrix)
    print('Already generate the time_coo_matrix.')
    return time_coo_matrix, relation_nums





def get_rating_matrix(data_input):
    L_uid_itemA, L_uid_itemB, L_itemA_uid, L_itemB_uid, L_neighbor_item_A, L_neighbor_item_B, L_uid_uid \
        = [], [], [], [], [], [], []
    output_ratings = []
    for data_unit in data_input:
        seq_A = data_unit[0]
        seq_B = data_unit[1]
        items_A = [int(i) for i in seq_A[1:]]
        items_B = [int(j) for j in seq_B[1:]]

        uid = int(seq_A[0])
        for item_A in items_A:
            L_uid_itemA.append([uid, item_A])
            L_itemA_uid.append([item_A, uid])

        for item_B in items_B:
            L_uid_itemB.append([uid, item_B])
            L_itemB_uid.append([item_B, uid])

        for item_index_A in range(0, len(items_A) - 1):
            item_temp_A = items_A[item_index_A]
            next_item_A = items_A[item_index_A + 1]
            L_neighbor_item_A.append([item_temp_A, item_temp_A])
            L_neighbor_item_A.append([item_temp_A, next_item_A])

        for item_index_B in range(0, len(items_B) - 1):
            item_temp_B = items_B[item_index_B]
            next_item_B = items_B[item_index_B + 1]
            L_neighbor_item_B.append([item_temp_B, item_temp_B])
            L_neighbor_item_B.append([item_temp_B, next_item_B])

        L_uid_uid.append([uid, uid])

    matrix_U_A = dumplicate_matrix(L_uid_itemA)
    matrix_U_B = dumplicate_matrix(L_uid_itemB)
    matrix_A_U = dumplicate_matrix(L_itemA_uid)
    matrix_B_U = dumplicate_matrix(L_itemB_uid)
    matrix_A_neighbor = dumplicate_matrix(L_neighbor_item_A)
    matrix_B_neighbor = dumplicate_matrix(L_neighbor_item_B)
    matrix_U_U = dumplicate_matrix(L_uid_uid)

    output_ratings.append(np.array(matrix_U_A))
    output_ratings.append(np.array(matrix_U_B))
    output_ratings.append(np.array(matrix_A_U))
    output_ratings.append(np.array(matrix_B_U))
    output_ratings.append(np.array(matrix_A_neighbor))
    output_ratings.append(np.array(matrix_B_neighbor))
    output_ratings.append(np.array(matrix_U_U))

    return output_ratings


def matrix2inverse(array_in, row_pre, col_pre, len_all):
    matrix_rows = array_in[:, 0] + row_pre
    matrix_columns = array_in[:, 1] + col_pre
    matrix_value = [1.] * len(matrix_rows)
    inverse_matrix = sp.coo_matrix((matrix_value, (matrix_rows, matrix_columns)),
                                   shape=(len_all, len_all))
    return inverse_matrix


def get_laplace_list(ratings, dict_A, dict_B, dict_U):
    adj_mat_list = []

    num_items_A = len(dict_A)
    num_items_B = len(dict_B)
    num_users = len(dict_U)
    num_all = num_items_A + num_users + num_items_B
    print("The dimension of all matrix is: {}".format(num_all))

    inverse_matrix_A_A = matrix2inverse(ratings[4], row_pre=0, col_pre=0, len_all=num_all)
    inverse_matrix_A_U = matrix2inverse(ratings[2], row_pre=0, col_pre=num_items_A, len_all=num_all)
    inverse_matrix_U_A = matrix2inverse(ratings[0], row_pre=num_items_A, col_pre=0, len_all=num_all)
    inverse_matrix_U_U = matrix2inverse(ratings[6], row_pre=num_items_A, col_pre=num_items_A,
                                        len_all=num_all)
    inverse_matrix_U_B = matrix2inverse(ratings[1], row_pre=num_items_A, col_pre=num_items_A + num_users,
                                        len_all=num_all)
    inverse_matrix_B_U = matrix2inverse(ratings[3], row_pre=num_items_A + num_users, col_pre=num_items_A,
                                        len_all=num_all)
    inverse_matrix_B_B = matrix2inverse(ratings[5], row_pre=num_items_A + num_users,
                                        col_pre=num_items_A + num_users, len_all=num_all)

    print('Already convert the rating matrix into adjusted matrix.')
    adj_mat_list.append(inverse_matrix_U_A)
    adj_mat_list.append(inverse_matrix_U_B)
    adj_mat_list.append(inverse_matrix_A_U)
    adj_mat_list.append(inverse_matrix_B_U)
    adj_mat_list.append(inverse_matrix_A_A)
    adj_mat_list.append(inverse_matrix_B_B)
    adj_mat_list.append(inverse_matrix_U_U)

    laplace_list = [adj.tocoo() for adj in adj_mat_list]

    return laplace_list


def reorder_list(org_list, order):
    new_list = np.array(org_list)
    new_list = new_list[order]
    return new_list


def process_laplace(laplace_list):
    row_index_list, column_index_list = [], []
    value_list = []
    for index, coo_matrix in enumerate(laplace_list):
        row_index_list += list(coo_matrix.row)
        column_index_list += list(coo_matrix.col)
        value_list += list(coo_matrix.data)

    assert len(row_index_list) == sum([len(lap.data) for lap in laplace_list])
    print('Start reordering indices...')
    matrix_sum_dict = dict()

    for index, row_index in enumerate(row_index_list):
        if row_index not in matrix_sum_dict.keys():
            matrix_sum_dict[row_index] = [[], []]
        matrix_sum_dict[row_index][0].append(column_index_list[index])
        matrix_sum_dict[row_index][1].append(value_list[index])
    sorted_sum_dict = dict()
    for row_index in matrix_sum_dict.keys():
        column_list, values_list = matrix_sum_dict[row_index]
        order2sort = np.argsort(np.array(column_list))

        sorted_column_list = reorder_list(column_list, order2sort)
        sorted_values_list = reorder_list(values_list, order2sort)

        sorted_sum_dict[row_index] = [sorted_column_list, sorted_values_list]
    print('Already reordered the list...')

    ordered_dict = collections.OrderedDict(sorted(sorted_sum_dict.items()))
    row_list_output, column_list_output, values_list_output = [], [], []

    for row_index, value_index in ordered_dict.items():
        row_list_output += [row_index] * len(value_index[0])
        column_list_output += list(value_index[0])
        values_list_output += list(value_index[1])

    assert sum(row_list_output) == sum(row_index_list)
    assert sum(column_list_output) == sum(column_index_list)

    return row_list_output, column_list_output, values_list_output


def get_batches(input_data, batch_size, padding_num_A, padding_num_B, dict_A, isTrain):
    uid_all, seq_A_list, seq_B_list, len_A_list, len_B_list, target_A_list, target_B_list = [], [], [], [], [], [], []
    num_batches = int(len(input_data) / batch_size)

    if isTrain is True:
        random.shuffle(input_data)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        batch = input_data[start_index:start_index + batch_size]
        uid, seq_A, seq_B, len_A, len_B, target_A, target_B = batch_to_input(batch=batch, padding_num_A=padding_num_A,
                                                                             padding_num_B=padding_num_B, dict_A=dict_A)
        uid_all.append(uid)
        seq_A_list.append(seq_A)
        seq_B_list.append(seq_B)

        len_A_list.append(len_A)
        len_B_list.append(len_B)

        target_A_list.append(target_A)
        target_B_list.append(target_B)

    return list((uid_all, seq_A_list, seq_B_list, len_A_list, len_B_list, target_A_list, target_B_list, num_batches))


def batch_to_input(batch, padding_num_A, padding_num_B, dict_A):
    uid, seq_A, seq_B, len_A, len_B, target_A, target_B = [], [], [], [], [], [], []
    for data_index in batch:
        len_A.append(data_index[4])
        len_B.append(data_index[5])
    maxlen_A = max(len_A)
    maxlen_B = max(len_B)
    i = 0
    for data_index in range(len(batch)):
        uid.append(batch[data_index][0][0])
        seq_A.append(batch[data_index][0][1:] + [padding_num_A] * (maxlen_A - len_A[i]))
        seq_B.append(batch[data_index][1][1:] + [padding_num_B] * (maxlen_B - len_B[i]))
        target_A.append(batch[data_index][6])
        target_B.append(batch[data_index][7])
        i += 1

    return np.array(uid), np.array(seq_A), np.array(seq_B), np.array(len_A).reshape(len(len_A), 1), np.array(
        len_B).reshape(len(len_B), 1), np.array(target_A), np.array(target_B)

def get_hvideo_dict(path):
    itemE1 = []
    itemV1 = []
    account = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            U = line[0]
            account.append(U)
            for item in line[1:]:
                item = item.split('|')[0]
                if item[0] == 'E':
                    itemE1.append(item)
                else:
                    itemV1.append(item)
        itemE1 = pd.DataFrame(itemE1, columns=['itemEID'])
        itemV1 = pd.DataFrame(itemV1, columns=['itemVID'])
        account = pd.DataFrame(account, columns=['accountID'])
        itemE1.duplicated()
        itemE1.drop_duplicates(inplace=True)
        itemV1.duplicated()
        itemV1.drop_duplicates(inplace=True)
        account.duplicated()
        account.drop_duplicates(inplace=True)
        itemE1 = itemE1.values.tolist()
        itemV1 = itemV1.values.tolist()
        account = account.values.tolist()
        itemE1 = list(itertools.chain(*itemE1))
        itemV1 = list(itertools.chain(*itemV1))
        account = list(itertools.chain(*account))
        I1= list(range(len(itemE1)))
        I2 = list(range(len(itemV1)))
        U1 = list(range(len(account)))
        itemdictE = dict(zip(itemE1,I1))
        itemdictV = dict(zip(itemV1,I2))
        itemdictU = dict(zip(account,U1))
    return itemdictE, itemdictV,itemdictU
