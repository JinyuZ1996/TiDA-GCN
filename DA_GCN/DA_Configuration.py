# @Author: Jinyu Zhang
# @Time: 2021/1/03 14:37
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

import collections
import scipy.sparse as sp
import numpy as np
import pandas as pd
import os
import random
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[ ]:

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


def get_data(data_path, dict_A, dict_B, dict_U):
    with open(data_path, 'r') as file_object:
        mixed_data = []
        lines = file_object.readlines()
        for line in lines:
            temp_sequence = []
            line = line.strip().split('\t')
            sequence_all = []
            user = line[0]  # 每行第一个是uid
            sequence_all.append(dict_U[user])  # 现在混合seq第一个位置上拼上uid
            for item in line[1:]:  # 从line中的第二项开始，遍历line中的item，从'E241'到'V326'；for循环将line转换为对应的索引列表；
                item_info = item.split('|')
                item_id = item_info[0]
                if item_id in dict_A:
                    sequence_all.append(dict_A[item_id])
                else:
                    sequence_all.append(dict_B[item_id] + len(dict_A))  # 为了区分序列中的E与V物品，len(itemE)=8367；
            temp_sequence.append(sequence_all)  # [0]
            mixed_data.append(temp_sequence)
    return mixed_data


def process_data(mixed_data, dict_A):
    data_outputs = []
    maxlen_A, maxlen_B = 0, 0
    for index in mixed_data:
        temp = []
        seq_A, seq_B = [], []
        len_A, len_B = 0, 0
        mixed_sequence = index[0]
        seq_A.append(mixed_sequence[0])  # 先给两个seq拼上uid
        seq_B.append(mixed_sequence[0])
        for item_id in mixed_sequence[1:-2]:  # 第一个是uid, 最后两个是target
            if item_id < len(dict_A):  # 这个边界值是小于还是小于等于要去测一下
                seq_A.append(item_id)
                len_A += 1
            else:
                seq_B.append(item_id - len(dict_A))
                len_B += 1
        temp.append(seq_A)  # [0]
        temp.append(seq_B)  # [1]
        temp.append(len_A)  # [2]
        temp.append(len_B)  # [3]
        if maxlen_A <= len_A:
            maxlen_A = len_A
        if maxlen_B <= len_B:
            maxlen_B = len_B
        temp.append(mixed_sequence[-2])  # [4]
        temp.append(mixed_sequence[-1] - len(dict_A))  # [5]
        data_outputs.append(temp)

    return data_outputs, maxlen_A, maxlen_B


def dumplicate_matrix(matrix_in):
    frame_temp = pd.DataFrame(matrix_in, columns=['row', 'column'])
    frame_temp.duplicated()
    frame_temp.drop_duplicates(inplace=True)

    return frame_temp.values.tolist()


def get_rating_matrix(all_data_input):
    L_uid_itemA, L_uid_itemB, L_itemA_uid, L_itemB_uid, L_neighbor_item_A, L_neighbor_item_B, L_uid_uid \
        = [], [], [], [], [], [], []
    output_ratings = []
    for data_unit in all_data_input:
        seq_A = data_unit[0]
        seq_B = data_unit[1]
        items_A = [int(i) for i in seq_A[1:]]  # [0]是Uid
        items_B = [int(j) for j in seq_B[1:]]

        uid = int(seq_A[0])
        for item_A in items_A:
            L_uid_itemA.append([uid, item_A])  # 第1个list存的是uid和item_A的对儿[0]
            L_itemA_uid.append([item_A, uid])  # 第3个list存的是item_A和uid的对儿[2]

        for item_B in items_B:
            L_uid_itemB.append([uid, item_B])  # 第2个list存的是uid, item_B[1]
            L_itemB_uid.append([item_B, uid])  # 第4个list存的是item_B, uid[3]

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

        L_uid_uid.append([uid, uid])  # [6]

    matrix_U_A = dumplicate_matrix(L_uid_itemA)
    matrix_U_B = dumplicate_matrix(L_uid_itemB)
    matrix_A_U = dumplicate_matrix(L_itemA_uid)
    matrix_B_U = dumplicate_matrix(L_itemB_uid)
    matrix_A_neighbor = dumplicate_matrix(L_neighbor_item_A)
    matrix_B_neighbor = dumplicate_matrix(L_neighbor_item_B)
    matrix_U_U = dumplicate_matrix(L_uid_uid)

    output_ratings.append(np.array(matrix_U_A))  # [0]
    output_ratings.append(np.array(matrix_U_B))  # [1]
    output_ratings.append(np.array(matrix_A_U))  # [2]
    output_ratings.append(np.array(matrix_B_U))  # [3]
    output_ratings.append(np.array(matrix_A_neighbor))  # [4]
    output_ratings.append(np.array(matrix_B_neighbor))  # [5]
    output_ratings.append(np.array(matrix_U_U))  # [6]

    return output_ratings


### 这个方法是为了求矩阵的逆？
def matrix2inverse(array_in, row_pre, col_pre, len_all):  # np.matrix转换为sparse_matrix;
    matrix_rows = array_in[:, 0] + row_pre  # X[:,0]表示对一个二维数组train_data取所有行的第一列数据;是numpy中数组的一种写法，
    matrix_columns = array_in[:, 1] + col_pre  # X[:,1]就是取所有行的第2列数据；类型为 ndarray;
    matrix_value = [1.] * len(matrix_rows)  # 只对交互过的（user,item）赋值 1.0；类型为list;
    inverse_matrix = sp.coo_matrix((matrix_value, (matrix_rows, matrix_columns)),
                                   shape=(len_all, len_all))  # shape=(129955,129955); dtype=float64;
    return inverse_matrix


def get_laplace_list(ratings, dict_A, dict_B, dict_U):
    adj_mat_list = []  # 定义一个列表；

    num_items_A = len(dict_A)
    num_items_B = len(dict_B)
    num_users = len(dict_U)
    num_all = num_items_A + num_users + num_items_B
    print("The dimension of all matrix is: {}".format(num_all))

    # 2021-11-18 重新按照在大矩阵中的位置给他们排一下序：A-U-B-tA-tB
    # 1: [item_A, next_item_A] + [item_A, item_A]
    inverse_matrix_A_A = matrix2inverse(ratings[4], row_pre=0, col_pre=0, len_all=num_all)
    # 2: [item_A, uid]
    inverse_matrix_A_U = matrix2inverse(ratings[2], row_pre=0, col_pre=num_items_A, len_all=num_all)
    # 3: [uid, item_A]
    inverse_matrix_U_A = matrix2inverse(ratings[0], row_pre=num_items_A, col_pre=0, len_all=num_all)
    # 4: [uid, uid]
    inverse_matrix_U_U = matrix2inverse(ratings[6], row_pre=num_items_A, col_pre=num_items_A,
                                        len_all=num_all)
    # 5: [uid, item_B]
    inverse_matrix_U_B = matrix2inverse(ratings[1], row_pre=num_items_A, col_pre=num_items_A + num_users,
                                        len_all=num_all)
    # 6: [item_B, uid]
    inverse_matrix_B_U = matrix2inverse(ratings[3], row_pre=num_items_A + num_users, col_pre=num_items_A,
                                        len_all=num_all)
    # 7: [item_B, next_item_B] + [item_B, item_B]
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

    # 将矩阵转为坐标格式，再存进list里
    laplace_list = [adj.tocoo() for adj in adj_mat_list]  # 拉普拉斯矩阵；

    return laplace_list


# 实现ndarray升序排列
def reorder_list(org_list, order):
    new_list = np.array(org_list)
    new_list = new_list[order]
    return new_list


def process_all_data(laplace_list):
    row_index_list, column_index_list = [], []  # 定义4个list;   添加all_r_list;
    value_list = []
    for index, coo_matrix in enumerate(laplace_list):  # 遍历 lap_list;
        row_index_list += list(coo_matrix.row)  # 存储 lap_list的行索引；把7组coo_matrix的行索引放在一起；
        column_index_list += list(coo_matrix.col)  # 存储 lap_list的列索引；
        value_list += list(coo_matrix.data)  # 存储 lap_list的values(经过规范化的)；

    assert len(row_index_list) == sum([len(lap.data) for lap in laplace_list])  # assert:断言；如果满足条件表达式，程序继续往下运行；
    ### 这里应该只有亚马逊数据集需要执行以下的操作
    print('Start reordering indices...')
    matrix_sum_dict = dict()

    for index, row_index in enumerate(row_index_list):
        ### 将 row_index ,column_index, values对应起来2021-11-11
        if row_index not in matrix_sum_dict.keys():
            matrix_sum_dict[row_index] = [[], []]
        matrix_sum_dict[row_index][0].append(column_index_list[index])
        matrix_sum_dict[row_index][1].append(value_list[index])
    #### 这部分代码的作用就是将上面的dict进行重排序，肯定可以优化，以后再说，2021-11-11
    sorted_sum_dict = dict()
    for row_index in matrix_sum_dict.keys():
        column_list, values_list = matrix_sum_dict[row_index]
        order2sort = np.argsort(np.array(column_list))  # np.argsort():返回数组从小到大的索引值；ndarray;

        sorted_column_list = reorder_list(column_list, order2sort)
        sorted_values_list = reorder_list(values_list, order2sort)

        sorted_sum_dict[row_index] = [sorted_column_list, sorted_values_list]
    print('Already reordered the list...')

    ordered_dict = collections.OrderedDict(sorted(sorted_sum_dict.items()))  # OrderedDict:字典的子类，保存了他们被添加的顺序;
    row_list_output, column_list_output, values_list_output = [], [], []

    for row_index, value_index in ordered_dict.items():
        row_list_output += [row_index] * len(value_index[0])
        column_list_output += list(value_index[0])
        values_list_output += list(value_index[1])

    assert sum(row_list_output) == sum(row_index_list)  # 14595009823;
    assert sum(column_list_output) == sum(column_index_list)  # 14592683979;

    return row_list_output, column_list_output, values_list_output


def get_batches(input_data, batch_size, padding_num_A, padding_num_B, isTrain):
    uid_all, seq_A_list, seq_B_list, len_A_list, len_B_list, target_A_list, target_B_list = [], [], [], [], [], [], []
    num_batches = int(len(input_data) / batch_size)

    if isTrain is True:
        random.shuffle(input_data)

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        batch = input_data[start_index:start_index + batch_size]
        uid, seq_A, seq_B, len_A, len_B, target_A, target_B = batch_to_input(batch=batch, padding_num_A=padding_num_A,
                                                                             padding_num_B=padding_num_B)
        uid_all.append(uid)
        seq_A_list.append(seq_A)
        seq_B_list.append(seq_B)

        # len_A_list.append(len_A)
        # len_B_list.append(len_B)

        target_A_list.append(target_A)
        target_B_list.append(target_B)

    return list((uid_all, seq_A_list, seq_B_list, target_A_list, target_B_list, num_batches))


def batch_to_input(batch, padding_num_A, padding_num_B):
    uid, seq_A, seq_B, len_A, len_B, target_A, target_B = [], [], [], [], [], [], []
    ### len只是为了算一下最长的len是多少
    for data_index in batch:
        len_A.append(data_index[2])
        len_B.append(data_index[3])
    maxlen_A = max(len_A)
    maxlen_B = max(len_B)
    i = 0
    for data_index in range(len(batch)):
        uid.append(batch[data_index][0][0])  # 之前在format处理的时候将第一位的user单独拿出来
        seq_A.append(batch[data_index][0][1:] + [padding_num_A] * (maxlen_A - len_A[i]))
        seq_B.append(batch[data_index][1][1:] + [padding_num_B] * (maxlen_B - len_B[i]))
        target_A.append(batch[data_index][4])
        target_B.append(batch[data_index][5])
        i += 1

    return np.array(uid), np.array(seq_A), np.array(seq_B), np.array(len_A).reshape(len(len_A), 1), np.array(
        len_B).reshape(len(len_B), 1), np.array(target_A), np.array(target_B)


def get_hvideo_dict(path):  # r_path:形参；
    itemE1 = []
    itemV1 = []
    account = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            # if len(line)==31:
            #     del line[14]
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
        I1 = list(range(len(itemE1)))  # 8367;
        I2 = list(range(len(itemV1)))  # 11404;
        U1 = list(range(len(account)))  # 13714;
        itemdictE = dict(zip(itemE1, I1))
        itemdictV = dict(zip(itemV1, I2))
        itemdictU = dict(zip(account, U1))
    return itemdictE, itemdictV, itemdictU
