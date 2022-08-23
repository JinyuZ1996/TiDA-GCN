# @Author: Jinyu Zhang
# @Time: 2021/1/05 15:25
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8
from DA_GCN.DA_Evaluation import *
from DA_GCN.DA_Module import *


args = Settings()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num


def train(sess, GCN_net, batches_train):
    uid_all, seq_A_all, seq_B_all, target_A_all, target_B_all, train_batch_num = (
        batches_train[0], batches_train[1], batches_train[2], batches_train[3],
        batches_train[4], batches_train[5])

    shuffled_batch_indexes = np.random.permutation(train_batch_num)
    avg_loss = 0

    for batch_index in shuffled_batch_indexes:
        uid = uid_all[batch_index]
        seq_A = seq_A_all[batch_index]
        seq_B = seq_B_all[batch_index]
        target_A = target_A_all[batch_index]
        target_B = target_B_all[batch_index]

        train_loss, _ = GCN_net.train_GCN(sess=sess, uid=uid, seq_A=seq_A, seq_B=seq_B,
                                          target_A=target_A, target_B=target_B, learning_rate=args.learning_rate,
                                          dropout_rate=args.dropout_rate, keep_prob=args.keep_prob)
        avg_loss += train_loss

    rec_loss = avg_loss / train_batch_num
    return rec_loss


def evaluate_module(sess, GCN_net, test_batches, test_len):
    uid_all, seq_A_all, seq_B_all, target_A_all, target_B_all, test_batch_num \
        = (test_batches[0], test_batches[1], test_batches[2], test_batches[3], test_batches[4],
           test_batches[5])

    return evaluate_ratings(sess=sess, GCN_net=GCN_net, uid=uid_all, seq_A=seq_A_all, seq_B=seq_B_all,
                            target_A=target_A_all, target_B=target_B_all,
                            test_batch_num=test_batch_num, test_length=test_len)