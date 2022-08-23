# @Author: Jinyu Zhang
# @Time: 2021/11/15 13:52
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

import logging


def print_rec_message(epoch, rec_loss, rec_pre_time):
    print('Epoch {} - Training Loss: {:.5f} - Training time: {:.3}'.format(epoch, rec_loss,
                                                                           rec_pre_time))
    logging.info('Epoch {} - Training Loss: {:.5f}'.format(epoch, rec_loss))

def print_configuration(args):
    print("Dataset:{}, Learning Rate:{:.4f}, Epoch:{}, Batch-Size:{}, Alpha:{:.2f}, Embedding-Size:{}, H:{:.2f}".format(
        args.dataset, args.learning_rate, args.epochs, args.batch_size, args.Alpha, args.embedding_size, args.n_member
    ))

def print_recommender_train(epoch, RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A, RC_5_B, RC_10_B,
                            RC_20_B, MRR_5_B, MRR_10_B, MRR_20_B, rec_test_time):
    print(
        "Evaluate on Domain-A, Epoch %d : MRR5 = %.4f, MRR20 = %.4f, RC5 = %.4f, RC20 = %.4f, MRR10 = %.4f, RC10 = %.4f" % (
            epoch, MRR_5_A, MRR_20_A, RC_5_A, RC_20_A, MRR_10_A, RC_10_A))
    print(
        "Evaluate on Domain-B, Epoch %d : MRR5 = %.4f, MRR20 = %.4f, RC5 = %.4f, RC20 = %.4f, MRR10 = %.4f, RC10 = %.4f" % (
            epoch, MRR_5_B, MRR_20_B, RC_5_B, RC_20_B, MRR_10_B, RC_10_B))
    logging.info(
        "Evaluate on Domain-A, Epoch %d : MRR5 = %.4f, MRR20 = %.4f, RC5 = %.4f, RC20 = %.4f, MRR10 = %.4f, RC10 = %.4f" % (
            epoch, MRR_5_A, MRR_20_A, RC_5_A, RC_20_A, MRR_10_A, RC_10_A))
    logging.info(
        "Evaluate on Domain-B, Epoch %d : MRR5 = %.4f, MRR20 = %.4f, RC5 = %.4f, RC20 = %.4f, MRR10 = %.4f, RC10 = %.4f" % (
            epoch, MRR_5_B, MRR_20_B, RC_5_B, RC_20_B, MRR_10_B, RC_10_B))
    print("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, rec_test_time))
    logging.info("Epoch: {}, Recommender evaluate time: {:.3f}".format(epoch, rec_test_time))
