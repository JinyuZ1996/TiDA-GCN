# @Author: Jinyu Zhang
# @Time: 2021/1/03 14:55
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

from time import time
from DA_GCN.DA_Configuration import *
from DA_GCN.DA_Printer import *
from DA_GCN.DA_Train import *

np.seterr(all='ignore')
args = Settings()

if __name__ == '__main__':

    # to get the dict of dataset
    if args.dataset is 'Hvideo':
        dict_A, dict_B, dict_U = get_hvideo_dict(path=args.path_dict)
    else:  # Hamazon
        dict_A = load_dict(dict_path=args.path_dict_A)
        dict_B = load_dict(dict_path=args.path_dict_B)
        dict_U = load_dict(dict_path=args.path_dict_U)
    # to get the mixed data
    mixed_train = get_data(data_path=args.train_path, dict_A=dict_A, dict_B=dict_B, dict_U=dict_U)
    mixed_test = get_data(data_path=args.test_path, dict_A=dict_A, dict_B=dict_B, dict_U=dict_U)
    print("Already load the mixed data.")

    mixed_train = mixed_train[:int(len(mixed_train)*1)]

    train_data, _, _ = process_data(mixed_data=mixed_train, dict_A=dict_A)
    test_data, _, _ = process_data(mixed_data=mixed_test, dict_A=dict_A)
    all_data, maxlen_A, maxlen_B = process_data(mixed_data=mixed_train, dict_A=dict_A)
    print("The data processing is completed.")
    output_ratings = get_rating_matrix(all_data)
    print("Already load the ratings.")
    laplace_list = get_laplace_list(output_ratings, dict_A, dict_B, dict_U)
    print("Already load the adj_list.")
    A_in = sum(laplace_list)

    processed_row_list, processed_column_list, processed_value_list = process_all_data(laplace_list)
    print("Already finished the process of all_data.")

    train_batches = get_batches(input_data=train_data, batch_size=args.batch_size, padding_num_A=args.padding_int,
                                padding_num_B=args.padding_int, isTrain=True)
    test_batches = get_batches(input_data=test_data, batch_size=args.batch_size, padding_num_A=args.padding_int,
                               padding_num_B=args.padding_int, isTrain=False)
    print("Already load the batches.")

    n_items_A = len(dict_A)
    n_items_B = len(dict_B)
    n_users = len(dict_U)

    recommender = DA_GCN(n_items_A=n_items_A, n_items_B=n_items_B, n_users=n_users, A_in=A_in,
                         all_h_list=processed_row_list, all_t_list=processed_column_list,
                         all_v_list=processed_value_list)
    print("Already initialize the GCN_Module.")
    with tf.Session(graph=recommender.graph, config=recommender.config) as sess:

        recommender.sess = sess
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        best_score_domain_A = -1
        best_score_domain_B = -1

        for epoch in range(args.epochs):
            rec_pre_begin_time = time()
            rec_loss = train(sess=sess, GCN_net=recommender, batches_train=train_batches)
            rec_pre_time = time() - rec_pre_begin_time

            epoch_to_print = epoch + 1
            print_rec_message(epoch=epoch_to_print, rec_loss=rec_loss,
                              rec_pre_time=rec_pre_time)

            if epoch_to_print % args.verbose == 0:
                rec_test_begin_time = time()
                [RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A,
                 RC_5_B, RC_10_B, RC_20_B, MRR_B_5, MRR_B_10, MRR_B_20] = \
                    evaluate_module(sess=sess, GCN_net=recommender, test_batches=test_batches, test_len=len(test_data))
                rec_test_time = time() - rec_test_begin_time

                print_recommender_train(epoch_to_print, RC_5_A, RC_10_A, RC_20_A, MRR_5_A, MRR_10_A, MRR_20_A, RC_5_B,
                                        RC_10_B, RC_20_B, MRR_B_5, MRR_B_10, MRR_B_20, rec_test_time)

                if RC_5_A >= best_score_domain_A or RC_5_B >= best_score_domain_B:
                    best_score_domain_A = RC_5_A
                    best_score_domain_B = RC_5_B
                    saver.save(sess, args.checkpoint, global_step=epoch_to_print, write_meta_graph=False)
                    print("Recommender performs better, saving current model....")
                    logging.info("Recommender performs better, saving current model....")

            train_batches = get_batches(input_data=train_data, batch_size=args.batch_size,
                                        padding_num_A=args.padding_int,
                                        padding_num_B=args.padding_int, isTrain=True)

        print("Recommender training finished.")
        logging.info("Recommender training finished.")

        print("All process finished.")
