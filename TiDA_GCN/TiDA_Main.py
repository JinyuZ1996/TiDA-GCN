# @Author: Jinyu Zhang
# @Time: 2021/11/1 16:18
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

from TiDA_GCN.TiDA_Configuration import *
from TiDA_GCN.TiDA_Train import *
from TiDA_GCN.TiDA_Printer import *
from time import *

np.seterr(all='ignore')
args = Settings()

if __name__ == '__main__':
    print_configuration(args=args)
    if args.dataset is 'Hvideo':
        dict_A, dict_B, dict_U = get_hvideo_dict(path=args.path_dict)
    else:
        dict_A = load_dict(dict_path=args.path_dict_A)
        dict_B = load_dict(dict_path=args.path_dict_B)
        dict_U = load_dict(dict_path=args.path_dict_U)
    tdict_A, tdict_B = load_tdict(all_data_path=args.all_path, dict_A=dict_A, dataset=args.dataset)
    print("Already load the dicts of datasets and timestamps.")
    mixed_train = get_data(data_path=args.train_path, dict_A=dict_A, dict_B=dict_B, dict_U=dict_U, tdict_A=tdict_A,
                           tdict_B=tdict_B, dataset=args.dataset)
    mixed_test = get_data(data_path=args.test_path, dict_A=dict_A, dict_B=dict_B, dict_U=dict_U, tdict_A=tdict_A,
                          tdict_B=tdict_B, dataset=args.dataset)
    print("Already load the mixed data.")

    mixed_train = mixed_train[:int(len(mixed_train)*1)]
    train_data, maxlen_A, maxlen_B = process_data(mixed_data=mixed_train, dict_A=dict_A, tdict_A=tdict_A)
    test_data, _, _ = process_data(mixed_data=mixed_test, dict_A=dict_A, tdict_A=tdict_A)
    print("The data processing is completed. Start initialize time matrix.")
    time_matrix, num_relations = generate_time_matrix(train_data, dict_A, dict_B, dict_U)
    print("Already get the time matrix.")
    output_ratings = get_rating_matrix(train_data)
    print("Already load the ratings.")
    laplace_list = get_laplace_list(output_ratings, dict_A, dict_B, dict_U)

    print("Already load the adj_list.")
    matrix_AUB_info = sum(laplace_list)

    processed_row_list, processed_column_list, processed_value_list = process_laplace(laplace_list)
    print("Already finished the process of all_data.")

    padding_num_A = len(dict_A)
    padding_num_B = len(dict_B)
    padding_sum = padding_num_A + padding_num_B

    train_batches = get_batches(input_data=train_data, batch_size=args.batch_size, padding_num_A=padding_num_A,
                                padding_num_B=padding_sum, dict_A=dict_A, isTrain=True)
    test_batches = get_batches(input_data=test_data, batch_size=args.batch_size, padding_num_A=padding_num_A,
                               padding_num_B=padding_sum, dict_A=dict_A, isTrain=False)
    print("Already load the batches.")

    n_items_A = len(dict_A)
    n_items_B = len(dict_B)
    n_users = len(dict_U)

    recommender = GCN_Module(n_items_A=n_items_A, n_items_B=n_items_B, n_users=n_users, maxlen_A=maxlen_A,
                             maxlen_B=maxlen_B, matrix_AUB_info=matrix_AUB_info, time_matrix=time_matrix, num_relations=num_relations, processed_row_list=processed_row_list,
                             processed_column_list=processed_column_list, processed_value_list=processed_value_list)
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

            train_batches = get_batches(input_data=train_data, batch_size=args.batch_size, padding_num_A=padding_num_A,
                                        padding_num_B=padding_sum, dict_A=dict_A, isTrain=True)

        print("Recommender training finished.")
        logging.info("Recommender training finished.")

        print("End.")

