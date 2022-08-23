# @Author: Jinyu Zhang
# @Time: 2021/1/05 14:27
# @E-mail: JinyuZ1996@outlook.com

# coding: utf-8

class Settings:
    def __init__(self):
        self.code_path = "https://github.com/JinyuZ1996/TiDA-GCN"

        '''
            block 1: the hyper parameters for model training
        '''
        self.learning_rate = 0.001       # 0.001
        self.keep_prob = 0.9
        self.dropout_rate = 0.1
        self.batch_size = 256
        self.epochs = 50
        self.verbose = 10

        '''
            block 2: the hyper parameters for TiDA_Module.py
        '''
        self.embedding_size = 16
        self.n_layers = 1
        self.n_member = 1
        self.n_fold = 16
        self.layer_size = '['+str(self.embedding_size)+']'
        self.padding_int = 0
        self.gpu_num = '0'

        '''
            block 3: the hyper parameters for file paths
        '''
        self.dataset = 'Hvideo'     # Hvideo or Hamazon
        self.train_path = '../data/' + self.dataset + '/new_traindata.txt'
        self.test_path = '../data/' + self.dataset + '/new_testdata.txt'

        # for Hamazon dataset (only)
        self.path_dict_A = '../data/' + self.dataset + '/Alist.txt'
        self.path_dict_B = '../data/' + self.dataset + '/Blist.txt'
        self.path_dict_U = '../data/' + self.dataset + '/Ulist.txt'

        # for Hvideo dataset (only)
        self.path_dict = '../data/' + self.dataset + '/all_dict.txt'

        self.checkpoint = 'checkpoint/trained_model.ckpt'