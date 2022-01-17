import os
import re
import pathlib
import joblib
import multiprocessing
import sentencepiece as spm
import numpy as np
import random
from scipy.sparse import lil_matrix, csr_matrix, vstack
from datetime import datetime
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer

from classify import train_model
from hash import wta_vectorized
from utils import read_vocab, read_n_encode_dataset


class Fly:
    def __init__(self,
                 layer_size_list=None,  # e.g. 3 layers [10000, 5000, 3000]
                 num_proj_list=None,  # e.g. [3, 4, 5]
                 num_nonzero_list=None,  # e.g. [300, 300, 300]
                 n_dataset=3,
                 ):
        self.layer_size_list = layer_size_list
        self.num_proj_list = num_proj_list
        self.num_nonzero_list = num_nonzero_list
        # self.wta = num_nonzero / kc_size * 100
        self.val_score_mat = np.zeros([n_dataset, len(num_proj_list)])
        self.projection_list = self.create_proj_mats()

    def create_proj_mats(self):
        def _create_one_proj(in_size, out_size, num_proj):
            # print(in_size, out_size, num_proj)
            weight_mat = np.zeros((out_size, in_size))
            # uniformly random init, cover all vocab
            # if init_type == 1:
            idx = list(range(in_size))
            c = 0
            while c < out_size:
                random.shuffle(idx)
                for i in range(0, len(idx), num_proj):
                    p = idx[i:i + num_proj]
                    for j in p:
                        weight_mat[c][j] = 1
                    c += 1
                    if c >= out_size:
                        break
            # else:
                # pass
            return lil_matrix(weight_mat)

        proj_list = []
        for k in range(len(self.layer_size_list) - 1):
            proj_list.append(_create_one_proj(in_size=self.layer_size_list[k],
                                              out_size=self.layer_size_list[k+1],
                                              num_proj=self.num_proj_list[k]))
        return proj_list

    def get_avg_score(self):
        # average over 3 datasets
        return np.mean(self.val_score_mat, axis=0)

    def evaluate(self):
        # hash
        hs_train_list, hs_val_list = [], []
        for i in range(len(train_set_list)):
            hash_train = hash_multi_layer(dataset_mat=train_set_list[i],
                                          weight_list=self.projection_list,
                                          nonzero_list=self.num_nonzero_list,
                                          top_words=TOP_WORDS)
            hs_train_list.append(hash_train)
            hash_val = hash_multi_layer(dataset_mat=val_set_list[i],
                                        weight_list=self.projection_list,
                                        nonzero_list=self.num_nonzero_list,
                                        top_words=TOP_WORDS)
            hs_val_list.append(hash_val)

        # train
        n_dataset, n_layer = self.val_score_mat.shape
        val_scores_and_models = joblib.Parallel(n_jobs=n_dataset*n_layer, prefer="threads")(
            joblib.delayed(train_model)(m_train=hs_train_list[i][j],
                                        classes_train=train_label_list[i],
                                        m_val=hs_val_list[i][j],
                                        classes_val=val_label_list[i],
                                        C=C,
                                        num_iter=NUM_ITER,
                                        ) for i in range(n_dataset) for j in range(n_layer))
        val_score_list = [t[0] for t in val_scores_and_models]
        self.val_score_mat = np.array(val_score_list).reshape(n_dataset, n_layer)
        print(self.val_score_mat)


def hash_multi_layer(dataset_mat, weight_list, nonzero_list, top_words):
    def _wta(feat_mat, num_retain):
        m, n = feat_mat.shape
        wta_csr = csr_matrix(np.zeros(n))
        for i in range(0, m, 2000):
            part = wta_vectorized(feat_mat[i: i + 2000].toarray(), k=num_retain, percent=False)
            wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
        return wta_csr[1:]

    def _hash_vectorized(input_mat, weight_mat, num_nonzero):
        output_mat = input_mat.dot(weight_mat.T)
        hashed_output = _wta(feat_mat=output_mat, num_retain=num_nonzero)
        return hashed_output
    
    # apply wta to the data layer
    dataset_mat = csr_matrix(dataset_mat)
    layer_input = _wta(feat_mat=dataset_mat, num_retain=top_words)

    # hash layer by layer
    hs_list = []
    for layer in range(len(weight_list)):
        hs = _hash_vectorized(input_mat=layer_input,
                              weight_mat=weight_list[layer],
                              num_nonzero=nonzero_list[layer])
        bin_hs = (hs > 0).astype(np.int_)
        hs_list.append(bin_hs)
        # TODO add non linear here
        layer_input = bin_hs  # the hash becomes input of the next layer
    return hs_list


def create_fly():
    # modify here the config of layers
    layer_size_list = [10000, 10000, 10000]
    num_proj_list = [5, 5, 5]
    num_nonzero_list = [300, 300, 300]
    fly = Fly(layer_size_list=[len(vocab)]+layer_size_list,
              num_proj_list=num_proj_list,
              num_nonzero_list=num_nonzero_list)
    return fly


def test_add_layer(n_flies):
    print(f'creating a swarm of {n_flies} flies...')
    flies_swarm = joblib.Parallel(n_jobs=n_flies, prefer="threads")(
        joblib.delayed(create_fly)() for _ in range(n_flies))

    print('evaluating...')
    joblib.Parallel(n_jobs=MAX_THREADS, prefer="threads")(
        joblib.delayed(fly.evaluate)() for fly in flies_swarm)

    for fly in flies_swarm:
        print(fly.get_avg_score())


if __name__ == '__main__':
    # args = docopt(__doc__, version='Test adding more layers to fruit-fly, ver 0.1')

    # pathlib.Path('./log').mkdir(parents=True, exist_ok=True)
    # main_log_path = f'./log/logs_{dataset_name}.json'
    # pathlib.Path(main_log_path).touch(exist_ok=True)
    # pathlib.Path(f'./models/projection/{dataset_name}').mkdir(parents=True, exist_ok=True)
    # now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # pathlib.Path(f'./models/classification/{dataset_name}_{now}').mkdir(parents=True, exist_ok=True)

    # global variables
    sp = spm.SentencePieceProcessor()
    spm_model = "../spm/spmcc.model"
    spm_vocab = "../spm/spmcc.vocab"
    sp.load(spm_model)
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')

    TOP_WORDS = 200
    NUM_ITER = 2000
    C = 1

    print('reading datasets...')
    num_dataset = 3
    train_set_list, train_label_list = [None] * num_dataset, [None] * num_dataset
    val_set_list, val_label_list = [None] * num_dataset, [None] * num_dataset
    train_set_list[0], train_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-train.sp', vectorizer, logprobs)
    val_set_list[0], val_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-val.sp', vectorizer, logprobs)
    train_set_list[1], train_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-train.sp', vectorizer, logprobs)
    val_set_list[1], val_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-val.sp', vectorizer, logprobs)
    train_set_list[2], train_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-train.sp', vectorizer, logprobs)
    val_set_list[2], val_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-val.sp', vectorizer, logprobs)
        
    MAX_THREADS = int(multiprocessing.cpu_count() * 0.5)

    # test
    test_add_layer(n_flies=5)
