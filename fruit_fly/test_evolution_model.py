"""Run best performing fly on test sets

Usage:
  test_models_evolution.py --fly=<filename> 
  test_models_evolution.py (-h | --help)
  test_models_evolution.py --version
Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --fly=<filename>          Path to the fly to test. 

"""

import pickle
import numpy as np
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from evolve_flies import Fly
from hyperparam_search import read_n_encode_dataset
from utils import hash_dataset_, append_as_json, get_stats
from classify import train_model
from hash import read_vocab
from utils import hash_dataset_
from scipy import sparse
from scipy.sparse import csr_matrix
import pathlib
from docopt import docopt
import glob
import re

def test(fly_model):
  # load model
  with open(fly_model, 'rb') as f:
      best_fly = pickle.load(f)

      test_score_list = []
      for i in range(len(test_set_list)):
          hash_train = hash_dataset_(dataset_mat=train_set_list[i], weight_mat=best_fly.projection,
                                       percent_hash=best_fly.wta, top_words=top_word)
          hash_test = hash_dataset_(dataset_mat=test_set_list[i], weight_mat=best_fly.projection,
                                     percent_hash=best_fly.wta, top_words=top_word)
          test_score, _ = train_model(m_train=hash_train, classes_train=train_label_list[i],
                                       m_val=hash_test, classes_val=test_label_list[i],
                                       C=C, num_iter=num_iter)
          test_score_list.append(test_score)
          print("DATASET",i,"SCORE",test_score)
      return test_score_list


if __name__ == '__main__':
    args = docopt(__doc__, version='Evaluate a fly on test sets, ver 0.1')

    top_word = 250
    C = 100
    num_iter = 1000  # wikipedia and wos only need 50 steps
    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')
    vocab, reverse_vocab, logprobs = read_vocab()
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
    PN_SIZE = len(vocab)

    print('reading datasets')
    num_dataset = 3
    train_set_list, train_label_list = [None] * num_dataset, [None] * num_dataset
    test_set_list, test_label_list = [None] * num_dataset, [None] * num_dataset
    train_set_list[0], train_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-train.sp', vectorizer, logprobs)
    test_set_list[0], test_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-test.sp', vectorizer, logprobs)
    train_set_list[1], train_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-train.sp', vectorizer, logprobs)
    test_set_list[1], test_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-test.sp', vectorizer, logprobs)
    train_set_list[2], train_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-train.sp', vectorizer, logprobs)
    test_set_list[2], test_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-test.sp', vectorizer, logprobs)

    print(test(args["--fly"]))
