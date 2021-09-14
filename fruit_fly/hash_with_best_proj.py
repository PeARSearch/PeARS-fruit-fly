"""Hash documents with best performing fly
Usage:
  hash_with_best_proj.py --dataset=<filename> 
  hash_with_best_proj.py (-h | --help)
  hash_with_best_proj.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<filename>         Name of training file
"""

import pickle
import numpy as np
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from evolve_flies import Fly
from hyperparam_search import read_n_encode_dataset
from classify import train_model
from hash import read_vocab
from utils import hash_dataset_

# load model
with open(f'./models/evolution/best_val_score', 'rb') as f:  # modified the name of the fruit-fly here
    best_fly = pickle.load(f)

# load dataset
top_word = 700
C = 100
num_iter = 2000  # wikipedia and wos only need 50 steps
sp = spm.SentencePieceProcessor()
sp.load('../spmcc.model')
vocab, reverse_vocab, logprobs = read_vocab()
vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
PN_SIZE = len(vocab)

print('reading dataset')
train_set, train_label = read_n_encode_dataset('/home/nhut/Downloads/wiki_cats_test.sp', vectorizer, logprobs)
# and here load the validation set

print('hashing')
hash_train = hash_dataset_(dataset_mat=train_set, weight_mat=best_fly.projection,
                           percent_hash=best_fly.wta, top_words=top_word)

