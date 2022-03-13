"""Hyper-parameter search by Bayesian optimization
Usage:
  test_fly.py --dataset=<str> --logprob=<n>
  test_fly.py (-h | --help)
  test_fly.py --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --dataset=<str>              Name of dataset, either wiki, 20news, or wos.
  --logprob=<n>                Power of word log probabilities when vectorizing documents.
"""


import os
import re
import umap
import pathlib
import joblib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from joblib import Parallel, delayed,dump
import random
import multiprocessing
import sentencepiece as spm
import numpy as np
from datetime import datetime
from docopt import docopt
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack, lil_matrix, coo_matrix
from utils import read_vocab, hash_dataset_, read_n_encode_dataset

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from fly import Fly

import matplotlib.pyplot as plt

def run_tsne(m, n_components):
    tsne = TSNE(n_components,metric='cosine')
    tsne_result = tsne.fit_transform(m)
    return tsne_result

def run_pca(m, n_components):
    pca = PCA(n_components)
    pca_result = pca.fit_transform(m)
    return pca_result

def fruitfly_pipeline(kc_size, proj_size, wta, knn, num_trial):

    #Below parameters are needed to init the fruit fly, even if not used here
    init_method='random'
    eval_method='similarity'
    proj_store=None
    hyperparameters = {'C':100,'num_iter':200,'num_nns':knn}


    fly_list = [Fly(PN_SIZE, kc_size, wta, proj_size, init_method, eval_method, proj_store, hyperparameters) for _ in range(num_trial)]
    print('evaluating...')
    scores = []
    best_fly_score = 0.0


    with Parallel(n_jobs=max_thread, prefer="threads") as parallel:
        delayed_funcs = [delayed(lambda x:x.evaluate(train_set,val_set,train_label,val_label))(fly) for fly in fly_list]
        scores = parallel(delayed_funcs)

    score_list = [p[0] for p in scores]

    # average the validation acc
    avg_score = np.mean(score_list)
    std_score = np.std(score_list)
    print('average score:', avg_score)
    return avg_score


def optimize_fruitfly():
    knn=100
    def _evaluate(kc_size, wta, proj_size):
        kc_size = round(kc_size)
        proj_size = round(proj_size)
        wta = round(wta)
        num_trial = 3
        print(f'--- kc_size {kc_size}, wta {wta}, proj_size {proj_size}, knn {knn} ')
        return fruitfly_pipeline(kc_size, wta, proj_size, knn, num_trial)

    optimizer = BayesianOptimization(
        f=_evaluate,
        pbounds={"kc_size": (64, 256),"proj_size": (3, 10), "wta": (5, 10) },
        #random_state=1234,
        verbose=2
    )


    tmp_log_path = f'./log/logs_{dataset_name}_fly_{knn}.json'
    logger = JSONLogger(path=tmp_log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=2, n_iter=100)
    print("Final result:", optimizer.max['target'])
    
    #Saving a fly with the best params
    params = optimizer.max['params']
    print(params)
    hyperparameters = {'C':100,'num_iter':200,'num_nns':knn}
    fly = Fly(PN_SIZE, round(params['kc_size']), round(params['wta']), round(params['proj_size']), 'random', 'similarity', None, hyperparameters)
    score, _, _ = fly.evaluate(train_set,val_set,train_label,val_label)
    print("Score before pruning:",score)
    score, size = fly.prune(train_set,val_set,train_label,val_label)
    print("Score and size after pruning, saved fly:",score, size)

    filename = './models/flies/'+dataset+'.fly.m'
    joblib.dump(fly, filename)
    



if __name__ == '__main__':
    args = docopt(__doc__, version='Test fly, ver 0.1')
    dataset = args["--dataset"]
    power = int(args["--logprob"])

    if dataset == "wiki":
        train_path="../datasets/wikipedia/wikipedia-train.sp"
        spm_model = "../spm/spm.wiki.model"
        spm_vocab = "../spm/spm.wiki.vocab"
        umap_model = joblib.load("./models/umap/wiki.umap")
    if dataset == "20news":
        train_path="../datasets/20news-bydate/20news-bydate-train.sp"
        spm_model = "../spm/spm.20news.model"
        spm_vocab = "../spm/spm.20news.vocab"
        umap_model = joblib.load("./models/umap/20news.umap")
    if dataset == "wos":
        train_path="../datasets/wos/wos11967-train.sp"
        spm_model = "../spm/spm.wos.model"
        spm_vocab = "../spm/spm.wos.vocab"
        umap_model = joblib.load("./models/umap/wos.umap")

    
    dataset_name = train_path.split('/')[2].split('-')[0]
    print('Dataset name:', dataset_name)

    # global variables
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')

    print('reading dataset...')
    train_set, train_label = read_n_encode_dataset(train_path, vectorizer, logprobs, power)
    test_set, test_label = read_n_encode_dataset(train_path.replace('train', 'test'), vectorizer, logprobs, power)
    scaler = preprocessing.MinMaxScaler().fit(train_set.todense())
    train_set = scaler.transform(train_set.todense())
    test_set = scaler.transform(test_set.todense())
    categories = list(set(train_label))
    idx_train = [categories.index(l) for l in train_label]
    idx_test = [categories.index(l) for l in test_label]
    
    train_set = umap_model.transform(train_set)
    test_set = umap_model.transform(test_set)
    PN_SIZE = train_set.shape[1]

    plt.scatter(train_set[:, 0], train_set[:, 1], s= 5, c=idx_train, cmap='Spectral')
    plt.title('Embedding of the '+dataset+' training set by UMAP', fontsize=14);
    plt.savefig(dataset+".umap.train.png")

    plt.scatter(train_set[:, 0], train_set[:, 1], s= 5, c=idx_train, cmap='Spectral')
    plt.title('Embedding of the '+dataset+' test set by UMAP', fontsize=14);
    plt.savefig(dataset+".umap.test.png")
    
    fly = joblib.load('./models/flies/'+dataset+'.fly.m')
    fly.eval_method = 'similarity'
    fly.hyperparameters['num_nns']=100
    print("\nEvaluating on test set with similarity:")
    fly.evaluate(train_set,test_set,train_label,test_label)
    fly.eval_method = 'classification'
    print("\nEvaluating on test set with classification:")
    fly.evaluate(train_set,test_set,train_label,test_label)
