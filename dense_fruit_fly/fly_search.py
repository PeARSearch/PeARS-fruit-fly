"""Hyper-parameter search by Bayesian optimization
Usage:
  fly_search.py --dataset=<str> --logprob=<n>
  fly_search.py (-h | --help)
  fly_search.py --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --dataset=<str>              Name of dataset, either wiki, 20news, or wos.
  --logprob=<n>                Power of word log probabilities when vectorizing documents.
"""


import os
import re
import pathlib
import joblib
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


def fruitfly_pipeline(kc_size, proj_size, wta, knn, num_trial, save):

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
    
    best_score = max(score_list)
    best_fly = fly_list[score_list.index(best_score)]
    if save:
        filename = './models/flies/'+dataset+'.fly.m'
        joblib.dump(best_fly, filename)
    return avg_score


def optimize_fruitfly():
    knn=100
    num_trial = 10
    def _evaluate(kc_size, wta, proj_size):
        kc_size = round(kc_size)
        proj_size = round(proj_size)
        wta = round(wta)
        print(f'--- kc_size {kc_size}, wta {wta}, proj_size {proj_size}, knn {knn} ')
        return fruitfly_pipeline(kc_size, wta, proj_size, knn, num_trial, False)

    optimizer = BayesianOptimization(
        f=_evaluate,
        pbounds={"kc_size": (512, 1024),"proj_size": (3, 10), "wta": (3, 10) },
        #random_state=1234,
        verbose=2
    )


    tmp_log_path = f'./log/logs_{dataset_name}_fly_{knn}.json'
    logger = JSONLogger(path=tmp_log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=2, n_iter=10)
    print("Final result:", optimizer.max['target'])
    
    #Saving a fly with the best params
    params = optimizer.max['params']
    print(params)
    hyperparameters = {'C':100,'num_iter':200,'num_nns':knn}
    score = fruitfly_pipeline(round(params['kc_size']), round(params['wta']), round(params['proj_size']), knn, num_trial, True)
    print("Best fly score:",score, ". Fly saved.")
    #score, size = fly.prune(train_set,val_set,train_label,val_label)
    #print("Score and size after pruning, saved fly:",score, size)

    



if __name__ == '__main__':
    args = docopt(__doc__, version='Hyper-parameter search by Bayesian optimization, ver 0.1')
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

    pathlib.Path('./log').mkdir(parents=True, exist_ok=True)

    # global variables
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')

    print('reading dataset...')
    train_set, train_label = read_n_encode_dataset(train_path, vectorizer, logprobs, power)
    val_set, val_label = read_n_encode_dataset(train_path.replace('train', 'val'), vectorizer, logprobs, power)
    scaler = preprocessing.MinMaxScaler().fit(train_set.todense())
    train_set = scaler.transform(train_set.todense())
    val_set = scaler.transform(val_set.todense())
    train_set = umap_model.transform(train_set)
    val_set = umap_model.transform(val_set)
    PN_SIZE = train_set.shape[1]

    max_thread = int(multiprocessing.cpu_count() * 0.2)

    # search
    optimize_fruitfly()
