"""Hyper-parameter search by Bayesian optimization
Usage:
  fly_search.py --dataset=<str>
  fly_search.py (-h | --help)
  fly_search.py --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --dataset=<str>              Name of dataset, either wiki, 20news, or wos.
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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from codecarbon import OfflineEmissionsTracker

from classify import train_model
from utils import read_vocab, hash_dataset_, read_n_encode_dataset
# from fly import Fly


def dim_reduction_pca(X_train, X_val, n_dim):
    pca = PCA(n_components=n_dim)
    pca.fit(X_train)
    X_train_tf = pca.transform(X_train)
    X_val_tf = pca.transform(X_val)
    return X_train_tf, X_val_tf


class FlyPCA:
    def __init__(self, pn_size=None, kc_size=None, wta=None, proj_size=None, dim_reduction_method=None, n_dim_reduction=None,
                 eval_method=None, hyperparameters=None):
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.wta = wta
        self.proj_size = proj_size
        self.eval_method = eval_method
        self.hyperparameters = hyperparameters
        weight_mat, self.shuffled_idx = self.create_projections(proj_size=self.proj_size)
        # weight_mat, self.shuffled_idx = self.create_projections_3(proj_size=self.proj_size, favor_order=std_rank)
        self.projections = lil_matrix(weight_mat)
        self.val_score_c = 0
        self.val_score_s = 0
        self.is_evaluated = False
        self.kc_use_sorted = None
        self.kc_in_hash_sorted = None
        # print("INIT",self.kc_size,self.proj_size,self.wta,self.get_coverage())

        self.dim_reduction_method = dim_reduction_method
        self.n_dim_reduction = n_dim_reduction


    def create_projections(self, proj_size):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        idx = list(range(self.pn_size))
        random.shuffle(idx)
        used_idx = idx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(0, len(idx), proj_size):
                p = idx[i:i + proj_size]
                for j in p:
                    weight_mat[c][j] = 1
                c += 1
                if c >= self.kc_size:
                    break
            random.shuffle(idx)  # reshuffle if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]


    def evaluate(self, train_set, val_set, train_label, val_label):
        def _parallel_eval(hash_train, hash_val, n_dim_reduction):
            hash_train, hash_val = dim_reduction_pca(X_train=hash_train, X_val=hash_val, n_dim=n_dim_reduction)
            hash_train = (hash_train > 0).astype(np.int_)
            hash_val = (hash_val > 0).astype(np.int_)
            if self.eval_method == 'similarity':
                val_score_s, _ = self.prec_at_k(m_val=hash_val, classes_val=val_label_prec,
                                                k=self.hyperparameters['num_nns'])
                return val_score_s
            else:  # classification
                val_score_c, _ = train_model(m_train=hash_train, classes_train=train_label,
                                             m_val=hash_val, classes_val=val_label,
                                             C=self.hyperparameters['C'], num_iter=self.hyperparameters['num_iter'])
                return val_score_c

        # # dim reduction
        # train_set, val_set = dim_reduction(X_train=train_set, X_val=val_set,
        #                                    n_dim=1000, method='pca')

        hash_val, kc_use_val, kc_sorted_val = hash_dataset_(dataset_mat=val_set, weight_mat=self.projections,
                                                            percent_hash=self.wta)
        # if self.eval_method == "classification":
        # We only need the train set for classification, not similarity
        hash_train, kc_use_train, kc_sorted_train = hash_dataset_(dataset_mat=train_set,
                                                                  weight_mat=self.projections,
                                                                  percent_hash=self.wta)

        # dim reduction
        hash_train = hash_train.toarray()
        hash_val = hash_val.toarray()
        # # hash_train, hash_val = dim_reduction(X_train=hash_train, X_val=hash_val,
        # #                                      n_dim=self.n_dim_reduction, method=self.dim_reduction_method)
        # hash_train = (hash_train > 0).astype(np.int_)
        # hash_val = (hash_val > 0).astype(np.int_)
        #
        # # self.val_score_c, _ = train_model(m_train=hash_train, classes_train=train_label,
        # #                                 m_val=hash_val, classes_val=val_label,
        # #                                 C=self.hyperparameters['C'], num_iter=self.hyperparameters['num_iter'])
        # #if self.eval_method == "similarity":
        # self.val_score_s, kc_in_hash_sorted = self.prec_at_k(m_val=hash_val, classes_val=val_label_prec,
        #                                                      k=self.hyperparameters['num_nns'])
        # self.kc_in_hash_sorted = list(kc_in_hash_sorted)
        # self.is_evaluated = True
        # self.kc_use_sorted = list(kc_sorted_val)

        with Parallel(n_jobs=5, prefer="threads") as parallel:
            delayed_funcs = [delayed(_parallel_eval)(hash_train, hash_val, n_dim_reduction) for n_dim_reduction in
                             [8, 16, 32, 64, 128]]
            scores = parallel(delayed_funcs)
        if self.eval_method == 'similarity':
            self.val_score_s = scores
        else:
            self.val_score_c = scores

        return self.val_score_c, self.val_score_s#, self.kc_use_sorted, self.kc_in_hash_sorted
        # return np.random.random(), np.random.random()

    def compute_nearest_neighbours(self, hammings, labels, i, num_nns):
        i_sim = np.array(hammings[i])
        i_label = labels[i]
        ranking = np.argsort(-i_sim)
        neighbours = [labels[n] for n in ranking][1:num_nns + 1]  # don't count first neighbour which is itself
        n_sum = 0
        # print("neighbours: ", neighbours)
        # print("i_label", i_label)
        for n in neighbours:
            for lab in n:
                if lab in i_label:
                    n_sum += 1
                    break
        score = n_sum / num_nns
        return score, neighbours

    def prec_at_k(self, m_val=None, classes_val=None, k=None):
        hammings = 1 - pairwise_distances(m_val, metric="hamming")
        kc_hash_use = np.zeros(m_val.shape[1])
        scores = []
        for i in range(hammings.shape[0]):
            score, neighbours = self.compute_nearest_neighbours(hammings, classes_val, i, k)
            for idx in np.nonzero(m_val[i] == 0)[0]: #m_val[i].indices:
                kc_hash_use[idx] += 1
            scores.append(score)
        kc_hash_use = kc_hash_use / sum(kc_hash_use)
        kc_sorted_hash_use = np.argsort(kc_hash_use)[
                             :-kc_hash_use.shape[0] - 1:-1]  # Give sorted list from most to least used KCs
        return np.mean(scores), kc_sorted_hash_use


def fruitfly_pipeline(kc_size, proj_size, wta, knn, num_trial, C, save):

    #Below parameters are needed to init the fruit fly, even if not used here
    init_method = 'random'
    # eval_method = 'similarity'
    eval_method = 'classification'
    proj_store = None
    hyperparameters = {'C': C, 'num_iter': 200, 'num_nns': knn}

    fly_list = [FlyPCA(pn_size=PN_SIZE, kc_size=kc_size, wta=wta, proj_size=proj_size,
                       eval_method=eval_method, hyperparameters=hyperparameters) for _ in range(num_trial)]
    print('evaluating...')
    scores = []
    best_fly_score = 0.0

    with Parallel(n_jobs=max_thread, prefer="threads") as parallel:
        delayed_funcs = [delayed(lambda x:x.evaluate(train_set,val_set,train_label,val_label))(fly) for fly in fly_list]
        scores = parallel(delayed_funcs)

    if eval_method == 'classification':
        score_list = [p[0] for p in scores]
    else:
        score_list = [p[1] for p in scores]
    score_list = np.array(score_list)
    # print(score_list, score_list.shape)

    # average the validation acc
    avg_score = np.mean(score_list, axis=0)
    std_score = np.std(score_list, axis=0)
    print('average score:', avg_score, 'std:', std_score)

    if save:
        best_score = np.max(score_list)
        best_fly = fly_list[np.argmax(score_list) // len(fly_list)]  # argmax returns flattened index
        filename = './models/flies/'+dataset+'.fly.m'
        joblib.dump(best_fly, filename)

    return avg_score


def optimize_fruitfly():
    knn = 100
    num_trial = 5
    def _evaluate(kc_size, wta, proj_size, C):
        kc_size = round(kc_size)
        proj_size = round(proj_size)
        wta = round(wta)
        print(f'--- kc_size {kc_size}, wta {wta}, proj_size {proj_size}, knn {knn}, C {C} ')
        score_list = fruitfly_pipeline(kc_size=kc_size, proj_size=proj_size, wta=wta,
                                       knn=knn, num_trial=num_trial, C=C, save=False)
        return np.sum(score_list)

    optimizer = BayesianOptimization(
        f=_evaluate,
        pbounds={"kc_size": (512, 15000), "proj_size": (3, 10), "wta": (3, 60), "C": (1, 100)},
        random_state=1234,
        verbose=2
    )

    tmp_log_path = f'./log/bayes_opt/logs_{dataset_name}_fly_{knn}.json'
    logger = JSONLogger(path=tmp_log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    tracker = OfflineEmissionsTracker(project_name='Fruitfly_BO_PCA_' + dataset_name,
                                      country_iso_code="ITA",
                                      measure_power_secs=60*5,
                                      output_dir='./log/emissions_tracker')
    tracker.start()
    optimizer.maximize(init_points=50, n_iter=200)
    tracker.stop()
    print("Final result:", optimizer.max['target'])

    # Saving a fly with the best params
    params = optimizer.max['params']
    print(params)
    score = fruitfly_pipeline(kc_size=round(params['kc_size']), proj_size=round(params['proj_size']),
                              wta=round(params['wta']), knn=knn, num_trial=num_trial, C=params['C'], save=True)
    print("Best fly score:", score, ". Fly saved.")
    #score, size = fly.prune(train_set,val_set,train_label,val_label)
    #print("Score and size after pruning, saved fly:",score, size)


if __name__ == '__main__':
    args = docopt(__doc__, version='Hyper-parameter search by Bayesian optimization, ver 0.1')
    dataset = args["--dataset"]
    power = 5

    if dataset == "wiki" or dataset == "enwiki":
        train_path = "../datasets/wikipedia/wikipedia-train.sp"
        spm_model = "../spm/spm.enwiki.model"
        spm_vocab = "../spm/spm.enwiki.vocab"
        # umap_model = joblib.load("./models/umap/wiki.umap")
    if dataset == "20news":
        train_path = "../datasets/20news-bydate/20news-bydate-train.sp"
        spm_model = "../spm/spm.20news.model"
        spm_vocab = "../spm/spm.20news.vocab"
        # umap_model = joblib.load("./models/umap/20news.umap")
    if dataset == "wos":
        train_path = "../datasets/wos/wos11967-train.sp"
        spm_model = "../spm/spm.wos.model"
        spm_vocab = "../spm/spm.wos.vocab"
        # umap_model = joblib.load("./models/umap/wos.umap")
    if dataset == "tmc":
        train_path = "../datasets/tmc/tmc-train.sp"
        spm_model = "../spm/spm.tmc.model"
        spm_vocab = "../spm/spm.tmc.vocab"
    if dataset == "reuters":
        train_path = "../datasets/reuters/reuters-train.sp"
        spm_model = "../spm/spm.reuters.model"
        spm_vocab = "../spm/spm.reuters.vocab"
    if dataset == "agnews":
        train_path = "../datasets/agnews/agnews-train.sp"
        spm_model = "../spm/spm.agnews.model"
        spm_vocab = "../spm/spm.agnews.vocab"

    dataset_name = train_path.split('/')[2].split('-')[0]
    print('Dataset name:', dataset_name)

    pathlib.Path('./log/bayes_opt').mkdir(parents=True, exist_ok=True)
    pathlib.Path('./log/emissions_tracker').mkdir(parents=True, exist_ok=True)

    # global variables
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')

    print('reading dataset...')
    train_set, train_label = read_n_encode_dataset(train_path, vectorizer, logprobs, power)
    val_set, val_label = read_n_encode_dataset(train_path.replace('train', 'val'), vectorizer, logprobs, power)
    val_label_prec = val_label

    if dataset == "tmc" or dataset == "reuters":
        labels = set()
        for split in [train_label, val_label]:
            for i in split:
                for lab in i:
                    labels.add(lab)
        onehotencoder = MultiLabelBinarizer(classes=sorted(labels))
        train_label = onehotencoder.fit_transform(train_label)  # .tolist()
        val_label = onehotencoder.fit_transform(val_label)  # .tolist()

    scaler = preprocessing.Normalizer(norm='l2').fit(train_set.todense())
    train_set = scaler.transform(train_set.todense())
    val_set = scaler.transform(val_set.todense())

    PN_SIZE = train_set.shape[1]

    max_thread = int(multiprocessing.cpu_count() * 0.25)

    # search
    optimize_fruitfly()
