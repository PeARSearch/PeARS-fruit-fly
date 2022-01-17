"""Hyper-parameter search by Bayesian optimization
Usage:
  hyperparam_search.py --dataset=<str> [--continue_log=<filename>]
  hyperparam_search.py (-h | --help)
  hyperparam_search.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<str>              Name of file the dataset, either wiki, 20news, or wos  (processed by sentencepeice)
  [--continue_log=<filename>]     Name of the json log file that we want the Bayesian optimization continues
"""


import os
import re
# import torch
import pathlib
import joblib
import multiprocessing
import sentencepiece as spm
import numpy as np
import random
from scipy.sparse import lil_matrix
from scipy.stats import truncnorm
from datetime import datetime
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer

from classify import train_model
from utils import read_vocab, read_n_encode_dataset, hash_dataset_
# from evolve_on_budget import Fly

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from test_add_layer import Fly, hash_multi_layer


def fruitfly_pipeline(top_word, KC_size_0, KC_size_1, proj_size_0, proj_size_1,
                      num_nonzero_0, num_nonzero_1, C, num_iter, num_trial):
    def _hash_n_train(fly):
        hash_train = hash_multi_layer(dataset_mat=train_set,
                                      weight_list=fly.projection_list,
                                      nonzero_list=fly.num_nonzero_list,
                                      top_words=top_word)
        hash_val = hash_multi_layer(dataset_mat=val_set,
                                    weight_list=fly.projection_list,
                                    nonzero_list=fly.num_nonzero_list,
                                    top_words=top_word)
        val_score, model = train_model(m_train=hash_train[-1], classes_train=train_label,
                                       m_val=hash_val[-1], classes_val=val_label,
                                       C=C, num_iter=num_iter)
        return val_score, model

    print('creating projections')
    fly_list = [Fly(layer_size_list=[PN_SIZE, KC_size_0, KC_size_1],
                    num_proj_list=[proj_size_0, proj_size_1],
                    num_nonzero_list=[num_nonzero_0, num_nonzero_1],
                    n_dataset=1,
                    ) for _ in range(num_trial)]

    print('training')
    score_list, model_list = [], []
    score_model_list = joblib.Parallel(n_jobs=max_thread, prefer="threads")(
        joblib.delayed(_hash_n_train)(fly) for fly in fly_list)
    score_list += [i[0] for i in score_model_list]
    model_list += [i[1] for i in score_model_list]

    # select the max performance
    max_idx = np.argmax(score_list)
    save_name = 'kc0' + str(KC_size_0) + 'kc1' + str(KC_size_1) + '_proj0' + str(proj_size_0) + '_proj1' + str(proj_size_1) +\
                '_top' + str(top_word) + '_nonzero0' + str(num_nonzero_0) + '_nonzero1' + str(num_nonzero_1) + '_C' + str(C) + \
                '_iter' + str(num_iter) + '_score' + str(score_list[max_idx])[2:]  # remove 0.
    global max_val_score
    if score_list[max_idx] > max_val_score:
        max_val_score = score_list[max_idx]
        for f in pathlib.Path(f'./models/classification/{dataset_name}_{now}').glob('*.sav'):
            f.unlink()
        joblib.dump(model_list[max_idx], f'./models/classification/{dataset_name}_{now}/{save_name}.sav')

    # average the validation acc
    avg_score = np.mean(score_list)
    std_score = np.std(score_list)
    print('average score:', avg_score)

    # write the std
    with open(f'./log/logs_{dataset_name}.tsv', 'a') as f:
        f.writelines('\t'.join(str(i) for i in [KC_size_0, KC_size_1, proj_size_0, proj_size_1, top_word,
                                                num_nonzero_0, num_nonzero_1, C, num_iter, avg_score, std_score]))
        f.writelines('\n')

    return avg_score


def optimize_fruitfly(continue_log):
    def _classify(topword, KC_size_0, KC_size_1, proj_size_0, proj_size_1, C):
        topword = round(topword)
        KC_size_0, KC_size_1 = round(KC_size_0), round(KC_size_1)
        proj_size_0, proj_size_1 = round(proj_size_0), round(proj_size_1)
        num_iter = 2000
        num_trial = 3
        num_nonzero_0, num_nonzero_1 = 300, 300

        print(f'--- KC_size_0 {KC_size_0}, KC_size_1 {KC_size_1}, proj_size_0 {proj_size_0}, proj_size_1 {proj_size_1}, '
              f'top_word {topword}, C {C}, num_iter {num_iter} ---')
        return fruitfly_pipeline(topword, KC_size_0, KC_size_1, proj_size_0, proj_size_1, num_nonzero_0, num_nonzero_1,
                                 C, num_iter, num_trial)

    optimizer = BayesianOptimization(
        f=_classify,
        pbounds={"topword": (100, 500), "KC_size_0": (300, 15000), "KC_size_1": (300, 15000),
                 "proj_size_0": (2, 20), "proj_size_1": (2, 20), "C": (1, 100)
                 # "percent_hash": (5, 20),
                 # 'C':(C), 'num_iter':(num_iter)
                 },
        # random_state=1234,
        verbose=2
    )

    if continue_log:
        load_logs(optimizer, logs=[continue_log])
        print("Optimizer is now aware of {} points.".format(len(optimizer.space)))
    tmp_log_path = f'./log/logs_{dataset_name}_{now}.json'
    logger = JSONLogger(path=tmp_log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(n_iter=400, init_points=100)
    print("Final result:", optimizer.max)
    with open(main_log_path, 'a') as f_main:
        with open(tmp_log_path) as f_tmp:
            tmp_log = f_tmp.read()
            f_main.write(tmp_log)


if __name__ == '__main__':
    args = docopt(__doc__, version='Hyper-parameter search by Bayesian optimization, ver 0.1')
    dataset = args["--dataset"]
    continue_log = args["--continue_log"]
    # num_iter = args["--num_iter"]
    # C = args["--C"]

    if dataset == "wiki":
        train_path="../datasets/wikipedia/wikipedia-train.sp"
        spm_model = "../spm/spm.wikipedia.model"
        spm_vocab = "../spm/spm.wikipedia.vocab"
    if dataset == "20news":
        train_path="../datasets/20news-bydate/20news-bydate-train.sp"
        spm_model = "../spm/spm.20news.model"
        spm_vocab = "../spm/spm.20news.vocab"
    if dataset == "wos":
        train_path="../datasets/wos/wos11967-train.sp"
        spm_model = "../spm/spm.wos.model"
        spm_vocab = "../spm/spm.wos.vocab"

    dataset_name = train_path.split('/')[2].split('-')[0]
    print('Dataset name:', dataset_name)

    pathlib.Path('./log').mkdir(parents=True, exist_ok=True)
    main_log_path = f'./log/logs_{dataset_name}.json'
    pathlib.Path(main_log_path).touch(exist_ok=True)
    pathlib.Path(f'./models/projection/{dataset_name}').mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pathlib.Path(f'./models/classification/{dataset_name}_{now}').mkdir(parents=True, exist_ok=True)

    # global variables
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    PN_SIZE = len(vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    print('reading dataset')
    train_set, train_label = read_n_encode_dataset(train_path, vectorizer, logprobs)
    val_set, val_label = read_n_encode_dataset(train_path.replace('train', 'val'), vectorizer, logprobs)
    max_val_score = -1
    max_thread = int(multiprocessing.cpu_count() * 0.2)

    # search
    optimize_fruitfly(continue_log)


