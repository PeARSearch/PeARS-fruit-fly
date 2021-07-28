"""Hyper-parameter search by Bayesian optimization
Usage:
  hyperparam_search.py --train_path=<filename> [--continue_log=<filename>]
  hyperparam_search.py (-h | --help)
  hyperparam_search.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<filename>         Name of file to train (processed by sentencepeice)
  [--continue_log=<filename>]     Name of the json log file that we want the Bayesian optimization continues
"""


import os
# import torch
import pathlib
import joblib
import sentencepiece as spm
import numpy as np
from datetime import datetime
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer

from mkprojections import create_projections
from hash import read_vocab, hash_dataset, read_n_encode_dataset
from classify import train_model

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def generate_projs(KC_size, proj_size):
    d = "models/projection/"+dataset_name+"/kc"+str(KC_size)+"-p"+str(proj_size)
    if not os.path.isdir(d):
        os.mkdir(d)
    trial = len(os.listdir(d))
    model_file = create_projections(PN_size, KC_size, proj_size, d, trial)
    return model_file


def fruitfly_pipeline(top_word, KC_size, proj_size, percent_hash,
                      C, num_iter, max_val_score):
    model_file = generate_projs(KC_size, proj_size)

    print('hashing dataset')
    hash_train = hash_dataset(dataset_mat=train_set, projection_path=model_file,
                              percent_hash=percent_hash, top_words=top_word)
    hash_val = hash_dataset(dataset_mat=val_set, projection_path=model_file,
                            percent_hash=percent_hash, top_words=top_word)

    print('training and evaluating')
    val_score, model = train_model(m_train=hash_train, classes_train=train_label,
                                      m_val=hash_val, classes_val=val_label,
                                      C=C, num_iter=num_iter)
    trial = model_file.split('.')[0].split('_')[1]
    save_name = 'kc' + str(KC_size) + '_proj' + str(proj_size) + '_trial' + str(trial) +\
                '_top' + str(top_word) + '_wta' + str(percent_hash) + '_C' + str(C) + \
                '_iter' + str(num_iter) + '_score' + str(val_score)[2:]  # remove 0.
    if max_val_score:
        if val_score > max_val_score['target']:
            for f in pathlib.Path(f'./models/classification/{dataset_name}_{now}').glob('*.sav'):
                f.unlink()
            joblib.dump(model, f'./models/classification/{dataset_name}_{now}/{save_name}.sav')
    else:  # the first run does not return the max score
        joblib.dump(model, f'./models/classification/{dataset_name}_{now}/{save_name}.sav')

    return val_score


def optimize_fruitfly(continue_log):
    def _classify(topword, KC_size, proj_size, percent_hash, C):
        topword = round(topword)
        KC_size = round(KC_size)
        proj_size = round(proj_size)
        percent_hash = round(percent_hash)
        C = round(C)
        num_iter = 10
        return fruitfly_pipeline(topword, KC_size, proj_size, percent_hash,
                                 C, num_iter, optimizer.max)

    optimizer = BayesianOptimization(
        f=_classify,
        pbounds={"topword": (10, 250), "KC_size": (3000, 9000),
                 "proj_size": (2, 10), "percent_hash": (2, 20), "C": (1, 100)},
        #random_state=1234,
        verbose=2
    )

    if continue_log:
        load_logs(optimizer, logs=[continue_log])
        print("Optimizer is now aware of {} points.".format(len(optimizer.space)))
    tmp_log_path = f'./log/logs_{dataset_name}_{now}.json'
    logger = JSONLogger(path=tmp_log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=2, n_iter=2)
    print("Final result:", optimizer.max)
    with open(main_log_path, 'a') as f_main:
        with open(tmp_log_path) as f_tmp:
            tmp_log = f_tmp.read()
            f_main.write(tmp_log)


if __name__ == '__main__':
    args = docopt(__doc__, version='Hyper-parameter search by Bayesian optimization, ver 0.1')
    train_path = args["--train_path"]
    continue_log = args["--continue_log"]
    dataset_name = train_path.split('/')[2].split('-')[0]
    print('Dataset name:', dataset_name)
    print('reading dataset')
    pathlib.Path('./log').mkdir(parents=True, exist_ok=True)
    main_log_path = f'./log/logs_{dataset_name}.json'
    pathlib.Path(main_log_path).touch(exist_ok=True)
    pathlib.Path(f'./models/projection/{dataset_name}').mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pathlib.Path(f'./models/classification/{dataset_name}_{now}').mkdir(parents=True, exist_ok=True)

    # global variables
    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')
    vocab, reverse_vocab, logprobs = read_vocab()
    PN_size = len(vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
    train_set, train_label = read_n_encode_dataset(train_path)
    val_set, val_label = read_n_encode_dataset(train_path.replace('train', 'val'))

    # search
    optimize_fruitfly(continue_log)


