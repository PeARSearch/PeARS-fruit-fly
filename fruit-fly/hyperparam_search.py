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
import re
# import torch
import pathlib
import joblib
import multiprocessing
import sentencepiece as spm
import numpy as np
from datetime import datetime
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

from mkprojections import create_projections
from hash import read_vocab, hash_dataset
from classify import train_model

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def generate_projs(PN_size, KC_size, proj_size, dataset_name):
    d = "models/projection/"+dataset_name+"/kc"+str(KC_size)+"-p"+str(proj_size)
    if not os.path.isdir(d):
        os.mkdir(d)
    trial = len(os.listdir(d))
    model_file = create_projections(PN_size, KC_size, proj_size, d, trial)
    return model_file


def read_n_encode_dataset(path, vectorizer, logprobs):
    # read
    doc_list, label_list = [], []
    doc = ""
    with open(path) as f:
        for l in f:
            l = l.rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*class=([^ ]*)>", l)
                label = m.group(1)
                label_list.append(label)
            elif l[:5] == "</doc":
                doc_list.append(doc)
                doc = ""
            else:
                doc += l + ' '

    # encode
    X = vectorizer.fit_transform(doc_list)
    X = csr_matrix(X)
    X = X.multiply(logprobs)

    return X, label_list


def fruitfly_pipeline(top_word, KC_size, proj_size, percent_hash,
                      C, num_iter, num_trial):
    def _hash_n_train(model_file):
        # print('hashing dataset')
        hash_train = hash_dataset(dataset_mat=train_set, projection_path=model_file,
                                  percent_hash=percent_hash, top_words=top_word)
        hash_val = hash_dataset(dataset_mat=val_set, projection_path=model_file,
                                percent_hash=percent_hash, top_words=top_word)
        # print('training and evaluating')
        val_score, model = train_model(m_train=hash_train, classes_train=train_label,
                                       m_val=hash_val, classes_val=val_label,
                                       C=C, num_iter=num_iter)
        return val_score, model

    print('creating projections')
    model_files = [generate_projs(PN_size, KC_size, proj_size, dataset_name) for _ in range(num_trial)]

    print('training')
    score_list, model_list = [], []
    score_model_list = joblib.Parallel(n_jobs=max_thread, prefer="threads")(
        joblib.delayed(_hash_n_train)(model_file) for model_file in model_files)
    score_list += [i[0] for i in score_model_list]
    model_list += [i[1] for i in score_model_list]

    # select the max performance
    max_idx = np.argmax(score_list)
    trial = model_files[max_idx].split('.')[0].split('_')[1]
    save_name = 'kc' + str(KC_size) + '_proj' + str(proj_size) + '_trial' + str(trial) +\
                '_top' + str(top_word) + '_wta' + str(percent_hash) + '_C' + str(C) + \
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
        f.writelines('\t'.join(str(i) for i in [KC_size, proj_size, top_word,
                                                percent_hash, C, num_iter, avg_score, std_score]))
        f.writelines('\n')

    return avg_score


def optimize_fruitfly(continue_log):
    def _classify(topword, KC_size, proj_size, percent_hash, C):
        topword = round(topword)
        KC_size = round(KC_size)
        proj_size = round(proj_size)
        percent_hash = round(percent_hash)
        C = round(C)
        num_trial = 3
        num_iter = 50
        if dataset_name == '20news':
            num_iter = 2000  # 50 wos wiki, 2000 20news
        print(f'--- KC_size {KC_size}, proj_size {proj_size}, '
              f'top_word {topword}, wta {percent_hash}, C {C} ---')
        return fruitfly_pipeline(topword, KC_size, proj_size, percent_hash,
                                 C, num_iter, num_trial)

    optimizer = BayesianOptimization(
        f=_classify,
        pbounds={"topword": (200, 1000), "KC_size": (8000, 20000),
                 "proj_size": (7, 20), "percent_hash": (15, 40), "C": (50, 200)},
        #random_state=1234,
        verbose=2
    )

    if continue_log:
        load_logs(optimizer, logs=[continue_log])
        print("Optimizer is now aware of {} points.".format(len(optimizer.space)))
    tmp_log_path = f'./log/logs_{dataset_name}_{now}.json'
    logger = JSONLogger(path=tmp_log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(n_iter=500)
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
    print('reading dataset')
    train_set, train_label = read_n_encode_dataset(train_path, vectorizer, logprobs)
    val_set, val_label = read_n_encode_dataset(train_path.replace('train', 'val'), vectorizer, logprobs)
    max_val_score = -1
    max_thread = int(multiprocessing.cpu_count() * 0.7)

    # search
    optimize_fruitfly(continue_log)


