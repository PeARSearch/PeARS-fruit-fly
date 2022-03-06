"""Hyper-parameter search by Bayesian optimization
Usage:
  hyperparam_search.py --dataset=<str> [--continue_log=<filename>] (--random|--store) (--classification|--similarity)
  hyperparam_search.py (-h | --help)
  hyperparam_search.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<str>              Name of file the dataset, either wiki, 20news, or wos  (processed by sentencepeice)
  --random|--store           Initialisation method.
  --classification|--similarity           Eval method.
  [--continue_log=<filename>]     Name of the json log file that we want the Bayesian optimization continues
"""


import os
import re
import pathlib
from joblib import Parallel, delayed,dump
import random
import multiprocessing
import sentencepiece as spm
import numpy as np
from datetime import datetime
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack, lil_matrix, coo_matrix
from bo_utils import read_vocab, hash_dataset_, append_as_json, get_stats

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from fly import Fly

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

def read_projections(proj_size):
    proj_store = []
    with open(proj_store_path+str(proj_size)) as f:
        for l in f:
            ps = l.split(" :: ")[0]
            ps = [int(i) for i in ps.split()]
            proj_store.append(ps)
    return proj_store


def fruitfly_pipeline(pn_size, kc_size, proj_size, wta, top_words, init_method, eval_method, 
                      proj_store, num_trial):

    hyperparameters = {'C':100,'num_iter':200,'num_nns':100}
    fly_list = [Fly(pn_size, kc_size, wta, proj_size, top_words, init_method, eval_method, proj_store, hyperparameters) for _ in range(num_trial)]
    print('evaluating...')
    score_list = []

    with Parallel(n_jobs=max_thread, prefer="threads") as parallel:
        delayed_funcs = [delayed(lambda x:x.evaluate(train_set,val_set,train_label,val_label))(fly) for fly in fly_list]
        score_kcs_list = parallel(delayed_funcs)
    
    score_list = [p[0] for p in score_kcs_list]
    kcs_list = [p[1] for p in score_kcs_list]
    #score_model_list = joblib.Parallel(n_jobs=max_thread, prefer="threads")(
    #    joblib.delayed(_hash_n_train)(fly) for fly in fly_list)
   
    # select the max performance
    #max_idx = np.argmax(score_list)
    #save_name = 'kc' + str(kc_size) + '_proj' + str(proj_size) +\
    #            '_top' + str(top_words) + '_wta' + str(wta) + \
    #            '_score' + str(score_list[max_idx])[2:]  # remove 0.
    #global max_val_score
    #if score_list[max_idx] > max_val_score:
    #    max_val_score = score_list[max_idx]
    #    for f in pathlib.Path(f'./models/classification/{dataset_name}_{now}').glob('*.sav'):
    #        f.unlink()
    #    dump(fly_list[max_idx], f'./models/classification/{dataset_name}_{now}/{save_name}.sav')

    # average the validation acc
    avg_score = np.mean(score_list)
    std_score = np.std(score_list)
    print('average score:', avg_score)

    # write the std
    with open(f'./log/logs_{dataset_name}.tsv', 'a') as f:
        f.writelines('\t'.join(str(i) for i in [kc_size, proj_size, top_words,
                                                wta, avg_score, std_score]))
        f.writelines('\n')

    return avg_score


def optimize_fruitfly(pn_size, init_method, eval_method, continue_log):
    def _evaluate(kc_size, wta, proj_size, top_words):
        top_words = round(top_words)
        kc_size = round(kc_size)
        proj_size = round(proj_size)
        proj_store = None
        if init_method == "store":
            print("PROJ SIZE",proj_size)
            if proj_size % 10 != 0:
                proj_size = int(proj_size / 10) * 10
            proj_store = read_projections(proj_size)
        wta = round(wta)
        num_trial = 3
        print(f'--- pn_size {pn_size}, kc_size {kc_size}, wta {wta}, proj_size {proj_size}, '
                f'top_words {top_words}, init_method {init_method} ---')
        return fruitfly_pipeline(pn_size, kc_size, wta, proj_size, top_words,
                                 init_method, eval_method, proj_store, num_trial)

    optimizer = BayesianOptimization(
        f=_evaluate,
        pbounds={"top_words": (1500,1501), "kc_size": (1000, 2000),
                 "proj_size": (10, 30), "wta": (30, 100) },
        #random_state=1234,
        verbose=2
    )

    if continue_log:
        load_logs(optimizer, logs=[continue_log])
        print("Optimizer is now aware of {} points.".format(len(optimizer.space)))
    tmp_log_path = f'./log/logs_{dataset_name}_{now}_{init_method}_{eval_method}.json'
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
    dataset = args["--dataset"]
    continue_log = args["--continue_log"]
    init_method = "random" if args['--random'] else "store"
    eval_method = "classification" if args['--classification'] else "similarity"


    if dataset == "wiki":
        train_path="../datasets/wikipedia/wikipedia-train.sp"
        spm_model = "../spm/spm.wiki.model"
        spm_vocab = "../spm/spm.wiki.vocab"
        proj_store_path = "../projection_store/projection_data/wiki.projection.store."
    if dataset == "20news":
        train_path="../datasets/20news-bydate/20news-bydate-train.sp"
        spm_model = "../spm/spm.20news.model"
        spm_vocab = "../spm/spm.20news.vocab"
        proj_store_path = "../projection_store/projection_data/20news.projection.store."
    if dataset == "wos":
        train_path="../datasets/wos/wos11967-train.sp"
        spm_model = "../spm/spm.wos.model"
        spm_vocab = "../spm/spm.wos.vocab"
        proj_store_path = "../projection_store/projection_data/wos.projection.store."

    
    dataset_name = train_path.split('/')[2].split('-')[0]
    print('Dataset name:', dataset_name)

    pathlib.Path('./log').mkdir(parents=True, exist_ok=True)
    main_log_path = f'./log/logs_{dataset_name}_{init_method}_{eval_method}.json'
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
    print('reading dataset...')
    train_set, train_label = read_n_encode_dataset(train_path, vectorizer, logprobs)
    val_set, val_label = read_n_encode_dataset(train_path.replace('train', 'val'), vectorizer, logprobs)
    max_val_score = -1
    max_thread = int(multiprocessing.cpu_count() * 0.2)

    # search
    optimize_fruitfly(PN_SIZE,init_method,eval_method,continue_log)


