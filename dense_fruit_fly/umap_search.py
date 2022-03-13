"""Analysis of PN space
Usage:
  umap_search.py --dataset=<str> 
  umap_search.py (-h | --help)
  umap_search.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<str>              Name of file the dataset, either wiki, 20news, or wos  (processed by sentencepiece)
"""


import umap
import joblib
import sentencepiece as spm
import numpy as np
from datetime import datetime
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack, lil_matrix, coo_matrix
from utils import read_vocab, hash_dataset_, read_n_encode_dataset
from eval import prec_at_k
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def evaluate(logprob_power=5, umap_nns=10, umap_min_dist=0.0, umap_components=16, knn=100, save=False):
    train_set, train_label = read_n_encode_dataset(train_path, vectorizer, logprobs, logprob_power)
    val_set, val_label = read_n_encode_dataset(train_path.replace('train', 'val'), vectorizer, logprobs, logprob_power)
    scaler = preprocessing.MinMaxScaler().fit(train_set.todense())
    train_set = scaler.transform(train_set.todense())
    val_set = scaler.transform(val_set.todense())
    umap_model = umap.UMAP(n_neighbors=umap_nns, min_dist=umap_min_dist, n_components=umap_components, metric='hellinger', random_state=32).fit(train_set)
    #train_set = umap_model.transform(train_set)
    val_set = umap_model.transform(val_set)
    val_set = np.nan_to_num(val_set)

    #print("Prec at k, train:")
    #score = prec_at_k(m=csr_matrix(train_set),classes=train_label,k=knn,metric="cosine")
    #print(score)
    
    print("Prec at k, val:")
    score = prec_at_k(m=csr_matrix(val_set),classes=val_label,k=knn,metric="cosine")
    print(score)
    
    if save:
        filename = './models/umap/'+dataset+'.umap'
        joblib.dump(umap_model, filename)
    return score


def optimize(knn):
    def _evaluate(logprob_power, umap_nns, umap_min_dist, umap_components):
        logprob_power = round(logprob_power)
        umap_nns = round(umap_nns)
        umap_components = round(umap_components)
        print(f'--- power {logprob_power}, umap_nns {umap_nns}, umap_min_dist {umap_min_dist}, umap_components {umap_components}')
        return evaluate(logprob_power, umap_nns, umap_min_dist, umap_components, knn, False)

    optimizer = BayesianOptimization(
        f=_evaluate,
        pbounds={"logprob_power": (3, 7), "umap_nns": (5,200), "umap_min_dist": (0.0,0.2), "umap_components": (2,32) },
        #pbounds={"logprob_power": (6, 7), "umap_nns": (17,18), "umap_min_dist": (0.18,0.2), "umap_components": (16,17) },
        #random_state=1234,
        verbose=2
    )

    tmp_log_path = f'./log/logs_{dataset_name}_umap_{knn}.json'
    logger = JSONLogger(path=tmp_log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=2, n_iter=3)
    print("Final result:", optimizer.max['target'])
    params = optimizer.max['params']
    print(params)
    evaluate(round(params['logprob_power']), round(params['umap_nns']), params['umap_min_dist'], round(params['umap_components']), knn, True)






if __name__ == '__main__':
    args = docopt(__doc__, version='PN analysis, ver 0.1')
    dataset = args["--dataset"]

    if dataset == "wiki":
        train_path="../datasets/wikipedia/wikipedia-train.sp"
        spm_model = "../spm/spm.wiki.model"
        spm_vocab = "../spm/spm.wiki.vocab"
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


    # global variables
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    
    knn=100 #The k for precision at k
    optimize(knn)
