"""Hyper-parameter search by Bayesian optimization
Usage:
  hyperparam_search.py --train_path=<filename> --num_iter=<n>
  hyperparam_search.py (-h | --help)
  hyperparam_search.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<filename>         Name of file to train (processed by sentencepeice)
  --dir=<dir>                     Number of iterations in Bayesian optimization
"""



import glob
import os
import shutil
import re
import pickle
import torch
import sentencepiece as spm
import numpy as np
from docopt import docopt
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import CountVectorizer

from mkprojections import create_projections
from hash import read_vocab, read_projections, projection, wta, hash_input
from classify import prepare_data, train_model

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs


def generate_hash(KC_size, proj_size):
    vocab, reverse_vocab, logprobs = read_vocab()

    PN_size = len(vocab)
    d = "models/kc"+str(KC_size)+"-p"+str(proj_size)
    if not os.path.isdir(d):
        os.mkdir(d)
    trial = len(os.listdir(d))
    model_file = create_projections(PN_size, KC_size, proj_size,d, trial)
    return model_file


def hash(model_file, in_file_path, output_dir, top_tokens, percent_hash):
    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')

    d = model_file
    vocab, reverse_vocab, logprobs = read_vocab()
    projection_functions, pn_to_kc = read_projections(d)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

    # Setting up the fly
    PN_size = len(vocab)
    KC_size = len(projection_functions)
    proj_size = len(projection_functions[0])
    print("SIZES PN LAYER:",PN_size,"KC LAYER:",KC_size)
    print("SIZE OF PROJECTIONS:",proj_size)
    print("SIZE OF FINAL HASH:",percent_hash,"%")

    projection_layer = np.zeros(PN_size)
    kenyon_layer = np.zeros(KC_size)

    #Reading through documents
    n_doc = 0
    doc = ""

    M_data = []
    M_col = []
    M_row = []
    IDs = []
    classes = {}
    keywords = {}

    in_file = in_file_path.split('/')[-1]
    trial = d.split('.')[0].split('_')[1]
    params = '.kc'+str(KC_size) + '.size'+str(proj_size) + '.trial'+str(trial) + ".top"+str(top_tokens)+".wta"+str(percent_hash)

    hs_file = os.path.join(output_dir ,in_file.replace('.sp',params+'.hs')).replace('.projs/', '')
    ID_file = os.path.join(output_dir ,in_file.replace('.sp',params+'.ids')).replace('.projs/', '')
    class_file = os.path.join(output_dir ,in_file.replace('.sp',params+'.cls')).replace('.projs/', '')
    keyword_file = os.path.join(output_dir ,in_file.replace('.sp',params+'.kwords')).replace('.projs/', '')

    with open(in_file_path,'r') as f:
        for l in f:
            l = l.rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*id=([^ ]*) ",l)
                ID=m.group(1)
                m = re.search(".*class=([^ ]*)>",l)
                cl=m.group(1)
                IDs.append(ID+'_'+cl)
                classes[IDs[-1]] = m.group(1)
            #                 print("Processing",IDs[-1])
            elif l[:5] == "</doc":
                #                 print(doc)
                ll = sp.encode_as_pieces(doc)
                X = vectorizer.fit_transform([doc])
                X = X.toarray()[0]
                vec = logprobs * X
                vec = wta(vec,top_tokens)
                hs = hash_input(vec,reverse_vocab,percent_hash, KC_size, pn_to_kc, projection_functions)
                hs = coo_matrix(hs)
                #                 keywords[IDs[-1]] = [reverse_vocab[w] for w in return_keywords(vec)]
                #                 print(keywords[IDs[-1]])
                for i in range(len(hs.data)):
                    M_row.append(n_doc)
                    M_col.append(hs.col[i])
                    M_data.append(hs.data[i])
                doc = ""
                n_doc+=1
                #time.sleep(0.002)    #Sleep a little to consume less CPU
            else:
                doc+=l+' '
    M = coo_matrix((M_data, (M_row, M_col)), shape=(n_doc, KC_size))

    with open(hs_file,"wb") as hsf:
        pickle.dump(M,hsf)
    with open(ID_file,"wb") as IDf:
        pickle.dump(IDs,IDf)
    with open(keyword_file,"wb") as kf:
        pickle.dump(keywords,kf)
    with open(class_file,"wb") as cf:
        pickle.dump(classes,cf)

    return hs_file


def classify(tr_file, lrate, batchsize, epochs, hiddensize, wdecay, n_run_classify):
    tr_file = tr_file.replace('./', '')
    dataset_name = tr_file.split('/')[-1].split('-')[0]

    checkpointsdir = ""
    m_train,classes_train,m_val,classes_val,ids_train,ids_val = prepare_data(tr_file)

    val_scores = []
    for _ in range(n_run_classify):
        val_score = train_model(m_train,classes_train,m_val,classes_val,ids_train,ids_val,hiddensize,lrate,wdecay,batchsize,epochs,checkpointsdir)
        val_scores.append(val_score)
    val_scores = np.array(val_scores)
    with open('./log/results.tsv', 'a') as f:
        tmp = tr_file.split('.')
        kc = tmp[1][2:]
        size = tmp[2][4:]
        trial = tmp[3][5:]
        top = tmp[4][3:]
        wta = tmp[5][3:]
        mean = val_scores.mean()
        std = val_scores.std()
        l = '\t'.join([dataset_name, kc, size, trial, top, wta, str(mean), str(std)])
        f.writelines(l + '\n')

    return mean


def fruitfly_pipeline(train_path, topword, KC_size, proj_size, percent_hash, n_run_classify):
    model_file = generate_hash(KC_size, proj_size)
    shutil.rmtree('./tmp', ignore_errors=True)
    os.mkdir('./tmp')
    print('hashing files')
    hash_file = hash(model_file=model_file, in_file_path=train_path,
                     output_dir='./tmp', top_tokens=topword, percent_hash=percent_hash)
    val_path = train_path.replace('train', 'test')
    hash(model_file=model_file, in_file_path=val_path,
         output_dir='./tmp', top_tokens=topword, percent_hash=percent_hash)
    print('training and evaluating')
    val_score = classify(tr_file=hash_file, lrate=0.0002,
                         batchsize=2048, epochs=1000,
                         hiddensize=100, wdecay=0.0001,
                         n_run_classify=n_run_classify)
    return val_score


def optimize_fruitfly(train_path, num_iter, n_run_classify):
    def classify_val(topword, KC_size, proj_size, percent_hash):
        topword = round(topword)
        KC_size = round(KC_size)
        proj_size = round(proj_size)
        percent_hash = round(percent_hash)
        return fruitfly_pipeline(train_path, topword, KC_size,
                                 proj_size, percent_hash, n_run_classify)

    optimizer = BayesianOptimization(
        f=classify_val,
        pbounds={"topword": (10, 100), "KC_size": (4500, 9000),
                 "proj_size": (1, 5), "percent_hash": (2, 20)},
        random_state=1234,
        verbose=2
    )

    logger = JSONLogger(path="./log/logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    load_logs(optimizer, logs=["./log/old_logs.json"])
    print("Optimizer is now aware of {} points.".format(len(optimizer.space)))

    optimizer.maximize(n_iter=num_iter)

    print("Final result:", optimizer.max)


if __name__ == '__main__':
    args = docopt(__doc__, version='Hyper-parameter search by Bayesian optimization 0.1')
    train_path = args["--train_path"]
    num_iter = int(args["--num_iter"])
    n_run_classify = int(args["--n_run_classify"])
    #torch.set_num_threads(1)
    optimize_fruitfly(train_path, num_iter, n_run_classify)

