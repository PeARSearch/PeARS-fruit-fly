import re
import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import pairwise_distances
from os.path import exists
from hash import read_projections, projection_vectorized, wta_vectorized


def read_vocab(vocab_file):
    c = 0
    vocab = {}
    reverse_vocab = {}
    logprobs = []
    with open(vocab_file) as f:
        for l in f:
            l = l.rstrip('\n')
            wp = l.split('\t')[0]
            logprob = -(float(l.split('\t')[1]))
            #logprob = log(lp + 1.1)
            if wp in vocab or wp == '':
                continue
            vocab[wp] = c
            reverse_vocab[c] = wp
            logprobs.append(logprob**3)
            c+=1
    return vocab, reverse_vocab, logprobs


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


def hash_input_vectorized_(pn_mat, weight_mat, percent_hash):
    kc_mat = pn_mat.dot(weight_mat.T)
    #print(pn_mat.shape,weight_mat.shape,kc_mat.shape)
    kc_use = np.squeeze(kc_mat.toarray().sum(axis=0,keepdims=1))
    kc_use = kc_use / sum(kc_use)
    kc_sorted_ids = np.argsort(kc_use)[:-kc_use.shape[0]-1:-1] #Give sorted list from most to least used KCs
    m, n = kc_mat.shape
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(kc_mat[i: i+2000].toarray(), k=percent_hash)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hashed_kenyon = wta_csr[1:]
    return hashed_kenyon, kc_use, kc_sorted_ids


def hash_dataset_(dataset_mat, weight_mat, percent_hash, top_words):
    m, n = dataset_mat.shape
    dataset_mat = csr_matrix(dataset_mat)
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(dataset_mat[i: i+2000].toarray(), k=top_words, percent=False)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hs, kc_use, kc_sorted_ids = hash_input_vectorized_(wta_csr[1:], weight_mat, percent_hash)
    hs = (hs > 0).astype(np.int_)
    return hs, kc_use, kc_sorted_ids


