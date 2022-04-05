import re
import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from os.path import exists


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
            logprobs.append(logprob)
            c+=1
    return vocab, reverse_vocab, logprobs

def read_projections(proj_size):
    proj_store = []
    with open(proj_store_path+str(proj_size)) as f:
        for l in f:
            ps = l.split(" :: ")[0]
            ps = [int(i) for i in ps.split()]
            proj_store.append(ps)
    return proj_store


def projection_vectorized(projection_mat, projection_functions):
    KC_size = len(projection_functions)
    PN_size = projection_mat.shape[1]
    weight_mat = np.zeros((KC_size, PN_size))
    for kc_idx, pn_list in projection_functions.items():
        weight_mat[kc_idx][pn_list] = 1
    weight_mat = coo_matrix(weight_mat.T)
    return projection_mat.dot(weight_mat)


def wta_vectorized(feature_mat, k, percent=True):
    # thanks https://stackoverflow.com/a/59405060

    m, n = feature_mat.shape
    if percent:
        k = int(k * n / 100)
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(feature_mat, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = feature_mat[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    is_smaller_than_kth = feature_mat < kth_vals[:, None]
    # replace mask by 0
    feature_mat[is_smaller_than_kth] = 0
    return feature_mat

def encode_docs(doc_list, vectorizer, logprobs, power):
    logprobs = np.array([logprob ** power for logprob in logprobs])
    X = vectorizer.fit_transform(doc_list)
    X = csr_matrix(X)
    X = X.multiply(logprobs)
    return X

def read_n_encode_dataset(path, vectorizer, logprobs, power):
    # read
    stopwords = ['of', 'in', 'and', 'the', 'at', 'from', 'by', 'with', 'for', 'to', 'de', 'a']
    doc_list, title_list, label_list = [], [], []
    doc = ""
    with open(path) as f:
        for l in f:
            l = l.rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*title=\"([^\"]*)\"", l)
                title = m.group(1).lower()
                title_list.append(title)
                m = re.search(".*categories=\"([^\"]*)\"", l)
                categories = m.group(1).replace('|',' ').lower()
                keywords = title.split()+categories.split()
                keywords = [k for k in keywords if k not in stopwords]
                label_list.append(' '.join(keywords))
            elif l[:5] == "</doc":
                doc_list.append(doc)
                doc = ""
            else:
                doc += l + ' '

    # encode
    X = encode_docs(doc_list, vectorizer, logprobs, power)
    return X, title_list, label_list


def write_as_json(dic, f):
    output_file = open(f, 'w', encoding='utf-8')
    json.dump(dic, output_file)


def append_as_json(dic, f):
    output_file = open(f, 'a', encoding='utf-8')
    json.dump(dic, output_file)
    output_file.write("\n")


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

def hamming_cdist(matrix, vector):
    #Compute the hamming distances between each row of matrix and vector.
    v = vector.reshape(1, -1)
    return cdist(matrix, v, 'hamming').reshape(-1)
