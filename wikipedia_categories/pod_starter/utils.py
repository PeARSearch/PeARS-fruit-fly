import numpy as np
from scipy.sparse import csr_matrix, vstack

def read_vocab():
    c = 0
    vocab = {}
    reverse_vocab = {}
    logprobs = []
    with open("../../spmcc.vocab") as f:
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

def wta(layer, top, percent=True):
    activations = np.zeros(len(layer))
    if percent:
        top = int(top * len(layer) / 100)
    activated_cs = np.argpartition(layer, -top)[-top:]
    for cell in activated_cs:
        activations[cell] = layer[cell]
    return activations

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

def return_keywords(vec):
    keywords = []
    vs = np.argsort(vec)
    for i in vs[-10:]:
        keywords.append(i)
    return keywords

def hash_input_vectorized_(pn_mat, weight_mat, percent_hash):
    kc_mat = pn_mat.dot(weight_mat.T)
    m, n = kc_mat.shape
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(kc_mat[i: i+2000].toarray(), k=percent_hash)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hashed_kenyon = wta_csr[1:]
    return hashed_kenyon

def hash_dataset_(dataset_mat, weight_mat, percent_hash, top_words):
    m, n = dataset_mat.shape
    dataset_mat = csr_matrix(dataset_mat)
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(dataset_mat[i: i+2000].toarray(), k=top_words, percent=False)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hs = hash_input_vectorized_(wta_csr[1:], weight_mat, percent_hash)
    return hs

