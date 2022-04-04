"""Common Crawl hashing - creating hashes for documents

Usage:
  hash.py --file=<filename> --dir=<dir> --topwords=<n> --wta=<n>
  hash.py (-h | --help)
  hash.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --file=<filename>               Name of file to process
  --dir=<dir>                     Directory containing projections
  --topwords=<n>                  Percentage of document tokens to retain
  --wta=<n>                       Percentage of KCs to retain

"""

import os
import re
import random
import pickle
import numpy as np
import time
import pathlib
from docopt import docopt
import sentencepiece as spm
from scipy.sparse import coo_matrix, csr_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer


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



def read_projections(d):
    c = 0
    projections = {}
    pn_to_kc = {}
    with open(d) as f:
        for l in f:
            l=l.rstrip('\n')
            p = np.array([int(n) for n in l.split()])
            projections[c]=p
            for pn in p:
                if pn in pn_to_kc:
                    pn_to_kc[pn].append(c)
                else:
                    pn_to_kc[pn] = [c]
            c+=1
    return projections, pn_to_kc


def show_projections(hashed_kenyon, reverse_vocab):
    important_words = {}
    for i in range(len(hashed_kenyon)):
        if hashed_kenyon[i] == 1:
            activated_pns = projection_functions[i]
            print([reverse_vocab[pn] for pn in activated_pns])
            for pn in activated_pns:
                w = reverse_vocab[pn]
                if w in important_words:
                    important_words[w]+=1
                else:
                    important_words[w]=1
    print("BEST PNS", sorted(important_words, key=important_words.get, reverse=True)[:proj_size])


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


def hash_input_vectorized(projection_mat, percent_hash, projection_functions):
    kc_mat = projection_vectorized(projection_mat, projection_functions)
    m, n = kc_mat.shape
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(kc_mat[i: i+2000].toarray(), k=percent_hash)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hashed_kenyon = wta_csr[1:]
    return hashed_kenyon


def hash_dataset(dataset_mat, projection_path, percent_hash, top_words):
    # read projection file
    projection_functions, pn_to_kc = read_projections(projection_path)

    # hash
    m, n = dataset_mat.shape
    dataset_mat = csr_matrix(dataset_mat)
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(dataset_mat[i: i+2000].toarray(), k=top_words, percent=False)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hs = hash_input_vectorized(wta_csr[1:], percent_hash, projection_functions)
    hs = (hs > 0).astype(np.int_)

    return hs


def return_keywords(vec):
    keywords = []
    vs = np.argsort(vec)
    for i in vs[-10:]:
        keywords.append(i)
    return keywords


###################################### old version ################################
def projection(projection_layer, KC_size, pn_to_kc, projection_functions):
    kenyon_layer = np.zeros(KC_size)
    nzs = np.where(projection_layer > 0)
    kcs = []
    #print(nzs[0])
    for pn in nzs[0]:
        if pn in pn_to_kc:
            kcs.extend(pn_to_kc[pn])
        #else:
        #    print("WARNING: ",pn,"not in use")
    kcs = list(set(kcs))
    for cell in kcs:
        activated_pns = projection_functions[cell]
        for pn in activated_pns:
            kenyon_layer[cell]+=projection_layer[pn]
    return kenyon_layer


def wta(layer, top, percent=True):
    activations = np.zeros(len(layer))
    if percent:
        top = int(top * len(layer) / 100)
    activated_cs = np.argpartition(layer, -top)[-top:]
    for cell in activated_cs:
        activations[cell] = layer[cell]
    return activations


def hash_input(vec, reverse_vocab, percent_hash, KC_size, pn_to_kc, projection_functions):
    kenyon_layer = projection(vec, KC_size, pn_to_kc, projection_functions)
    hashed_kenyon = wta(kenyon_layer, percent_hash)
    #show_projections(hashed_kenyon,reverse_vocab)
    return hashed_kenyon


if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Hashing 0.1')
    d = args["--dir"]
    top_tokens = int(args["--topwords"])
    percent_hash = int(args["--wta"])

    t = Timer()
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

    in_file_path = args["--file"]
    in_file = in_file_path.split('/')[-1]
    trial = d.split('.')[0].split('_')[1]
    params = '.kc'+str(KC_size) + '.size'+str(proj_size) + '.trial'+str(trial) + ".top"+str(top_tokens)+".wta"+str(percent_hash)

    pathlib.Path('./tmp').mkdir(parents=True, exist_ok=True)
    hs_file = os.path.join('tmp', in_file.replace('.sp',params+'.hs')).replace('.projs/', '')
    ID_file = os.path.join('tmp', in_file.replace('.sp',params+'.ids')).replace('.projs/', '')
    class_file = os.path.join('tmp', in_file.replace('.sp',params+'.cls')).replace('.projs/', '')
    keyword_file = os.path.join('tmp', in_file.replace('.sp',params+'.kwords')).replace('.projs/', '')


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
                print("Processing",IDs[-1])
            elif l[:5] == "</doc":
                #print("Wordpiecing...")
                #t.start()
                ll = sp.encode_as_pieces(doc)
                #t.stop()
                #print("Vectorizing...")
                #t.start()
                X = vectorizer.fit_transform([doc])
                #t.stop()
                X = X.toarray()[0]
                vec = logprobs * X
                vec = wta(vec, top_tokens, percent=False)
                #print("Hashing...")
                #t.start()
                hs = hash_input(vec,reverse_vocab,percent_hash, KC_size, pn_to_kc, projection_functions)
                #t.stop()
                hs = coo_matrix(hs)
                #print(IDs[-1],' '.join([str(i) for i in hs.col]))
                keywords[IDs[-1]] = [reverse_vocab[w] for w in return_keywords(vec)]
                print(keywords[IDs[-1]])
                for i in range(len(hs.data)):
                    M_row.append(n_doc)
                    M_col.append(hs.col[i])
                    M_data.append(hs.data[i])
                    #M_data.append(1)
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
