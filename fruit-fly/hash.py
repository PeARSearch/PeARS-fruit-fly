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
from timer import Timer
from docopt import docopt
import sentencepiece as spm
from itertools import combinations
from scipy.sparse import coo_matrix
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('../spmcc.model')

def read_vocab():
    c = 0
    vocab = {}
    reverse_vocab = {}
    logprobs = []
    with open("../spmcc.vocab") as f:
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


def show_projections(hashed_kenyon,reverse_vocab):
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


def wta(layer,percent):
    activations = np.zeros(len(layer))
    top = int(percent * len(layer) / 100)
    activated_cs = np.argpartition(layer, -top)[-top:]
    for cell in activated_cs:
        activations[cell] = layer[cell]
    return activations


def hash_input(vec, reverse_vocab, percent_hash, KC_size, pn_to_kc, projection_functions):
    kenyon_layer = projection(vec, KC_size, pn_to_kc, projection_functions)
    hashed_kenyon = wta(kenyon_layer,percent_hash)
    #show_projections(hashed_kenyon,reverse_vocab)
    return hashed_kenyon


def return_keywords(vec):
    keywords = []
    vs = np.argsort(vec)
    for i in vs[-10:]:
        keywords.append(i)
    return keywords


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
    in_file = in_file_path.replace("datasets/20news-bydate/","")
    params = ".top"+str(top_tokens)+".wta"+str(percent_hash)

    hs_file = os.path.join(d,in_file.replace('.sp',params+'.hs'))
    ID_file = os.path.join(d,in_file.replace('.sp',params+'.ids'))
    class_file = os.path.join(d,in_file.replace('.sp',params+'.cls'))
    keyword_file = os.path.join(d,in_file.replace('.sp',params+'.kwords'))


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
                vec = wta(vec,top_tokens)
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
