"""Fruit Fly hashing - creating hashes for documents in the 20 newgroup corpus

Usage:
  wordpiece.py
  wordpiece.py (-h | --help)
  wordpiece.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.

"""

import os
from os.path import join
import re
import random
import numpy as np
from docopt import docopt
import sentencepiece as spm


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


def output_wordpieces(train=True):
    if train:
        out_file = open("./20news-bydate/20news-bydate-train.sp", 'w')
        base_dir = "./20news-bydate/20news-bydate-train"
    else:
        out_file = open("./20news-bydate/20news-bydate-test.sp", 'w')
        base_dir = "./20news-bydate/20news-bydate-test"

    # get folders in 20_newsgroup corpus
    folders = os.listdir(base_dir)
    print(folders)

    for folder in folders:
        d = join(base_dir,folder)
        file_ids = os.listdir(d)
        files = [join(d,file_id) for file_id in file_ids]

        for i in range(len(files)):
            in_file = files[i]
            doc = ""
            with open(in_file, encoding="utf8", errors='ignore') as f:
                for l in f:
                    #Ignore headers
                    words = l.split()
                    if len(words) > 0 and words[0][-1] != ':':
                        doc+=l+'\n'
            ll = sp.encode_as_pieces(doc)
            out_file.write("<doc id="+file_ids[i]+" class="+folder+">\n")
            out_file.write(' '.join([wp for wp in ll])+'\n')
            out_file.write("</doc>\n")
    out_file.close()



if __name__ == '__main__':
    args = docopt(__doc__, version='Fruit Fly Hashing, sentencepiece 0.1')

    vocab, reverse_vocab, logprobs = read_vocab()
    print("Computing wordpieces for training set...")
    output_wordpieces(train=True)
    print("Computing wordpieces for test set...")
    output_wordpieces(train=False)
