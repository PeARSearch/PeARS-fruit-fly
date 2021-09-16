"""Hash a document with best performing fly

Usage:
  hash_with_best_proj.py --docfile=<filename> --flyfolder=<foldername>
  hash_with_best_proj.py (-h | --help)
  hash_with_best_proj.py --version
Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --docfile=<filename>      Name of file containing doc and information about the doc such as URL and label
  --flyfolder=<foldername>  Name of folder where best fly is located

"""

import pickle
import numpy as np
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from evolve_flies import Fly
from hyperparam_search import read_n_encode_dataset
from classify import train_model
from hash import read_vocab
from utils import hash_dataset_
from scipy import sparse
from scipy.sparse import csr_matrix, vstack
import pathlib
from docopt import docopt
import glob
import re
from hash import wta, return_keywords

def hash_a_document(f_dataset, best_fly):

  top_words = 700
  C = 100
  num_iter = 2000  # wikipedia and wos only need 50 steps
  sp = spm.SentencePieceProcessor()
  sp.load('../spmcc.model')
  vocab, reverse_vocab, logprobs = read_vocab()
  vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

  with open(f_dataset,'r') as f:
    doc=""
    for e, l in enumerate(f):
      l = l.rstrip('\n')
      if l[:4] == "<doc":
        m = re.search(".*id=([^ ]*) ",l)
        ID=m.group(1)
        m = re.search(".*class=([^ ]*)>",l)
        lab=m.group(1)
        m = re.search(".*url=([^ ]*) ",l)
        url=m.group(1)
        continue

      if l[:5] != "</doc" and l[:4] != "<doc":
        doc = l 
        continue

      if l[:5] == "</doc" and doc != "":
        ll = sp.encode_as_pieces(doc)
        X = vectorizer.fit_transform([" ".join(ll)])
        X = csr_matrix(X)
        X = X.multiply(logprobs)

        hs = hash_dataset_(dataset_mat=X, weight_mat=best_fly.projection,
                                   percent_hash=best_fly.wta, top_words=top_words)

        vec = wta(X.toarray()[0], top_words, percent=False)
        keywords = [reverse_vocab[w] for w in return_keywords(vec)]

        hs_file='./hashes/'+lab+".hs"
        if hs_file in glob.glob('./hashes/*.hs'):
          hs_matrix = pickle.load(open(hs_file, 'rb'))
          # print(hs_file)
          hs_matrix = vstack([hs_matrix, hs])
          pickle.dump(hs_matrix, open(hs_file, 'wb'))
          ids = pickle.load(open(hs_file.replace(".hs", ".ids"), 'rb'))
          ids.append(ID)
          pickle.dump(ids, open(hs_file.replace(".hs", ".ids"), 'wb'))
          labels = pickle.load(open(hs_file.replace(".hs", ".cls"), 'rb'))
          labels.append(lab)
          pickle.dump(labels, open(hs_file.replace(".hs", ".cls"), 'wb'))
          urls = pickle.load(open(hs_file.replace(".hs", ".url"), 'rb'))
          urls.append(url)
          pickle.dump(urls, open(hs_file.replace(".hs", ".url"), 'wb'))
          keyws = pickle.load(open(hs_file.replace(".hs", ".kwords"), 'rb'))
          keyws.append(keywords)
          pickle.dump(keyws, open(hs_file.replace(".hs", ".kwords"), 'wb'))
        else:
          pickle.dump(hs, open(hs_file, 'wb'))
          pickle.dump([ID], open(hs_file.replace(".hs", ".ids"), 'wb'))
          pickle.dump([lab], open(hs_file.replace(".hs", ".cls"), 'wb'))
          pickle.dump([url], open(hs_file.replace(".hs", ".url"), 'wb'))
          pickle.dump([keywords], open(hs_file.replace(".hs", ".kwords"), 'wb'))
        doc=""
        if e % 50 == 0:
          print(f'{e} lines of the input document processed so far...')
        continue


if __name__ == '__main__':
    args = docopt(__doc__, version='Hashing a document, ver 0.1')

    f_dataset = args['--docfile']
    flyfolder = args['--flyfolder']

    pathlib.Path('./hashes').mkdir(parents=True, exist_ok=True)

    with open(flyfolder+'best_val_score', 'rb') as f:  # modified the name of the fruit-fly here
      best_fly = pickle.load(f)

    hash_a_document(f_dataset, best_fly)