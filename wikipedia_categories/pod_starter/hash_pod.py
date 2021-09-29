"""Hash base pod with selected fly

Usage:
  hash_pod.py --docfile=<filename> --fly=<path>
  hash_pod.py (-h | --help)
  hash_pod.py --version
Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --docfile=<path>          Path of file containing documents and information about each doc such as URL and label.
  --fly=<path>              Path to selected (deployed) fly model.

"""

import pickle
import numpy as np
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from utils import read_vocab, wta, return_keywords
from utils import hash_dataset_
from scipy import sparse
from scipy.sparse import csr_matrix, vstack
import pathlib
from docopt import docopt
import glob
import re

class DeployedFly:
    def __init__(self):
        self.kc_size = None
        self.wta = None
        self.projection = None
        self.val_scores = []


def hash_documents(f_dataset, best_fly):

  top_words = 250
  sp = spm.SentencePieceProcessor()
  sp.load('../../spmcc.model')
  vocab, reverse_vocab, logprobs = read_vocab()
  vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
  new_ids, new_labels, new_urls, new_keywords, hashes= [np.array([])]*5
  doc=""
  with open(f_dataset,'r') as f:
    for l in f:
      l = l.rstrip('\n')
      if l[:4] == "<doc":
        m = re.search(".*id=([^ ]*) ",l)
        ID=m.group(1)
        new_ids = np.append(new_ids, ID)
        m = re.search(".*class=([^ ]*)>",l)
        lab=m.group(1)
        new_labels = np.append(new_labels, lab)
        m = re.search(".*url=([^ ]*) ",l)
        url=m.group(1)
        new_urls = np.append(new_urls, url)
        continue

      if l[:5] != "</doc" and l[:4] != "<doc":
        doc = l 
        continue

      if l[:5] == "</doc" and doc != "":
        ll = sp.encode_as_pieces(doc)
        X = vectorizer.fit_transform([" ".join(ll)])
        X = csr_matrix(X)
        X = X.multiply(logprobs)
        hashes = np.append(hashes, X)
        vec = wta(X.toarray()[0], top_words, percent=False)
        kwds = [reverse_vocab[w] for w in return_keywords(vec)]
        new_keywords = np.append(new_keywords, kwds)
        doc=""
        continue

  print("Start hashing...")
  new_hs_mat = hash_dataset_(dataset_mat=vstack(hashes), weight_mat=best_fly.projection,
                     percent_hash=best_fly.wta, top_words=top_words)
  print("Saving information...")
  lab = new_labels[0] #all labels should be the same

  hs_file='./hashes/'+lab+".hs"
  if hs_file in glob.glob('./hashes/*.hs'):
      hs_mat = pickle.load(open(hs_file, 'rb'))
      hs_matrix = vstack([hs_mat, new_hs_mat])
      pickle.dump(hs_mat, open(hs_file, 'wb'))
      ids = pickle.load(open(hs_file.replace(".hs", ".ids"), 'rb'))
      ids = np.append(ids, new_ids)
      pickle.dump(ids, open(hs_file.replace(".hs", ".ids"), 'wb'))
      labels = pickle.load(open(hs_file.replace(".hs", ".cls"), 'rb'))
      labels = np.append(labels, new_labels)
      pickle.dump(labels, open(hs_file.replace(".hs", ".cls"), 'wb'))
      urls = pickle.load(open(hs_file.replace(".hs", ".url"), 'rb'))
      urls = np.append(urls, new_urls)
      pickle.dump(urls, open(hs_file.replace(".hs", ".url"), 'wb'))
      keyws = pickle.load(open(hs_file.replace(".hs", ".kwords"), 'rb'))
      keyws = np.append(keyws, new_keywords)
      pickle.dump(keyws, open(hs_file.replace(".hs", ".kwords"), 'wb'))
  else:
      pickle.dump(new_hs_mat, open(hs_file, 'wb'))
      pickle.dump(new_ids, open(hs_file.replace(".hs", ".ids"), 'wb'))
      pickle.dump(new_labels, open(hs_file.replace(".hs", ".cls"), 'wb'))
      pickle.dump(new_urls, open(hs_file.replace(".hs", ".url"), 'wb'))
      pickle.dump(new_keywords, open(hs_file.replace(".hs", ".kwords"), 'wb'))


if __name__ == '__main__':
    args = docopt(__doc__, version='Hashing a base pod, ver 0.1')

    f_dataset = args['--docfile']
    fly_model = args['--fly']

    pathlib.Path('./hashes').mkdir(parents=True, exist_ok=True)

    with open(fly_model, 'rb') as f: 
      fly_model = pickle.load(f)

    hash_documents(f_dataset, fly_model)
