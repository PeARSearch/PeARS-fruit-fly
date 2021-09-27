"""Hash documents with selected fly

Usage:
  hash_with_best_proj.py --fly=<selected fly> --docfile=<filename>
  hash_with_best_proj.py (-h | --help)
  hash_with_best_proj.py --version
Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --docfile=<path>          Path of file containing documents and information about each doc such as URL and label.
  --fly=<path>              Path to selected fly model.

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
from collections import defaultdict
from hash import wta, return_keywords

def hash_documents(f_dataset, best_fly):
  top_words = 250
  C = 100
  num_iter = 2000  # wikipedia and wos only need 50 steps
  sp = spm.SentencePieceProcessor()
  sp.load('../spmcc.model')
  vocab, reverse_vocab, logprobs = read_vocab()
  vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
  doc=""
  c=0
  dic_labs={}
  with open(f_dataset,'r') as f:
    for l in f:
      l = l.rstrip('\n')
      if l[:4] == "<doc":
        m = re.search(".*class=([^ ]*)>",l)
        lab=m.group(1)
        if lab not in dic_labs.keys():
          dic_labs[lab]=defaultdict(list)
        m = re.search(".*id=([^ ]*) ",l)
        ID=m.group(1)
        dic_labs[lab]['ids'].append(ID)
        m = re.search(".*url=([^ ]*) ",l)
        url=m.group(1)
        dic_labs[lab]['urls'].append(url)
        continue

      if l[:5] != "</doc" and l[:4] != "<doc":
        doc = l 
        continue

      if l[:5] == "</doc" and doc != "":
        ll = sp.encode_as_pieces(doc)
        X = vectorizer.fit_transform([" ".join(ll)])
        X = csr_matrix(X)
        X = X.multiply(logprobs)
        dic_labs[lab]['X'].append(X)
        vec = wta(X.toarray()[0], top_words, percent=False)
        kwds = [reverse_vocab[w] for w in return_keywords(vec)]
        dic_labs[lab]['keywords'].append(kwds)
        doc=""
        c+=1
        if c % 200 == 0:
          print(f"{c} documents processed so far...")
        continue

  print("Start hashing...")
  for e, lab in enumerate(dic_labs.keys()):
    Xs = vstack(np.array(dic_labs[lab]['X']))
    # print("Xs", Xs.shape, len(dic_labs[lab]['X']))
    hashes = hash_dataset_(dataset_mat=Xs, weight_mat=best_fly.projection,
                     percent_hash=best_fly.wta, top_words=top_words)

    hs_file='./hashes/'+lab+".hs"
    if hs_file in glob.glob('./hashes/*.hs'):
      hs_mat = pickle.load(open(hs_file, 'rb'))
      hs_matrix = vstack([hs_mat, hashes])
      pickle.dump(hs_matrix, open(hs_file, 'wb'))
      ids = pickle.load(open(hs_file.replace(".hs", ".ids"), 'rb'))
      ids = np.append(ids, np.array(dic_labs[lab]['ids']))
      pickle.dump(ids, open(hs_file.replace(".hs", ".ids"), 'wb'))
      urls = pickle.load(open(hs_file.replace(".hs", ".url"), 'rb'))
      urls = np.append(urls, np.array(dic_labs[lab]['urls']))
      pickle.dump(urls, open(hs_file.replace(".hs", ".url"), 'wb'))
      keyws = pickle.load(open(hs_file.replace(".hs", ".kwords"), 'rb'))
      keyws = np.append(keyws, np.array(dic_labs[lab]['keywords']))
      pickle.dump(keyws, open(hs_file.replace(".hs", ".kwords"), 'wb'))
    else:
      pickle.dump(hashes, open(hs_file, 'wb'))
      pickle.dump(np.array(dic_labs[lab]['ids']), open(hs_file.replace(".hs", ".ids"), 'wb'))
      pickle.dump(np.array(dic_labs[lab]['urls']), open(hs_file.replace(".hs", ".url"), 'wb'))
      pickle.dump(np.array(dic_labs[lab]['keywords']), open(hs_file.replace(".hs", ".kwords"), 'wb'))
    if e % 20 == 0:
      print(f'{e} categories saved with hashes...')


if __name__ == '__main__':
    args = docopt(__doc__, version='Hashing a document, ver 0.1')

    f_dataset = args['--docfile']
    fly_model = args['--fly']

    pathlib.Path('./hashes').mkdir(parents=True, exist_ok=True)

    with open(fly_model, 'rb') as f:  # modified the name of the fruit-fly here
      fly_model = pickle.load(f)

    hash_documents(f_dataset, fly_model)
