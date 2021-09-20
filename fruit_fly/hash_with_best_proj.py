"""Hash documents with selected fly

Usage:
  hash_with_best_proj.py --docfile=<filename> --flyfolder=<foldername>
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
from hash import wta, return_keywords

def hash_documents(f_dataset, best_fly):

  top_words = 250
  C = 100
  num_iter = 2000  # wikipedia and wos only need 50 steps
  sp = spm.SentencePieceProcessor()
  sp.load('../spmcc.model')
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
  hashes = hash_dataset_(dataset_mat=vstack(hashes), weight_mat=best_fly.projection,
                     percent_hash=best_fly.wta, top_words=top_words)
  print("Saving information...")
  for e, lab in enumerate(set(new_labels)):
    indices=[idx for idx, element in enumerate(new_labels) if element == lab]
    new_hs_mat = vstack(hashes[indices])

    hs_file='./hashes/'+lab+".hs"
    if hs_file in glob.glob('./hashes/*.hs'):
      hs_mat = pickle.load(open(hs_file, 'rb'))
      # print(hs_file)
      hs_matrix = vstack([hs_mat, new_hs_mat])
      pickle.dump(hs_mat, open(hs_file, 'wb'))
      ids = pickle.load(open(hs_file.replace(".hs", ".ids"), 'rb'))
      ids = np.append(ids, new_ids[indices])
      pickle.dump(ids, open(hs_file.replace(".hs", ".ids"), 'wb'))
      labels = pickle.load(open(hs_file.replace(".hs", ".cls"), 'rb'))
      labels = np.append(labels, new_labels[indices])
      pickle.dump(labels, open(hs_file.replace(".hs", ".cls"), 'wb'))
      urls = pickle.load(open(hs_file.replace(".hs", ".url"), 'rb'))
      urls = np.append(urls, new_urls[indices])
      pickle.dump(urls, open(hs_file.replace(".hs", ".url"), 'wb'))
      keyws = pickle.load(open(hs_file.replace(".hs", ".kwords"), 'rb'))
      keyws = np.append(keyws, new_keywords[indices])
      pickle.dump(keyws, open(hs_file.replace(".hs", ".kwords"), 'wb'))
    else:
      pickle.dump(new_hs_mat, open(hs_file, 'wb'))
      pickle.dump(new_ids[indices], open(hs_file.replace(".hs", ".ids"), 'wb'))
      pickle.dump(new_labels[indices], open(hs_file.replace(".hs", ".cls"), 'wb'))
      pickle.dump(new_urls[indices], open(hs_file.replace(".hs", ".url"), 'wb'))
      pickle.dump(new_keywords[indices], open(hs_file.replace(".hs", ".kwords"), 'wb'))
    if e % 10 == 0:
      print(f'{e} new categories saved out of {len(set(new_labels))}...')


if __name__ == '__main__':
    args = docopt(__doc__, version='Hashing a document, ver 0.1')

    f_dataset = args['--docfile']
    fly_model = args['--fly']

    pathlib.Path('./hashes').mkdir(parents=True, exist_ok=True)

    with open(fly_model, 'rb') as f:  # modified the name of the fruit-fly here
      fly_model = pickle.load(f)

    hash_documents(f_dataset, fly_model)
