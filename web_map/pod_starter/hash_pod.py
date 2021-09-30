"""Hash base pod with selected fly

Usage:
  hash_pod.py --fly=<path>
  hash_pod.py (-h | --help)
  hash_pod.py --version
Options:
  -h --help                 Show this screen.
  --version                 Show version.
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
from os.path import join
import glob
import re

class DeployedFly:
    def __init__(self):
        self.kc_size = None
        self.wta = None
        self.projection = None
        self.val_scores = []

def read_categories(metacat_dir):
    categories=glob.glob(metacat_dir+"/*")
    return categories


def hash_documents(f_dataset, best_fly):
  print("Processing",f_dataset)
  top_words = 250
  sp = spm.SentencePieceProcessor()
  sp.load('../../spmcc.model')
  vocab, reverse_vocab, logprobs = read_vocab()
  vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
  new_ids = [] 
  new_labels = [] 
  new_urls = [] 
  new_keywords = [] 
  hashes= [np.array([])]
  doc=""
  with open(f_dataset,'r') as f:
    for l in f:
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
        hashes = np.append(hashes, X)
        vec = wta(X.toarray()[0], top_words, percent=False)
        kwds = [reverse_vocab[w] for w in return_keywords(vec)]
        new_ids.append(ID)
        new_labels.append(lab)
        new_urls.append(url)
        new_keywords.append(kwds)
        doc=""
        continue

  new_hs_mat = hash_dataset_(dataset_mat=vstack(hashes), weight_mat=best_fly.projection,
                     percent_hash=best_fly.wta, top_words=top_words)
  lab = new_labels[0] #all labels should be the same

  hs_file='./hashes/'+lab+".hs"
  if hs_file in glob.glob('./hashes/*.hs'):
      hs_mat = pickle.load(open(hs_file, 'rb'))
      hs_matrix = vstack([hs_mat, new_hs_mat])
      pickle.dump(hs_matrix, open(hs_file, 'wb'))
      print(hs_mat.shape,new_hs_mat.shape,hs_matrix.shape,len(new_ids))

      ids = pickle.load(open(hs_file.replace(".hs", ".ids"), 'rb'))
      ids = ids + new_ids
      pickle.dump(ids, open(hs_file.replace(".hs", ".ids"), 'wb'))
      print(len(ids),len(new_ids))

      labels = pickle.load(open(hs_file.replace(".hs", ".cls"), 'rb'))
      labels = labels + new_labels
      pickle.dump(labels, open(hs_file.replace(".hs", ".cls"), 'wb'))
      print(len(labels),len(new_labels))

      urls = pickle.load(open(hs_file.replace(".hs", ".url"), 'rb'))
      urls = urls + new_urls
      pickle.dump(urls, open(hs_file.replace(".hs", ".url"), 'wb'))
      print(len(urls),len(new_urls))

      keyws = pickle.load(open(hs_file.replace(".hs", ".kwords"), 'rb'))
      keyws = keyws + new_keywords
      pickle.dump(keyws, open(hs_file.replace(".hs", ".kwords"), 'wb'))
      print(len(keyws),len(new_keywords))
  else:
      pickle.dump(new_hs_mat, open(hs_file, 'wb'))
      pickle.dump(new_ids, open(hs_file.replace(".hs", ".ids"), 'wb'))
      pickle.dump(new_labels, open(hs_file.replace(".hs", ".cls"), 'wb'))
      pickle.dump(new_urls, open(hs_file.replace(".hs", ".url"), 'wb'))
      pickle.dump(new_keywords, open(hs_file.replace(".hs", ".kwords"), 'wb'))


if __name__ == '__main__':
    args = docopt(__doc__, version='Hashing a base pod, ver 0.1')

    fly_model = args['--fly']
    pathlib.Path('./hashes').mkdir(parents=True, exist_ok=True)

    metacat = input("Please enter a category name: ").replace(' ','_')
    metacat_dir = "./data/categories/"+metacat
    cats = read_categories(metacat_dir)

    with open(fly_model, 'rb') as f: 
      fly_model = pickle.load(f)

    for cat in cats:
        hash_documents(join(cat,"linear.txt"), fly_model)

    print("Hashing complete!")
