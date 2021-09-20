"""Hash a document with selected fly

Usage:
  hash_with_best_proj.py --fly=<filename> --docfile=<filename> 
  hash_with_best_proj.py (-h | --help)
  hash_with_best_proj.py --version
Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --fly=<filename>             Path to selected fly for hashing.
  --docfile=<filename>         File containing text of Web documents. 

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
from scipy.sparse import csr_matrix
import pathlib
from docopt import docopt
import glob
import re

def hash_documents(docs, fly_model, outfile_txt):
  # load model
  with open(fly_model, 'rb') as f:  # modified the name of the fruit-fly here
      best_fly = pickle.load(f)

  top_word = 250
  sp = spm.SentencePieceProcessor()
  sp.load('../spmcc.model')
  vocab, reverse_vocab, logprobs = read_vocab()
  vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

  matrices_id = glob.glob('./hashes/*.npz')
  if len(matrices_id)==0:
    last_id=-1
    print(last_id)
  else:
    last_id = max([int(re.findall(r'\d+', i)[0]) for i in matrices_id])
    print(last_id)

  for e, doc in enumerate(docs):
    last_id+=1
    ll = sp.encode_as_pieces(doc)
    X = vectorizer.fit_transform([" ".join(ll)])
    X = csr_matrix(X)
    print(X.shape)
    X = X.multiply(logprobs)

    hs = hash_dataset_(dataset_mat=X, weight_mat=best_fly.projection,
                               percent_hash=best_fly.wta, top_words=top_word)

    outfile_txt.write("<doc id="+str(last_id)+" class="+"UNK"+">\n")  # url="+url+"
    outfile_txt.write("</doc>\n")

    sparse.save_npz(folder_hs+str((last_id))+".npz", hs)
    print("Output can be found in hashes")
    if e % 2 == 0:
      print(f"{e} documents hashed so far...")

if __name__ == '__main__':
    args = docopt(__doc__, version='Hashing a document, ver 0.1')

    f_in = args['--docfile']
    fly_model = args['--fly']

    pathlib.Path('./hashes').mkdir(parents=True, exist_ok=True)

    folder_hs = "./hashes/"
    outfile_txt = open("./hashes/hs_dats.sp", 'a')

    docs=[]
    f_text=open(f_in, 'r')
    for line in f_text.read().splitlines():
      if line != "":
        docs.append(line)

    hash_documents(docs, fly_model, outfile_txt)
