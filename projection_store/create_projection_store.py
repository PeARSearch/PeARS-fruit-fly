"""Create projection store from preprocessed sentencepiece docs' cooccurrences

Usage:
  create_projection_store.py --size=<n> --dataset=[wos|wiki|20news]
  create_projection_store.py (-h | --help)
  create_projection_store.py --version
Options:
  -h --help                             Show this screen.
  --version                             Show version.
  --size=<n>                            Size of projections.
  --dataset=[wos|wiki|20news]           Dataset to extract projections from.

"""

from docopt import docopt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from utils import read_vocab, wta, return_keywords
from utils import hash_dataset_
from scipy import sparse
from scipy.sparse import csr_matrix, vstack


def write_to_store(proj_list,kwd_list):
    print("Writing",len(proj_list),"to store...")
    f_out = open("./projection_data/"+dataset+".projection.store."+str(n_words),'w')
    for i in range(len(proj_list)):
        f_out.write(' '.join([str(i) for i in proj_list[i]])+" :: "+' '.join(kwd_list[i])+'\n')
    f_out.close()


def process_file(f,top_words):
    print("Processing",f)
    doc=""
    proj_list = []
    kwd_list = []
    with open(f,'r') as f:
        for l in f:
          l = l.rstrip('\n')

          if l[:5] != "</doc" and l[:4] != "<doc":
              doc+=l.lower()+' ' 

          if l[:5] == "</doc" and doc != "":
              X = vectorizer.fit_transform([doc])
              X = csr_matrix(X)
              X = X.multiply(logprobs)
              vec = wta(X.toarray()[0], top_words, percent=False)
              kwd_ids = return_keywords(vec,top_words)
              kwds = [reverse_vocab[w] for w in kwd_ids]
              proj_list.append(kwd_ids)
              kwd_list.append(kwds)
              doc=""
    write_to_store(proj_list,kwd_list)


if __name__ == '__main__':
    args = docopt(__doc__, version='Create a projection store, ver 0.1')
    dts = {'wos':'wos','wiki':'wikipedia','20news':'20news-bydate'}
    dataset = args['--dataset']
    n_words = int(args['--size'])

    vocab, reverse_vocab, logprobs = read_vocab("../spm/spm."+dataset+".vocab")
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

    if dataset == 'wos':
        path = '../datasets/'+dts[dataset]+'/'+dts[dataset]+'11967-train.sp' 
    else:
        path = '../datasets/'+dts[dataset]+'/'+dts[dataset]+'-train.sp' 
    process_file(path, n_words)
