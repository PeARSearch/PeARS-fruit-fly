"""Common Crawl processor - pre-processing of a number of texts to be fed into the topic model (LDA) of the OCTIS library. 

Usage:
  preprocess_gensim.py --folder=<foldername> --ndocs=<n> --lda_path=<foldername> 
  preprocess_gensim.py (-h | --help)
  preprocess_gensim.py --version


Options:
  -h --help                        Show this screen.
  --version                        Show version.
  --folder=<foldername>            Name of the folder where the zipped .xml files are located
  --ndocs=<n>                      Number of documents to be preprocessed for training the topic model
  --lda_path=<foldername>          Folder where the pre-processed dataset will be placed

"""

from nltk.tokenize import RegexpTokenizer
from gensim.corpora import Dictionary
import glob
import csv
import sys
import os
import pickle
from docopt import docopt
csv.field_size_limit(sys.maxsize)


def get_n_docs(folder, n_docs):
    globs = glob.glob("./"+folder+"/docs*.txt")
    f_in = open("./"+folder+"corpus_lda.txt", 'w', encoding='utf8')
    n=0
    for file in globs:
      f_read = open(file, 'r', encoding="utf-8")
      for line in f_read.read().splitlines():
        if n >= n_docs:
          break
        else:
          f_in.write(line+'\n')
          n+=1
        if n%200==0:
          print(f"{n} processed")
      f_read.close()

    f_in.close()

def list_docs(folder):
    docs = []  #complete web docs
    f_doc = open(folder+'corpus_lda.txt', 'r', encoding='utf-8')
    for line in f_doc.read().splitlines():
      docs.append(line)
    return docs

def preprocess(docs, lda_path):
    if os.path.isdir(lda_path):
        pass
    else:
        os.makedirs(lda_path)
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    txt_dict=open("vocabulary.txt", 'r')
    tokens=[[line] for line in txt_dict.read().splitlines()]

    dictionary = Dictionary(tokens)
    pickle.dump(dictionary, open(lda_path+'/dict_gensim.p', 'wb'))

    corpus = [dictionary.doc2bow(doc) for doc in docs]
    pickle.dump(corpus, open(lda_path+'/corpus_train.p', 'wb'))
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Processor')

    folder = args['--folder']+"/"
    n_docs = args['--ndocs']
    lda_path=args['--lda_path']

    get_n_docs(folder, int(n_docs))
    docs=list_docs(folder)
    preprocess(docs, lda_path)

