"""Common Crawl processor - pre-processing of a number of texts to be fed into the topic model (LDA) of the OCTIS library. 

Usage:
  preprocess_gensim.py --foldertxt=<foldername> --ndocs=<numberofwebdocs> --pathdataset=<foldername> 
  preprocess_gensim.py (-h | --help)
  preprocess_gensim.py --version


Options:
  -h --help     Show this screen.
  --version     Show version.
  --foldertxt=<foldername>     Name of the folder where the zipped .xml files are located
  --ndocs=<numberofdocs>         Number of documents to be preprocessed for training the topic model
  --pathdataset=<foldername>       Folder where the pre-processed dataset will be placed

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

def get_n_docs(folder_txt, n_docs):
    globs = glob.glob("./"+folder_txt+"/*.txt")
    f_in = open("./"+folder_txt+"/docs_octis.txt", 'a', encoding='utf8')
    n=0
    for file in globs:
      f_read = open(file, 'r', encoding="utf-8")
      for line in f_read.readlines():
        if n>n_docs:
          break
        else:
          f_in.write(line)
          n+=1
        if n%200==0:
          print(f"{n} processed")
      f_read.close()

    f_in.close()

def list_docs(folder_txt):
    docs = []  #complete web docs
    f_doc = open(folder_txt+'docs_octis.txt', 'r', encoding='utf-8')
    for line in f_doc.read().splitlines():
      docs.append(line)
    return docs

def preprocess(docs, pathdataset):
    if os.path.isdir(pathdataset):
        pass
    else:
        os.makedirs(pathdataset)
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    dictionary = Dictionary(docs)
    pickle.dump(dictionary, open(pathdataset+'/dict_gensim.p', 'wb'))

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=150, no_above=0.5)

    corpus = [dictionary.doc2bow(doc) for doc in docs]
    pickle.dump(corpus, open(pathdataset+'/corpus_train.p', 'wb'))
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Processor')

    folder_txt = args['--foldertxt']
    n_docs = args['--ndocs']
    pathdataset=args['--pathdataset']

    get_n_docs(folder_txt, int(n_docs))
    docs=list_docs(folder_txt)
    preprocess(docs, pathdataset)

