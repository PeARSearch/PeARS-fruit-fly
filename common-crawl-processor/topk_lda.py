"""Common Crawl processor - retrieve the top k topics with a specific keyword associated with the documents

Usage:
  topk_lda.py --foldertxt=<foldername> --pathdataset=<foldername> --pathmodel=<filename> --topk=<highestktopics> --word=<word>
  topk_lda.py (-h | --help)
  topk_lda.py --version


Options:
  -h --help     Show this screen.
  --version     Show version.
  --foldertxt=<foldername>     Name of the folder where the .txt files are located
  --pathdataset=<foldername>    Name of the folder where the preprocessed data is located  
  --pathmodel=<filename>        Name of the file that where the LDA model has been saved
  --topk=<highestktopics>       Number of topics with the highest topics associated with documents
  --word=<word>                 Word to be focused in your search of topics

"""

import pickle
import numpy as np
import csv
import sys
from gensim.test.utils import datapath
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from docopt import docopt
csv.field_size_limit(sys.maxsize)

def load_model(filepath):
  temp_file = datapath(filepath)
  lda = LdaModel.load(temp_file, mmap='r')
  return lda

def getktopics_prob(lda, word, k, corpus, idx_topics, docs):

  for i, doc in enumerate(docs):
      tops_text=lda.get_document_topics(corpus[i], minimum_probability=0)
      tops_text = sorted(tops_text, key=lambda x:x[1], reverse=True)
      for tup in tops_text[:k]:
        if word in idx_topics[tup[0]]:
            print(doc)
            print("Index topic:", tup[0])
            print('Probability of correlation between doc and topic', tup[1])

if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Processor')

    folder_txt = args['--foldertxt']
    pathdataset = args['--pathdataset']
    pathmodel=args['--pathmodel']
    k = int(args['--topk'])
    word = args['--word']

    docs=[]
    txt_ori = open(folder_txt+"docs_octis.txt")
    for line in txt_ori.read().splitlines():
        docs.append(line)

    corpus=pickle.load(open(pathdataset+'/corpus_train.p', 'rb'))
    print(f"Original documents and corpus loaded...")

    txt_topics = open(f'./{pathdataset}/100topics_lda.txt', 'r')
    idx_topics={}
    for line in txt_topics.read().splitlines():
        idx = line.split("\t")[-1]
        idx_topics[idx]=line.split("\t")[0].split(" ")

    lda = load_model(pathmodel)
    getktopics_prob(lda, word, k, corpus, idx_topics, docs)