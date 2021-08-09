"""Common Crawl processor - retrieve the top k topics with a specific keyword associated with the documents

Usage:
  topk_lda.py --folder=<foldername> --pathdataset=<foldername> --pathmodel=<filename> --topk=<highestktopics> --word=<word>
  topk_lda.py (-h | --help)
  topk_lda.py --version


Options:
  -h --help     Show this screen.
  --version     Show version.
  --folder=<foldername>     Name of the folder where the .txt files are located
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
from collections import defaultdict
import pandas as pd
from docopt import docopt
csv.field_size_limit(sys.maxsize)

def load_model(filepath):
  temp_file = datapath(filepath)
  lda = LdaModel.load(temp_file, mmap='r')
  return lda

def load_original_documents(folder_txt):
  docs=[]
  txt_ori = open(folder_txt+"/corpus_lda.txt")
  for line in txt_ori.read().splitlines():
    docs.append(line)
  txt_ori.close()
  return docs

def load_corpus(pathdataset):
  corpus=pickle.load(open(pathdataset+'/corpus_train.p', 'rb'))
  print(f"Original documents and corpus loaded...")
  return corpus

def load_idx_topics(pathdataset):
  idx_topics = {}
  txt = open(f'./{pathdataset}/topics_lda.txt', 'r')
  for top in txt.read().splitlines():
    top=top.split("\t")
    idx_topics[int(top[1])]=top[0].split(", ")
  txt.close()
  return idx_topics

def getktopics_prob(lda, word, k, pathdataset, folder_txt):
  dic={}
  corpus=load_corpus(pathdataset)
  idx_topics=load_idx_topics(pathdataset)
  docs = load_original_documents(folder_txt)
  for i, doc in enumerate(docs):
    tops_text=lda.get_document_topics(corpus[i], minimum_probability=0)
    tops_text = sorted(tops_text, key=lambda x:x[1], reverse=True)
    for tup in tops_text[:k]:
      if word in idx_topics[tup[0]]:
        dic[i]=defaultdict(list)
        dic[i]['doc']=doc
        dic[i]['probability'].append(tup[1])
        dic[i]["index_topic"].append(tup[0])
        dic[i]['topics'].append(idx_topics[tup[0]])

  df=pd.DataFrame.from_dict(dic, orient='index')
  df.to_csv(f"topkprob_{word}.csv")
  print(df)
  return df
  

if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Processor')

    folder_txt = "./"+args['--folder']
    pathdataset = "./"+args['--pathdataset']
    pathmodel=args['--pathmodel']
    k = int(args['--topk'])
    word = args['--word']

    lda = load_model(pathmodel)
    getktopics_prob(lda, word, k, pathdataset, folder_txt)
