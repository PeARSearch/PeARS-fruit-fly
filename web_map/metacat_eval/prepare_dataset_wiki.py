"""Dataset preparation for wikipedia meta-categories and their respective webpage documents.
Usage:
  prepare_dataset_wiki.py --num_docs=<integer> --num_metacats=<integer>
  prepare_dataset_wiki.py (-h | --help)
  prepare_dataset_wiki.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --num_docs=<integer>            Number of documents to keep per meta-category
  --num_metacats=<integer>        Number of metacatories to keep                        
"""

import random
import os
import sentencepiece as spm
import glob
import gzip
import re
from collections import defaultdict
import pickle
from wiki_cats import dic_metacategories, distribution_metacategories
from docopt import docopt


def save_dataset(meta_text, f_dataset):
  f_out=open(f_dataset, 'w')
  for e, tup in enumerate(meta_text):
    f_out.write("<doc id="+str(e)+" url="+tup[1]+" class="+tup[2]+">\n")
    f_out.write(tup[0]+'\n')
    f_out.write("</doc>\n")
  f_out.close()


def prepare_texts_labels(f_dataset, docs_dic, num_docs, num_metacats):
  """
  Returns a list of tuples with the doc in the first position and the label in the second. 
  This process takes at most 2000 documents per meta-cat and randomizes the order of labels. 
  """
  print("\nPreparing labels and texts...")
  f_disc=open("metacats_to_discard.txt", 'r')
  discard=set()
  for line in f_disc.read().splitlines():
    discard.add(line)
  f_disc.close()
  
  f_meta=open('./wiki_cats/distrib_metacategories.txt', 'r')
  metacats=set()
  for line in f_meta:
    line=line.rstrip('\n').split('\t')
    if line[0] not in discard:
      metacats.add(line[0])
      if len(metacats)>=num_metacats: # 160
        break
  f_meta.close()
  print("LENGTH metacategories:", len(metacats))

  meta_text=[]
  for meta in metacats:
    if meta in docs_dic.keys():
      if len(docs_dic[meta])>num_docs:  #1000
        docs = random.sample(docs_dic[meta], num_docs)
      else:
        docs = random.sample(docs_dic[meta], len(docs_dic[meta]))
      for doc in docs:
        meta_text.append((doc[0], doc[1], meta))  #doc[1] is the url of the document

  meta_text=random.sample(meta_text, len(meta_text)) 
  save_dataset(meta_text, f_dataset)

  return meta_text

def read_dataset(f_dataset):
  meta_text=[]
  with open(f_dataset,'r') as f:
    doc=""
    for l in f:
      l = l.rstrip('\n')
      if l[:4] == "<doc":
        m = re.search(".*id=([^ ]*) ",l)
        ID=m.group(1)
        m = re.search(".*class=([^ ]*)>",l)
        lab=m.group(1)
        continue
      if l[:5] != "</doc" and l[:4] != "<doc":
        doc += l + " "
        continue
      if l[:5] == "</doc":
        meta_text.append((doc, ID, lab))
        doc=""
        ID=""
        lab=""
        continue
    f.close()
  return meta_text


def output_wordpieces(f_dataset, train_p, val_p):
  """
  train_p, val_p: the proportion of traning part and validation part
  They should be > 0 and < 1, e.g. 0.6 and 0.2 (the testing part remains 0.2)
  """

  outfile_train = open("./wiki_cats/wiki_cats_train.sp", 'w')
  outfile_val = open("./wiki_cats/wiki_cats_val.sp", 'w')
  outfile_test = open("./wiki_cats/wiki_cats_test.sp", 'w')
  n_train, n_val, n_test = 0, 0, 0 # count the number of docs for each part

  meta_text=read_dataset(f_dataset)
  label2idx={k:v for v, k in enumerate(set([i[2] for i in meta_text]))}
  pickle.dump(label2idx, open('./wiki_cats/label2idx.p', 'wb'))

  for tup in meta_text:
    doc = tup[0]
    ID = tup[1]
    label = str(label2idx[tup[2]])

    ll = sp.encode_as_pieces(doc)

    # use random to decide the current doc belongs to train, val, or test
    is_train = random.choices([0, 1], weights=[1-train_p, train_p])[0]
    if is_train:
      outfile_train.write("<doc id="+str(ID)+" class="+label+">\n")  # url="+url+"
      outfile_train.write(' '.join([wp for wp in ll])+'\n')
      outfile_train.write("</doc>\n")
      n_train += 1
    else:
      is_val = random.choices([0, 1], weights=[1-val_p/(1-train_p), val_p/(1-train_p)])[0]
      if is_val:
        outfile_val.write("<doc id="+str(ID)+" class="+label+">\n") #url="+url+"
        outfile_val.write(' '.join([wp for wp in ll])+'\n')
        outfile_val.write("</doc>\n")
        n_val += 1
      else:
        outfile_test.write("<doc id="+str(ID)+" class="+label+">\n")  # url="+url+"
        outfile_test.write(' '.join([wp for wp in ll])+'\n')
        outfile_test.write("</doc>\n")
        n_test += 1

  outfile_train.close()
  outfile_val.close()
  outfile_test.close()

  # write the number of docs in each part
  with open('./wiki_cats/wiki_cats_stat.txt', 'w') as f:
    f.write(str(n_train) + ' ' + str(n_val) + ' ' + str(n_test)+'\n')
  f.close()


if __name__ == '__main__':
    args = docopt(__doc__, version='Prepare dataset of wikipedia s externial links, ver 0.1')
    
    random.seed(99)
    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')

    num_docs=int(args['--num_docs'])
    num_metacats=int(args['--num_metacats'])

    f_dataset = f"./wiki_cats/{num_metacats}_wikimetacats.txt"
    f_urls_docs="./wiki_cats/connect_cats_text.txt"

    docs_cat = distribution_metacategories(f_urls_docs, "False")
    meta_text = prepare_texts_labels(f_dataset, docs_cat, num_docs, num_metacats)
    output_wordpieces(f_dataset, train_p=0.6, val_p=0.2)

    print('Datasets ready to be used for the classification...')

