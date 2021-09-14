"""Dataset preparation for wikipedia meta-categories and their respective webpage documents.
Usage:
  prepare_dataset_wiki.py --linksfolder=<foldername> --num_docs=<integer> --num_metacats=<integer>
  prepare_dataset_wiki.py (-h | --help)
  prepare_dataset_wiki.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --linksfolder=<foldername>      Name of folder where the wikipedia external links and their 
                                  respective pages have been placed
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
import wiki_cats
from docopt import docopt

def metacats_with_texts(txt_files):
  '''
  Returns a dictionary with category as key and a list of texts belonging to the respective
  meta-category as value. 
  '''
  dic_metacats=wiki_cats.dic_metacategories('./wiki_cats/metacategories_topics.txt')
  dic_url={}
  docs_dic=defaultdict(list)
  pattern_url = "url='(.*?)'>"
  for txt_gz in txt_files:
    with gzip.open(txt_gz,'rt') as f:
      url=""
      doc=""
      for line in f.read().splitlines():
        if "url='" in line:
          try:
            url = re.search(pattern_url, line).group(1)
          except AttributeError:
            continue
        if line.startswith('<doc') == False and line.startswith('</doc>') == False:
          doc=line
          # print(line)
          if url != "" and doc != "":
            dic_url[url]=doc
            url=""
            doc=""
          else:
            continue

    cat_file = txt_gz.replace("links.txt.gz", "links.gz")
    with gzip.open(cat_file,'rt') as f:
      for line in f.read().splitlines():
        line=line.split("|")
        if line[0] in dic_url.keys():
          for cat in line[1:]:
            cat=cat.lower()
            if re.match(r"[A-Za-z0-9]+", cat) and cat in dic_metacats.keys():
              docs_dic[dic_metacats[cat]].append((dic_url[line[0]], line[0]))
  return docs_dic


def prepare_texts_labels(docs_dic, num_docs, num_metacats):
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
  metacats=[]
  for line in f_meta.read().splitlines():
    line=line.split('\t')
    if line[0] not in discard:
      metacats.append(line[0])
      if len(metacats)>=num_metacats:
        break
  f_meta.close()
  print("LENGTH metacategories:", len(metacats))

  label2idx={k:v for v, k in enumerate(metacats)}
  pickle.dump(label2idx, open('./wiki_cats/label2idx.p', 'wb'))

  meta_text=[]
  for meta in docs_dic.keys():
    if meta in metacats:
      if len(docs_dic[meta])>num_docs:
        docs = random.sample(docs_dic[meta], num_docs)
      else:
        docs = random.sample(docs_dic[meta], len(docs_dic[meta]))
      for doc in docs:
        meta_text.append((doc[0], doc[1], label2idx[meta]))  #doc[1] is the url of the document

  meta_text=random.sample(meta_text, len(meta_text)) 
  return meta_text

def output_wordpieces(meta_text, train_p, val_p):
  """
  train_p, val_p: the proportion of traning part and validation part
  They should be > 0 and < 1, e.g. 0.6 and 0.2 (the testing part remains 0.2)
  """

  outfile_train = open("./wiki_cats/wiki_cats_train.sp", 'w')
  outfile_val = open("./wiki_cats/wiki_cats_val.sp", 'w')
  outfile_test = open("./wiki_cats/wiki_cats_test.sp", 'w')
  n_train, n_val, n_test = 0, 0, 0 # count the number of docs for each part

  label_idx=0

  for tup in meta_text:
    doc = tup[0]
    # url=tup[1]
    label = str(tup[2])

    ll = sp.encode_as_pieces(doc)

    # use random to decide the current doc belongs to train, val, or test
    is_train = random.choices([0, 1], weights=[1-train_p, train_p])[0]
    if is_train:
      outfile_train.write("<doc id="+str(label_idx)+" class="+label+">\n")  # url="+url+"
      outfile_train.write(' '.join([wp for wp in ll])+'\n')
      outfile_train.write("</doc>\n")
      n_train += 1
    else:
      is_val = random.choices([0, 1], weights=[1-val_p/(1-train_p), val_p/(1-train_p)])[0]
      if is_val:
        outfile_val.write("<doc id="+str(label_idx)+" class="+label+">\n") #url="+url+"
        outfile_val.write(' '.join([wp for wp in ll])+'\n')
        outfile_val.write("</doc>\n")
        n_val += 1
      else:
        outfile_test.write("<doc id="+str(label_idx)+" class="+label+">\n")  # url="+url+"
        outfile_test.write(' '.join([wp for wp in ll])+'\n')
        outfile_test.write("</doc>\n")
        n_test += 1
    label_idx += 1

  outfile_train.close()
  outfile_val.close()
  outfile_test.close()

  # write the number of docs in each part
  with open('./wiki_cats/wiki_cats_stat.txt', 'w') as f:
    f.write(str(n_train) + ' ' + str(n_val) + ' ' + str(n_test)+'\n')
  f.close()

def wikipedia_cats(meta_text):

    output_wordpieces(meta_text, train_p=0.6, val_p=0.2)
    print('Datasets ready to be used for the classification...')


if __name__ == '__main__':
    args = docopt(__doc__, version='Prepare dataset of wikipedia s externial links, ver 0.1')
    
    random.seed(99)
    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')

    linksfolder=args['--linksfolder']
    num_docs=int(args['--num_docs'])
    num_metacats=int(args['--num_metacats'])

    txt_files = glob.glob(linksfolder+"*links.txt.gz")

    docs_cat = metacats_with_texts(txt_files)  #pickle.load(open('dic_cat.p', 'rb'))
    meta_text = prepare_texts_labels(docs_cat, num_docs, num_metacats)

    wikipedia_cats(meta_text)
