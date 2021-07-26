"""Common Crawl processor - pre-processing of a number of texts to be fed into the topic model (LDA) of the OCTIS library. 

Usage:
  preprocess_octis.py --foldertxt=<foldername> --ndocs=<numberofwebdocs> --pathdataset=<foldername> 
  preprocess_octis.py (-h | --help)
  preprocess_octis.py --version


Options:
  -h --help     Show this screen.
  --version     Show version.
  --foldertxt=<foldername>     Name of the folder where the zipped .xml files are located
  --ndocs=<numberofdocs>         Number of documents to be preprocessed for training the topic model
  --pathdataset=<foldername>       Folder where the pre-processed dataset will be placed

"""

import os
import string
import csv
import sys
import glob
from octis.preprocessing.preprocessing import Preprocessing
import time
from docopt import docopt
csv.field_size_limit(sys.maxsize)

def get_n_docs(folder, n_docs):
    globs = glob.glob("./"+folder+"/*.txt")
    f_in = open("./"+folder+"/docs_octis.txt", 'a', encoding='utf8')
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


def preprocess(folder_txt, folder_dataset):
    tic = time.time()

    print("Initialize pre-processing")
    preprocessor = Preprocessing(vocabulary=None, max_features=None, min_df=0.002, max_df=0.5,
                                 remove_punctuation=True, punctuation=string.punctuation,
                                 lemmatize=False, stopword_list='english',
                                 min_chars=1, min_words_docs=0)

    dataset = preprocessor.preprocess_dataset(documents_path="./"+folder_txt+"/docs_octis.txt")
    dataset.save(folder_dataset)

    toc = time.time()
    print(f"{round((toc-tic)/60, 3)} minutes to pre-process documents. Output can be found in '{folder_dataset}'.")

if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Processor')

    folder_txt = args['--foldertxt']
    n_docs = args['--ndocs']
    dataset = args['--pathdataset']

    get_n_docs(folder_txt, int(n_docs))
    preprocess(folder_txt, dataset)


