"""Common Crawl processor - training of the LDA model with the Gensim library

Usage:
  train_lda.py --lda_path=<foldername> --model_out=<filename>
  train_lda.py (-h | --help)
  train_lda.py --version

Options:
  -h --help                             Show this screen.
  --version                             Show version.
  --lda_path=<foldername>	        Name of the folder where the preprocessed documents are.
  --model_out=<filename>		Name of the file where the output of the lda model will be saved.

"""

import numpy as np
import csv
import sys
from docopt import docopt
import time
csv.field_size_limit(sys.maxsize)
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
from gensim.test.utils import datapath
import pickle
from itertools import chain
import pandas as pd

def train_lda(lda_path, model_out):
	tic = time.time()
	chunksize = 2000
	passes = 20
	iterations = 400
	num_topics=100
	eval_every = None  
	dictionary= pickle.load(open(lda_path+'/dict_gensim.p', 'rb'))
	corpus=pickle.load(open(lda_path+'/corpus_train.p', 'rb'))
	print(f"Dictionary and corpus loaded from {lda_path}...")

	temp = dictionary[0]  
	id2word = dictionary.id2token

	lda = LdaModel(
	    corpus=corpus,
	    id2word=id2word,
	    chunksize=chunksize,
	    alpha='auto',
	    eta='auto',
	    iterations=iterations,
	    num_topics=num_topics,
	    passes=passes,
	    eval_every=eval_every
	)

	# Save model to disk.
	temp_file = datapath(model_out)
	lda.save(temp_file)
		
	txt = open(f'./{lda_path}/topics_lda.txt', 'w')
	topics = lda.show_topics(num_topics=num_topics, num_words=10, log=False, formatted=False)
	for top in topics:
		txt.write(", ".join([t[0] for t in top[1]])+"\t"+str(top[0])+'\n')
	txt.close()
	toc = time.time()
	print((toc-tic)/60, 'minutes to train the model')
	return lda

if __name__ == '__main__':
	args = docopt(__doc__, version='Common Crawl Processor')
	
	lda_path=args['--lda_path']
	model_out=args['--model_out']
	train_lda(lda_path, model_out)
