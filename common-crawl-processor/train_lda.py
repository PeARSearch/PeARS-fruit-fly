"""Common Crawl processor - training of the LDA model with the Gensim library

Usage:
  train_lda.py --pathdataset=<foldername> --outputfile=<filename>
  train_lda.py (-h | --help)
  train_lda.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --pathdataset=<foldername>	Name of the folder where the preprocessed documents are.
  --outputfile=<filename>		Name of the file where the output of the lda model will be saved.

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

def train_lda(outputfile, pathdataset):
	tic = time.time()
	chunksize = 2000
	passes = 20
	iterations = 400
	num_topics=100
	eval_every = None  
	dictionary= pickle.load(open(pathdataset+'/dict_gensim.p', 'rb'))
	corpus=pickle.load(open(pathdataset+'/corpus_train.p', 'rb'))
	print(f"Dictionary and corpus loaded from {pathdataset}...")

	# Make a index to word dictionary.
	temp = dictionary[0]  # This is only to "load" the dictionary.
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
	temp_file = datapath(outputfile)
	lda.save(temp_file)
		
	txt = open(f'./{pathdataset}/topics_lda.txt', 'w')
	topics = lda.show_topics(num_topics=num_topics, num_words=10, log=False, formatted=False)
	for top in topics:
		txt.write(", ".join([t[0] for t in top[1]])+"\t"+str(top[0])+'\n')
	txt.close()
	toc = time.time()
	print((toc-tic)/60, 'minutes to train the model')
	return lda

if __name__ == '__main__':
	args = docopt(__doc__, version='Common Crawl Processor')
	
	pathdataset=args['--pathdataset']
	outputfile=args['--outputfile']
	train_lda(outputfile, pathdataset)
