"""Common Crawl processor - filter documents from innapropriate content and return .json files with clean documents and their respective metadata at each line

Usage:
  filter_documents.py --folder=<foldername> --model=<pathname> --lda_path=<foldername> --keep_discarded=<boolean>
  filter_documents.py (-h | --help)
  filter_documents.py --version

Options:
  -h --help                     Show this screen.
  --version                     Show version.
  --folder=<foldername>         Only the name of the folder where the zipped .xml files are located
  --model =<pathname>		Where the LDA model has been saved
  --lda_path=<foldername>	Where the information from LDA has been saved
  --keep_discarded=<boolean>	True if you want to keep the documents discarded, otherwise False.
"""

import glob
import csv
import gzip
import os
from docopt import docopt
import utils
import LDAmodel
import pickle
from nltk.tokenize import RegexpTokenizer
from gensim.test.utils import datapath
from gensim.models import LdaModel

def load_model(model):
  temp_file = datapath(model)
  lda = LdaModel.load(temp_file, mmap='r')
  return lda

def load_everything(lda_path, model):
	dictionary=pickle.load(open(lda_path+'dict_gensim.p', 'rb'))
	lda = load_model(model)
	tokenizer = RegexpTokenizer(r'\w+')
	topics={}
	txt=open('topics_threshold.txt', 'r')
	for t in txt.read().splitlines():
		t=t.split(" ")
		topics[int(t[0])]=float(t[-1])
	return dictionary, lda, tokenizer, topics


def filtering(folder, model, lda_path, keep_discarded):
	dictionary, lda, tokenizer, topics=load_everything(lda_path, model)
	if os.path.isdir("corpus"):
	    pass
	else:
	    os.makedirs("corpus")
	f_keep = './corpus/kept_0.txt'
	if keep_discarded == 'True':
		f_discard= './corpus/discarded_0.txt'
	n_doc=0
	n_kept=0
	f_globs= glob.glob(folder+"*.gz")
	for f_gz in f_globs:
		with gzip.open(f_gz,'rt') as f:
			for line in f.read().splitlines():
				if line.startswith("<doc"):
					doc=""
					try:
						title=line.split(" ")[2].replace("title=", "")
						url=line.split(" ")[1].replace("url=", "")
					except IndexError:
						print(line)
						title=line
						url=line
					continue
				if line.startswith("</doc>"):
					if doc != "" or doc != " ":
						label = LDAmodel.classify_removal(doc, dictionary, lda, tokenizer, topics)
						dic={}
						dic['doc']=doc
						dic['lang']='en'
						dic['title']=title
						dic['url']=url
						if label != 1:
							utils.append_txt_check_len(dic, f_keep)
							n_kept+=1
						else:
							if keep_discarded=='True':
								f_discard=utils.append_txt_check_len(dic, f_discard)
						n_doc+=1
						if n_doc%100==0:
							print(f"{n_doc} documents checked and {n_kept} documents kept so far...")
					continue
				if line == "" or line == " ":
					continue
				else:
					if doc == "":
						doc=line
					else:
						doc=doc+" "+line
					continue
					
	return f_keep

if __name__ == '__main__':
  args = docopt(__doc__, version='Common Crawl Processor')

  folder = "./"+args['--folder']+"/"
  print(folder)
  model=args['--model']
  lda_path="./"+args['--lda_path']+"/"
  keep_discarded=args['--keep_discarded']

  f_keep=filtering(folder, model, lda_path, keep_discarded)
  utils.compress_file(f_keep)

