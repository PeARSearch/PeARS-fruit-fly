"""Common Crawl processor - filter documents from innapropriate content and return .json files with clean documents and their respective metadata at each line

Usage:
  filter_documents.py --folder=<foldername> --pathmodel=<pathname> --pathdataset=<foldername> --keep_discarded=<boolean>
  filter_documents.py (-h | --help)
  filter_documents.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --folder=<foldername>     Only the name of the folder where the zipped .xml files are located
  --pathmodel =<pathname>		Where the LDA model has been saved
  --pathdataset=<foldername>	Where the information from LDA has been saved
  --keep_discarded=<boolean>	True if you want to keep the documents discarded, otherwise False.

"""

import glob
import csv
import gzip
import shutil
import os
from docopt import docopt
import utils
import LDAmodel
import pickle
from nltk.tokenize import RegexpTokenizer
from gensim.test.utils import datapath
from gensim.models import LdaModel


def load_model(pathmodel):
  temp_file = datapath(pathmodel)
  lda = LdaModel.load(temp_file, mmap='r')
  return lda

def load_everything(pathdataset, pathmodel):
	dictionary=pickle.load(open(pathdataset+'dict_gensim.p', 'rb'))
	lda = load_model(pathmodel)
	tokenizer = RegexpTokenizer(r'\w+')
	topics={}
	txt=open('topics_threshold.txt', 'r')
	for t in txt.read().splitlines():
		t=t.split(" ")
		topics[int(t[0])]=float(t[-1])
	return dictionary, lda, tokenizer, topics

def filtering(folder, pathmodel, pathdataset, keep_discarded):
	dictionary, lda, tokenizer, topics=load_everything(pathdataset, pathmodel)
	if os.path.isdir("corpus"):
	    pass
	else:
	    os.makedirs("corpus")
	j_keep = './corpus/kept_0.json'
	if keep_discarded == 'True':
		j_disc= './corpus/discarded_0.json'
	n_doc=0
	n_kept=0
	f_globs= glob.glob(folder+"*.gz")
	for f in f_globs:
		print(f)
		with gzip.open(f, 'rb') as f_in:
			unzipped_f = folder+f.replace(".gz", "").split("/")[-1] 
			with open(unzipped_f, 'wb') as f_out:
				shutil.copyfileobj(f_in, f_out)

		with open(unzipped_f, 'r') as filename:
			for line in filename.read().splitlines():
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
						if label != 1:
							dic={}
							dic['doc']=doc
							dic['lang']='en'
							dic['title']=title
							dic['url']=url
							j_keep=utils.append_json_check_len(dic, j_keep)
							n_kept+=1
						else:
							if keep_discarded=='True':
								dic={}
								dic['doc']=doc
								dic['lang']='en'
								dic['title']=title
								dic['url']=url
								j_disc=utils.append_json_check_len(dic, j_disc)
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
		os.remove(unzipped_f)

if __name__ == '__main__':
  args = docopt(__doc__, version='Common Crawl Processor')

  folder = "./"+args['--folder']+"/"
  print(folder)
  pathmodel=args['--pathmodel']
  pathdataset="./"+args['--pathdataset']+"/"
  keep_discarded=args['--keep_discarded']

  filtering(folder, pathmodel, pathdataset, keep_discarded)
