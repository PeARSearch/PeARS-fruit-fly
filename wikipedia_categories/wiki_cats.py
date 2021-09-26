"""Understanding and filtering the wikipedia categories that have been extracted from 
Wikipedia's external links

Usage:
  wiki_cats.py --linksfolder=<foldername> --function=<functionoption>
  wiki_cats.py (-h | --help)
  wiki_cats.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --linksfolder=<foldername>      Name of folder where the wikipedia external links and their 
  								  							respective pages have been placed
  --function=<functionname>		  	Name of the function you want to execute

"""
from docopt import docopt
import gzip
import glob
from collections import Counter, defaultdict
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import re
import os
import pathlib
nltk.download('punkt')
nltk.download('stopwords')
 
stop_words = set(stopwords.words('english'))

def save_dataset(dic_cat, f_urls_docs):
  f_out=open(f_urls_docs, 'w')
  for k, v in dic_cat.items():
    f_out.write("<doc url="+k+" categories="+"|".join(list(v[1]))+">\n")
    f_out.write(v[0]+'\n')
    f_out.write("</doc>\n")
  f_out.close()

def connect_cats_text(txt_files, f_urls_docs):
	"""
	Reads the categories that have extracted with Wikipedia's external links.
	Returns a .txt file with the distribution of categories. 
	"""
	dic_url={}
	dic_cat={}
	for txt_gz in txt_files:
		doc=""
		with gzip.open(txt_gz,'rt') as f:
			for l in f:
				l=l.rstrip("\n")
				if l[:4] == "<doc":
					try:
						m=re.search(r"(?<=url=')(.*)(?=')", l)
						url=m.group(1)
						continue
					except AttributeError:
						url=""
						continue
				if l[:5] != "</doc" and l[:4] != "<doc":
					doc += l + " "
					continue
				if l[:5] == "</doc" and doc != "":
					if url == "":
						doc=""
						continue
					else:
						dic_url[url]=doc.rstrip(" ")
						doc=""
						url=""
						continue
	# print(dic_url.keys())
	urls=set()
	for txt_gz in txt_files:	
		cat_file = txt_gz.replace("links.txt.gz", "links.gz")
		with gzip.open(cat_file,'rt') as f:
			for line in f:
				line=line.rstrip("\n")
				line=line.split("|")
				url=line[0]
				if line[1:]==['']:
					continue
				cats=set()
				if url in dic_url.keys():  #line[0] is the url
					for cat in line[1:]:
						cat=cat.lower()
						cats.add(cat)
						
					if url in dic_cat.keys():
						old_cats=dic_cat[url][1]
						cats.update(old_cats)
						dic_cat[url]=(dic_url[url], cats)
					else:
							dic_cat[url]=(dic_url[url], cats)
			
	# os.unlink(f_urls_docs)
	if not os.path.isfile(f_urls_docs): 
		save_dataset(dic_cat, f_urls_docs)


def count_categories(f_urls_docs):
  cats_url=defaultdict(list)
  with open(f_urls_docs,'r') as f:
    doc=""
    for l in f:
      l = l.rstrip('\n')
      if l[:4] == "<doc":
        m = re.search(".*url=([^ ]*) ",l)
        url=m.group(1)
        m=re.search(r"(?<=categories=)(.*)(?=>)", l)
        cats=m.group(1)
        cats=cats.split("|")
        continue
      if l[:5] != "</doc" and l[:4] != "<doc":
        doc += l + " "
        continue
      if l[:5] == "</doc":
      	for cat in cats:
      		if cat not in cats_url:
      			cats_url[cat]=1
      		else:
      			cats_url[cat]+=1
      	doc=""
      	url=""
      	cats=""
      	continue
    f.close()

  cats=Counter(cats_url)
  most = cats.most_common(len(cats.keys()))
  with open('./wiki_cats/distrib_categories.txt', 'w') as f:
  	for t in most:
  		f.write(t[0]+"\t"+str(t[1])+'\n')
  f.close()
  print("Find categories reverse sorted in './wiki_cats/distrib_categories.txt'")

def read_categories():
	cats=[]
	print("Loading categories...")
	with open('./wiki_cats/distrib_categories.txt', 'r') as f:
		for line in f.read().splitlines():
			line=line.split("\t")
			cats.append((line[0], line[1]))
	f.close()
	return cats

def create_ngrams(ngram_n):
	"""
	Returns a .txt file with the ngrams and their frequency in the categories. 
	"""
	cats=read_categories()
	cats=[i[0] for i in cats if i[0]!=""]
	ngrams_l=[]
	dic=defaultdict(list)
	for cat in cats:
		token = word_tokenize(cat)
		token = [i for i in token if i not in stop_words]
		gram = list(ngrams(token, ngram_n)) 
		try:
			ngrams_l.append(gram[0])
			dic[gram[0]].append(cat)
		except IndexError:
			pass
	fdist = nltk.FreqDist(ngrams_l)
	sort_orders = sorted(fdist.items(), key=lambda x: x[1], reverse=True)
	pickle.dump(sort_orders, open("./wiki_cats/ngram"+str(ngram_n)+".p", 'wb'))
	pickle.dump(dic, open("./wiki_cats/dic"+str(ngram_n)+".p", 'wb'))

	with open("./wiki_cats/"+str(ngram_n)+"grams.txt", 'w') as f:
		for k in sort_orders:
			f.write(", ".join(k[0])+"\t"+str(k[1])+'\n')
	print("Find the ngrams in the file: './wiki_cats/"+str(ngram_n)+"grams.txt'.")
	print("If you want, you can run function 1 again with another ngram number.")
	f.close()


def create_metacategories(threshold):
	"""
	Takes the threshold of the frequency of the ngrams. 
	Returns a .txt file with categories that belong to the same meta-category in each line.
	"""
	if os.path.isfile("./wiki_cats/ignore_cats.p"):
		ignore_cats=pickle.load(open("./wiki_cats/ignore_cats.p", 'rb'))
	else:
		ignore_cats=set()
	ngrams_freq=glob.glob("./wiki_cats/ngram*.p")
	with open("./wiki_cats/metacategories.txt", 'a') as f:
		for ngram in ngrams_freq: 
			num=re.findall(r'\d+', ngram)[0]
			ngram_freq=pickle.load(open(ngram, 'rb'))
			dic=pickle.load(open("./wiki_cats/dic"+num+".p", 'rb'))
			for k in ngram_freq:
				add_=set()
				if k[1]>threshold:	
					for i in set(dic[k[0]]):
						if i not in ignore_cats:
							add_.add(i)
					if len(add_)>1:
						f.write("|".join(add_)+'\n')
				ignore_cats.update(add_)
	f.close()
	pickle.dump(ignore_cats, open("./wiki_cats/ignore_cats.p", 'wb'))

def name_metacategories():
	"""
	Takes the .txt file where the categories have been divided by line
	Return a .txt file with the name of the meta-categories and their repective categories.
	"""
	f_in=open('./wiki_cats/metacategories.txt', 'r')
	f_out=open('./wiki_cats/metacategories_topics.txt', 'w')
	for line in f_in.read().splitlines():
		li=line.replace("|", " ")
		li=li.split(" ")
		for l in li: 
			if l == "" or l in stop_words:
				li.remove(l)
		cou=Counter(li)
		comm=cou.most_common(len(li))
		try:
			max_f=max([i[1] for i in comm])
		except ValueError:
			continue
		tops=[]
		for c in comm:
			if c[1]>=max_f*0.8 and c[0]!='the':
				tops.append(c[0])
		f_out.write("MAIN TOPIC:\t"+"-".join(tops)+'\n')
		f_out.write("CATEGORIES:\t"+line+'\n')
	f_out.close()
	print("Find the meta-categories with their respective categories in './wiki_cats/metacategories_topics.txt'")

def dic_metacategories(fname):
	"""
	Takes the .txt file where the meta-categories and their respective categories are. 
	Returns a dictionary with the categories as keys and the metacategories as values
	"""
	dic_meta={}
	f_in=open(fname, 'r')
	for line in f_in.read().splitlines():
		if line.startswith("MAIN"):
			metacat=line.split("\t")[1]
			continue
		if line.startswith("CATEGORIES"):
			cats=line.split("\t")[1]
			cats=cats.split('|')
			for cat in cats:
				dic_meta[cat]=metacat
	f_in.close()
	return dic_meta

def distribution_metacategories(f_urls_docs, save):
	""" 
	Returns a .txt file with the distribution of the meta-categories of the dataset 
	extracted from Wikipedia's external links.
	"""
	dic_metacats=dic_metacategories('./wiki_cats/metacategories_topics.txt')
	docs_dic=defaultdict(set)
	with open(f_urls_docs,'r') as f:
		doc=""
		for l in f:
			l = l.rstrip('\n')
			if l[:4] == "<doc":
				m = re.search(".*url=([^ ]*) ",l)
				url=m.group(1)
				m=re.search(r"(?<=categories=)(.*)(?=>)", l)
				cats=m.group(1)
				cats=cats.split("|")
				continue
			if l[:5] != "</doc" and l[:4] != "<doc":
				doc += l + " "
				continue
			if l[:5] == "</doc":
				for cat in cats:
					if cat in dic_metacats.keys():
						docs_dic[dic_metacats[cat]].add((doc, url))
				doc=""
				url=""
				cats=""
				continue
		f.close()

	if save=="True":
		count_dic={}
		for k in docs_dic.keys():
			count_dic[k]=len(docs_dic[k])
		count_dic=dict(sorted(count_dic.items(), key = lambda x: x[1], reverse=True))

		with open('./wiki_cats/distrib_metacategories.txt', 'w') as f:
			for met, n in count_dic.items():
				f.write(met+"\t"+str(n)+'\n')
		f.close()
		print("Find the distribution of pages extracted per metacategory in './wiki_cats/distrib_metacategories.txt'")
	return docs_dic

if __name__ == '__main__':
    args = docopt(__doc__, version='Understanding and filtering Wikipedia categories, ver 0.1')
    linksfolder=args['--linksfolder']
    function = int(args["--function"])

    pathlib.Path('./wiki_cats').mkdir(parents=True, exist_ok=True)

    txt_files = glob.glob(linksfolder+"*links.txt.gz")
    f_urls_docs="./wiki_cats/connect_cats_text.txt"

    if function==0:
    	connect_cats_text(txt_files, f_urls_docs)
    	count_categories(f_urls_docs)

    if function == 1:
    	if not os.path.isfile("./wiki_cats/distrib_categories.txt"):
    		connect_cats_text(txt_files, f_urls_docs)
    		count_categories(f_urls_docs)
    	ngram_n = int(input("Insert 1 for unigrams, 2 for bigrams, 3 for trigrams, and so forth: "))
    	create_ngrams(ngram_n)

    if function == 2:
    	if not os.path.isfile("./wiki_cats/distrib_categories.txt"):
    		connect_cats_text(txt_files, f_urls_docs)
    		count_categories(f_urls_docs)
    	if glob.glob("./wiki_cats/ngram*.p") == []:
    		print("\nERROR: You have to run function=1 first.\n")
    		exit()

    	threshold = int(input("Insert minumum frequency of ngrams (threshold): "))	   
    	#create_metacategories(threshold)
    	#name_metacategories()
    	distribution_metacategories(f_urls_docs, "True")