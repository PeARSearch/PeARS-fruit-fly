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
nltk.download('punkt')
nltk.download('stopwords')
 
stop_words = set(stopwords.words('english'))

def count_categories(txt_files):
	"""
	Reads the categories that have extracted with Wikipedia's external links.
	Returns a .txt file with the distribution of categories. 
	"""
	dic_url={}
	dic_cat={}
	pattern_url = "url='(.*?)'>"
	for txt_gz in txt_files:
		with gzip.open(txt_gz,'rt') as f:
			for line in f.read().splitlines():
				if "url='" in line:
					try:
						url = re.search(pattern_url, line).group(1)
						dic_url[url]=0
					except AttributeError:
						continue

		cat_file = txt_gz.replace("links.txt.gz", "links.gz")
		with gzip.open(cat_file,'rt') as f:
			for line in f.read().splitlines():
				line=line.split("|")
				if line[0] in dic_url.keys():
					for cat in line[1:]:
						cat=cat.lower()
						if re.match(r"[A-Za-z0-9]+", cat):
							if cat not in dic_cat.keys():
								dic_cat[cat]=1
							else:
								dic_cat[cat]+=1
	
	cats=Counter(dic_cat)
	most = cats.most_common(len(cats.keys()))

	with open('./wiki_cats/distrib_categories.txt', 'w') as f:
		for t in most:
			f.write(t[0]+"\t"+str(t[1])+'\n')
	f.close()
	print("Find categories' list in './wiki_cats/distrib_categories.txt'")

def read_categories():
	cats=[]
	print("Reading categories...")
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
		for ngram in sorted(ngrams_freq, reverse=True): 
			num=re.findall(r'\d+', ngram)[0]
			ngram_freq=pickle.load(open(ngram, 'rb'))
			dic=pickle.load(open("./wiki_cats/dic"+num+".p", 'rb'))
			for k in ngram_freq:
				add_=set()
				if k[1]>threshold:	
					for i in set(dic[k[0]]):
						if i not in ignore_cats:
							add_.add(i)
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

def distribution_metacategories():
	"""
	Takes the folder where the links.gz and links.txt.gz are and the .txt file where 
	the meta-categories and their respective categories are. 
	Returns a .txt file with the distribution of the meta-categories of the dataset 
	extracted from Wikipedia's external links.
	"""
	cats = read_categories()
	meta_cats = dic_metacategories('./wiki_cats/metacategories_topics.txt')
	final_cats=defaultdict(list)
	
	for cat in cats:
		if cat[0] in meta_cats.keys():
			if meta_cats[cat[0]] not in final_cats.keys():
				final_cats[meta_cats[cat[0]]]=int(cat[1])
			else:
				final_cats[meta_cats[cat[0]]]+=int(cat[1])

	cats=Counter(final_cats)
	most = cats.most_common(len(cats.keys()))

	with open('./wiki_cats/distrib_metacategories.txt', 'w') as f:
		for t in most:
			f.write(t[0]+"\t"+str(t[1])+'\n')
	f.close()
	print("Find the distribution of pages extracted per metacategory in './wiki_cats/distrib_metacategories.txt'")

if __name__ == '__main__':
    args = docopt(__doc__, version='Understanding and filtering Wikipedia categories, ver 0.1')
    linksfolder=args['--linksfolder']
    function = int(args["--function"])

    txt_files = glob.glob(linksfolder+"*links.txt.gz")

    if not os.path.exists("wiki_cats"):
    	os.makedirs("wiki_cats")

    if function==0:
    	count_categories(txt_files)

    if function == 1:
    	if not os.path.isfile("./wiki_cats/distrib_categories.txt"):
    		count_categories(txt_files)
    	ngram_n = int(input("Insert 1 for unigrams, 2 for bigrams, 3 for trigrams, and so forth: "))
    	create_ngrams(ngram_n)

    if function == 2:
    	if not os.path.isfile("./wiki_cats/distrib_categories.txt"):
    		count_categories(txt_files)
    	if glob.glob("./wiki_cats/ngram*.p") == []:
    		print("\nERROR: You have to run function=1 first.\n")
    		exit()

    	threshold = int(input("Insert minumum frequency of ngrams (threshold): "))	   
    	create_metacategories(threshold)
    	name_metacategories()
    	distribution_metacategories()
