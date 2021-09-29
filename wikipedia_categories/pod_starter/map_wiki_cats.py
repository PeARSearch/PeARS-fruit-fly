"""Understanding and filtering the wikipedia categories that have been extracted from 
Wikipedia's external links

Usage:
  map_wiki_cats.py
  map_wiki_cats.py (-h | --help)
  map_wiki_cats.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.

"""
from docopt import docopt
import gzip
import glob
from collections import Counter, defaultdict
import pickle
import nltk
import shutil
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import re
import os
 
stop_words = set(stopwords.words('english'))
meta_classes = ["Wikipedia","Articles","Category","Templates","Stubs","A-Class","B-Class","C-Class","Draft-Class","Stub-Class","List-Class","Start-Class","WikiProject","Low-Importance","Mid-Importance","High-Importance","redirects","infobox","files","Wikipedians"]

def read_categories():
	cats=[]
	print("Reading categories...")
	with open('./data/preprocessed_wiki_categories.txt', 'r') as f:
	    for line in f.read().splitlines():
	        cats.append((line, "0"))
	f.close()
	return cats

def get_ngram_list(txt, ngram_n):
    txt = txt.lower()
    meta_cl = [c.lower() for c in meta_classes]
    tokens = word_tokenize(txt)
    if ngram_n == 1:
        tokens = [i for i in tokens if i not in stop_words and i.isalpha() and i not in meta_cl]
    else:
        tokens = [i for i in tokens if i.isalpha() and i not in meta_classes]
    all_ngrams = list(ngrams(tokens, ngram_n))
    return all_ngrams

def create_ngrams(ngram_n):
    """
    Returns a .txt file with the ngrams and their frequency in the categories. 
    """
    print("\nComputing ngrams of size",ngram_n)
    cats=read_categories()
    cats=[i[0] for i in cats if i[0]!=""]
    ngrams_l=[]
    dic=defaultdict(list)
    for cat in cats:
        all_ngrams = get_ngram_list(cat, ngram_n)
        for ngram in all_ngrams:
            ngrams_l.append(ngram)
            if ngram in dic:
                dic[ngram].append(cat)
            else:
                dic[ngram] = [cat]
    fdist = nltk.FreqDist(ngrams_l)
    sort_orders = sorted(fdist.items(), key=lambda x: x[1], reverse=True)
    pickle.dump(sort_orders, open("./wiki_cats/ngram"+str(ngram_n)+".p", 'wb'))
    pickle.dump(dic, open("./wiki_cats/dic"+str(ngram_n)+".p", 'wb'))

    with open("./wiki_cats/"+str(ngram_n)+"grams.txt", 'w') as f:
        for k in sort_orders:
            f.write(",".join(k[0])+"\t"+str(k[1])+'\n')
    print("Find the ngrams in the file: './wiki_cats/"+str(ngram_n)+"grams.txt'.")
    f.close()


def create_metacategories(threshold):
    """
    Takes the threshold of the frequency of the ngrams. 
    Returns a .txt file with categories that belong to the same meta-category in each line.
    """
    print("Creating metacategories")
    if os.path.isfile("./wiki_cats/ignore_cats.p"):
        ignore_cats=pickle.load(open("./wiki_cats/ignore_cats.p", 'rb'))
    else:
        ignore_cats=set()
        ngrams_freq=glob.glob("./wiki_cats/ngram*.p")
        with open("./wiki_cats/metacategories.txt", 'w') as f:
            for ngram in ngrams_freq:
                ngram_size=re.findall(r'\d+', ngram)[0]
                ngram_freq=pickle.load(open(ngram, 'rb'))
                dic=pickle.load(open("./wiki_cats/dic"+ngram_size+".p", 'rb'))
                for k in ngram_freq:
                    add_=set()
                    if k[1] < threshold:
                        continue
                    for i in set(dic[k[0]]):
                        discard=False
                        for c in meta_classes:
                            if c.lower() in i.lower().split() or i in ignore_cats:
                                discard=True
                                break
                        if not discard:
                            add_.add(i)
                    f.write("|".join(add_)+'\n')
        f.close()

def name_metacategories():
    """
    Takes the .txt file where the categories have been divided by line
    Return a .txt file with the name of the meta-categories and their repective categories.
    """
    f_in=open('./wiki_cats/metacategories.txt', 'r')
    f_out=open('./wiki_cats/metacategories_topics.txt', 'w')
    for line in f_in.read().splitlines():
        cats = line.split('|')
        ngram_n = 6
        name_found = False
        while not name_found and ngram_n > 0:
            ngrams = []
            for item in cats:
                ngram_strs = [' '.join(li) for li in get_ngram_list(item,ngram_n)]
                ngrams.extend(ngram_strs)
            cou=Counter(ngrams) #count ngrams occurring most often in subcategory names
            comm=cou.most_common(len(cou))
            tops = [p[0] for p in comm if p[1] > 0.99 * len(cats)]
            if len(tops) > 0:
                name_found = True
                f_out.write("TOPICS: "+" | ".join(tops)+'\t')
                f_out.write("CATEGORIES: "+line+'\n')
            else:
                ngram_n-=1
    f_out.close()
    f_in.close()
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

    
    if os.path.exists("./wiki_cats"):
        shutil.rmtree("./wiki_cats")
    os.makedirs("./wiki_cats")

    for i in [2,3,4,5,6]:
        create_ngrams(i)

    create_metacategories(10)
    name_metacategories()
    #distribution_metacategories()
