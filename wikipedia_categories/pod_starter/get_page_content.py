#!/usr/bin/python3

import requests
import glob
import os

def read_titles(filename):
    IDs = []
    titles = []
    f = open(filename,'r')
    for l in f:
        l.rstrip('\n')
        IDs.append(l.split()[0])
        titles.append(' '.join(l.split()[1:]))
    return IDs,titles


def read_categories(metacat_dir):
    categories=glob.glob(metacat_dir+"/*")
    return categories



S = requests.Session()
metacat = input("Please enter a category name: ").replace(' ','_')
metacat_dir = "./data/categories/"+metacat

URL = "https://en.wikipedia.org/w/api.php"


categories = read_categories(metacat_dir)

for cat_dir in categories[:1]:
    print("Processing category",cat_dir)
    title_file = os.path.join(cat_dir,"titles.txt")
    IDs, titles = read_titles(title_file)

    content_file = open(os.path.join(cat_dir,"linear.txt"),'w')

    for i in range(len(titles)):
        PARAMS = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "redirects": True,
            "titles": titles[i]
        }

        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()

        PAGES = DATA["query"]["pages"]

        for page in PAGES:
            extract = PAGES[page]["extract"]
            content_file.write("<doc id="+IDs[i]+" url=https://en.wikipedia.org/wiki/"+titles[i].replace(' ','_')+" class="+metacat+">\n")
            content_file.write(extract+'\n')
            content_file.write("</doc>\n\n")

    content_file.close()
