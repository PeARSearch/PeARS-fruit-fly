#!/usr/bin/python3

import requests
import os

S = requests.Session()
if not os.path.isdir("./data/categories/"):
    os.mkdir("./data/categories")

URL = "https://en.wikipedia.org/w/api.php"



def read_categories(metacat):
    categories = []
    f = open("./wiki_cats/metacategories_topics.txt")
    for l in f:
        l = l.rstrip('\n')
        topics,cats = l.split('\t')
        for topic in topics[8:].split('|'):
            if topic == metacat:
                categories.extend([c for c in cats.split('|') if "CATEGORIES" not in c])
    return list(set(categories))



if not os.path.isdir("./data"):
    os.mkdir("./data")


metacat = input("Please enter a category name: ")
metacat_dir = "./data/categories/"+metacat.replace(' ','_')
if not os.path.isdir(metacat_dir):
    os.mkdir(metacat_dir)

categories = read_categories(metacat)
print(categories)


for cat in categories:
    cat_dir = os.path.join(metacat_dir,cat.replace(' ','_'))
    if not os.path.isdir(cat_dir):
        os.mkdir(cat_dir)
    title_file = open(os.path.join(cat_dir,"titles.txt"),'w')

    PARAMS = {
        "action": "query",
        "list": "categorymembers",
        "format": "json",
        "cmtitle": "Category:"+cat,
        "cmlimit": "20"
    }

    for i in range(1):    #increase 1 to more to get additional data
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()

        PAGES = DATA["query"]["categorymembers"]

        for page in PAGES:
            title = page["title"]
            ID = str(page["pageid"])
            if title[:9] != "Category:":
                title_file.write(ID+' '+title+'\n')

        if "continue" in DATA:
            PARAMS["cmcontinue"] = DATA["continue"]["cmcontinue"]
        else:
            break

    title_file.close()

