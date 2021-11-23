"""Fruit Fly dataset preparation

Usage:
  prepare_datasets.py --spm=<single|per_dataset>
  prepare_datasets.py (-h | --help)
  prepare_datasets.py --version

Options:
  --spm=<single|per_dataset>      Choose a single sentencepiece model for all datasets, or one per dataset.
  -h --help                       Show this screen.
  --version                       Show version.

"""


import urllib
import zipfile
import requests
import random
import tarfile
import pathlib
import shutil
import os
from os.path import join
from docopt import docopt
import sentencepiece as spm


def wikipedia():
    url = 'http://pearsproject.org/static/datasets/pears-fruit-fly-wikipedia.zip'
    extract_dir = '.'

    print('downloading...')
    zip_path, _ = urllib.request.urlretrieve(url)
    print('processing...')
    with zipfile.ZipFile(zip_path, 'r') as f:
        f.extractall(extract_dir)
    urllib.request.urlcleanup()


def output_wordpieces_wos(text_file, label_file, train_p, val_p):
    """
    train_p, val_p: the proportion of traning part and validation part
    They should be > 0 and < 1, e.g. 0.6 and 0.2 (the testing part remains 0.2)
    """

    outfile_train = open("./wos/wos11967-train.sp", 'w')
    outfile_val = open("./wos/wos11967-val.sp", 'w')
    outfile_test = open("./wos/wos11967-test.sp", 'w')
    n_train, n_val, n_test = 0, 0, 0 # count the number of docs for each part

    with open(label_file) as f:
        labels = f.readlines()
    label_idx = 0

    with open(text_file, encoding="utf8", errors='ignore') as f:
        for l in f:
            l = l.rstrip('\n')
            ll = sp.encode_as_pieces(l)
            label = labels[label_idx].rstrip('\n')

            # use random to decide the current doc belongs to train, val, or test
            is_train = random.choices([0, 1], weights=[1-train_p, train_p])[0]
            if is_train:
                outfile_train.write("<doc id="+str(label_idx)+" class="+label+">\n")
                outfile_train.write(' '.join([wp for wp in ll])+'\n')
                outfile_train.write("</doc>\n")
                n_train += 1
            else:
                is_val = random.choices([0, 1], weights=[1-val_p/(1-train_p), val_p/(1-train_p)])[0]
                if is_val:
                    outfile_val.write("<doc id="+str(label_idx)+" class="+label+">\n")
                    outfile_val.write(' '.join([wp for wp in ll])+'\n')
                    outfile_val.write("</doc>\n")
                    n_val += 1
                else:
                    outfile_test.write("<doc id="+str(label_idx)+" class="+label+">\n")
                    outfile_test.write(' '.join([wp for wp in ll])+'\n')
                    outfile_test.write("</doc>\n")
                    n_test += 1
            label_idx += 1

    outfile_train.close()
    outfile_val.close()
    outfile_test.close()

    # write the number of docs in each part
    with open('./wos/wos11967_stat.txt', 'w') as f:
        f.write(str(n_train) + ' ' + str(n_val) + ' ' + str(n_test))


def wos():
    url = 'https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded'

    print('downloading...')
    r = requests.get(url)
    with open('./WebOfScience.zip', 'wb') as outfile:
        outfile.write(r.content)
    with zipfile.ZipFile('./WebOfScience.zip', 'r') as f:
        f.extractall('./WebOfScience')

    print('processing...')
    pathlib.Path('./wos').mkdir(parents=True, exist_ok=True)
    output_wordpieces_wos(text_file='./WebOfScience/WOS11967/X.txt',
                          label_file='./WebOfScience/WOS11967/Y.txt',
                          train_p=0.6, val_p=0.2)

    # remove unused files and folders
    shutil.rmtree('./WebOfScience')
    os.remove('./WebOfScience.zip')

def output_wordpieces_20news(train=True):
    if train:
        out_file = open("./20news-bydate/20news-bydate-train.sp", 'w')
        base_dir = "./20news-bydate/20news-bydate-train"
    else:
        out_file = open("./20news-bydate/20news-bydate-test.sp", 'w')
        base_dir = "./20news-bydate/20news-bydate-test"

    # get folders in 20_newsgroup corpus
    folders = os.listdir(base_dir)
    print(folders)

    for folder in folders:
        d = join(base_dir,folder)
        file_ids = os.listdir(d)
        files = [join(d,file_id) for file_id in file_ids]

        for i in range(len(files)):
            in_file = files[i]
            doc = ""
            with open(in_file, encoding="utf8", errors='ignore') as f:
                for l in f:
                    #Ignore headers
                    words = l.split()
                    if len(words) > 0 and words[0][-1] != ':':
                        doc+=l+'\n'
            ll = sp.encode_as_pieces(doc)
            out_file.write("<doc id="+file_ids[i]+" class="+folder+">\n")
            out_file.write(' '.join([wp for wp in ll])+'\n')
            out_file.write("</doc>\n")
    out_file.close()


def _20news():
    url = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'
    extract_dir = './20news-bydate'

    print('downloading...')
    tar_path, _ = urllib.request.urlretrieve(url)
    tar = tarfile.open(tar_path, "r:gz")
    tar.extractall(extract_dir)
    tar.close()
    urllib.request.urlcleanup()

    print('processing...')
    output_wordpieces_20news(train=True)
    output_wordpieces_20news(train=False)

    # split the original train set to form validation set
    train_text = []
    chunk = ''
    with open('./20news-bydate/20news-bydate-train.sp') as f:
        for line in f:
            if line[:4] == "<doc":
                chunk += line
            elif line[:5] == "</doc":
                chunk += line
                train_text.append(chunk)
                chunk = ''
            else:
                chunk += line
    random.shuffle(train_text)

    # validation set
    with open('./20news-bydate/20news-bydate-val.sp', 'w') as f:
        for doc in train_text[8000:]:
            f.writelines(doc)

    # new training set
    with open('./20news-bydate/20news-bydate-train.sp', 'w') as f:
        for doc in train_text[:8000]:
            f.writelines(doc)

    # remove unused files and folders
    shutil.rmtree('./20news-bydate/20news-bydate-train')
    shutil.rmtree('./20news-bydate/20news-bydate-test')


if __name__ == '__main__':
    random.seed(99)
    args = docopt(__doc__, version='Fruit Fly Hashing, prepare_datasets 0.1')
    sp = spm.SentencePieceProcessor()

    if args['--spm'] == 'single':
        sp.load('../spmwiki.model')
    
        print('\nDataset: Wikipedia')
        wikipedia()
        print('\nDataset: Web of Science')
        wos()
        print('\nDataset: 20newsgroups-bydate')
        _20news()

    else:
        print('\nDataset: Wikipedia')
        sp.load("../spm/spm.wiki.model")
        wikipedia()

        print('\nDataset: Web of Science')
        sp.load("../spm/spm.wos.model")
        wos()

        print('\nDataset: 20newsgroups-bydate')
        sp.load("../spm/spm.20news.model")
        _20news()
