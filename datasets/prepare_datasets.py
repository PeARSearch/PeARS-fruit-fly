"""Fruit Fly dataset preparation

Usage:
  prepare_datasets.py [--no_meta] [--spm=<str>]
  prepare_datasets.py (-h | --help)
  prepare_datasets.py --version

Options:
  --no_meta          Return plain text, other wise keep meta tag <id, class> in each document.
  --spm=<str>        If the arg is not presence, do not apply sentencepeice.
                     If you want to apply a global sentencepeice for all dataset, pass the path of the model.
                     If you want to train a sentencepeice model for each dataset, pass "per_dataset".
  -h --help          Show this screen.
  --version          Show version.

"""

import urllib
import zipfile
import requests
import random
import tarfile
import pathlib
import shutil
import os
import re
import glob
from os.path import join
from docopt import docopt
import sentencepiece as spm
import pandas as pd
import numpy as np

RANDOM_SEED = 111


# util
def train_sentencepiece(dataset_name, txt_path):
    pathlib.Path(f"../spm").mkdir(parents=True, exist_ok=True)

    f = open(f"../spm/{dataset_name}_train_raw.txt", 'w')
    with open(txt_path) as txt_file:
        for line in txt_file:
            f.write(line.lower().replace('\n', " ") + '\n')
    f.close()

    print('training sentencepeice...')
    spm.SentencePieceTrainer.train(f'--input=../spm/{dataset_name}_train_raw.txt \
    --model_prefix=../spm/spm.{dataset_name} --vocab_size=10000 --minloglevel=2')

    os.remove(f"../spm/{dataset_name}_train_raw.txt")
    model_path = f'../spm/spm.{dataset_name}' + '.model'
    print('sentencepeice model is at ', model_path)
    return model_path


################################ Wiki ##########################################
def preprocess_wikipedia(train_p: float, val_p: float,
                         keep_meta: bool, is_sp_encode: bool, sp_model_path: str):
    """
    :param train_p: train percentage, from 0 to 1
    :param val_p: validation percentage, from 0 to 1
    :param keep_meta: keep the meta tag <id, class> in each document
    :param is_sp_encode: if use sentencepeice to preprocess documents
    :param sp_model_path: path to sentencepeice model, only use when is_sp_encode is True
    """
    if is_sp_encode:
        file_ext = '.sp'
        if sp_model_path:
            sp.load(sp_model_path)
    else:
        file_ext = '.txt'

    # read the dataset to pandas dataframe
    meta_files = glob.glob('./pears-fruit-fly-wikipedia-raw/wiki_dataset_raw/dataset/*/meta.csv')
    df_dataset = []
    for file in meta_files:
        df_dataset.append(pd.read_csv(file))
    df_dataset = pd.concat(df_dataset)
    df_dataset = df_dataset.drop_duplicates(subset=['ID'])
    df_dataset = df_dataset.set_index('ID')
    df_dataset.reset_index(inplace=True)
    df_dataset['doc'] = ''

    data_files = glob.glob('./pears-fruit-fly-wikipedia-raw/wiki_dataset_raw/dataset/*/data.txt')
    doc = ""
    for file in data_files:
        with open(file) as f:
            for l in f:
                l = l.rstrip('\n')
                if l[:4] == "<doc":
                    m = re.search(".*id=([^ ]*) ", l)
                    ID = m.group(1)
                    ID = int(ID.strip('"'))
                elif l[:5] == "</doc":
                    if is_sp_encode:
                        ll = sp.encode_as_pieces(doc)
                        doc = ' '.join([wp for wp in ll])
                    mask = df_dataset['ID'] == ID
                    pos = np.flatnonzero(mask)[0]
                    df_dataset.at[pos, 'doc'] = doc
                    doc = ""
                else:
                    doc += l + ' '

    # shuffle the dataset
    df_dataset = df_dataset.sample(frac=1, random_state=RANDOM_SEED)

    # split and write to files the the splits: train, val, test
    df_train = df_dataset.iloc[0: int(len(df_dataset)*train_p)]
    with open(f'./wikipedia/wikipedia-train{file_ext}', 'w') as f:
        for index, row in df_train.iterrows():
            if keep_meta:
                f.write("<doc id=" + str(row['ID']) + " class=" + row['label'] + ">\n")
            f.write(row['doc'] + '\n')
            if keep_meta:
                f.write("</doc>\n")

    df_val = df_dataset.iloc[int(len(df_dataset)*train_p): int(len(df_dataset)*(train_p+val_p))]
    with open(f'./wikipedia/wikipedia-val{file_ext}', 'w') as f:
        for index, row in df_val.iterrows():
            if keep_meta:
                f.write("<doc id=" + str(row['ID']) + " class=" + row['label'] + ">\n")
            f.write(row['doc'] + '\n')
            if keep_meta:
                f.write("</doc>\n")

    df_test = df_dataset.iloc[int(len(df_dataset)*(train_p+val_p)):]
    with open(f'./wikipedia/wikipedia-test{file_ext}', 'w') as f:
        for index, row in df_test.iterrows():
            if keep_meta:
                f.write("<doc id=" + str(row['ID']) + " class=" + row['label'] + ">\n")
            f.write(row['doc'] + '\n')
            if keep_meta:
                f.write("</doc>\n")

    # write the number of docs in each part
    with open('./wikipedia/wikipedia_stat.txt', 'w') as f:
        f.write(str(len(df_train)) + ' ' + str(len(df_val)) + ' ' + str(len(df_test)))


def prepare_wikipedia(keep_meta: bool, is_sp_encode: bool, sp_model_path: str):
    """
    :param keep_meta: keep the meta tag <id, class> in each document
    :param is_sp_encode: if use sentencepeice to preprocess documents
    :param sp_model_path: path to sentencepeice model, only use when is_sp_encode is True
    """

    url = 'http://pearsproject.org/static/datasets/pears-fruit-fly-wikipedia-raw.zip'
    extract_dir = '.'

    print('downloading...')
    zip_path, _ = urllib.request.urlretrieve(url)
    print('processing...')
    with zipfile.ZipFile(zip_path, 'r') as f:
        f.extractall(extract_dir)
    urllib.request.urlcleanup()

    print('processing...')
    if is_sp_encode:
        if not sp_model_path:  # need to train sentencepeice
            # create plain txt file
            pathlib.Path('./wikipedia').mkdir(parents=True, exist_ok=True)
            preprocess_wikipedia(train_p=0.6,
                                 val_p=0.2,
                                 keep_meta=False,
                                 is_sp_encode=False,
                                 sp_model_path=None)
            sp_model_path = train_sentencepiece(dataset_name='wikipedia',
                                                txt_path='./wikipedia/wikipedia-train.txt')
            shutil.rmtree('./wikipedia')

    pathlib.Path('./wikipedia').mkdir(parents=True, exist_ok=True)
    preprocess_wikipedia(train_p=0.6,
                         val_p=0.2,
                         keep_meta=keep_meta,
                         is_sp_encode=is_sp_encode,
                         sp_model_path=sp_model_path)

    # remove unused files and folders
    shutil.rmtree('./pears-fruit-fly-wikipedia-raw')


################################# WoS ##############################################
def preprocess_wos(text_file: str, label_file: str, train_p: float, val_p: float,
                   keep_meta: bool, is_sp_encode: bool, sp_model_path: str):
    """
    :param text_file: path
    :param label_file: path
    :param train_p: train percentage, from 0 to 1
    :param val_p: validation percentage, from 0 to 1
    :param keep_meta: keep the meta tag <id, class> in each document
    :param is_sp_encode: if use sentencepeice to preprocess documents
    :param sp_model_path: path to sentencepeice model, only use when is_sp_encode is True
    """

    if is_sp_encode:
        file_ext = '.sp'
        if sp_model_path:
            sp.load(sp_model_path)
    else:
        file_ext = '.txt'

    # read label file
    with open(label_file) as f:
        labels = f.readlines()
    label_idx = 0

    # read text file
    doc_list = []
    with open(text_file, encoding="utf8", errors='ignore') as f:
        for l in f:
            l = l.rstrip('\n')
            label = labels[label_idx].rstrip('\n')
            doc = ''
            if keep_meta:
                doc += "<doc id="+str(label_idx)+" class="+label+">\n"
            if is_sp_encode:
                doc += ' '.join([wp for wp in sp.encode_as_pieces(l)])+'\n'
            else:
                doc += l+'\n'
            if keep_meta:
                doc += "</doc>\n"
            doc_list.append(doc)
            label_idx += 1

    # shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(doc_list)
    doc_train = doc_list[0: int(len(doc_list)*train_p)]
    doc_val = doc_list[int(len(doc_list)*train_p): int(len(doc_list)*(train_p+val_p))]
    doc_test = doc_list[int(len(doc_list)*(train_p+val_p)):]

    with open(f"./wos/wos11967-train{file_ext}", 'w') as f:
        for doc in doc_train:
            f.write(doc)
    with open(f"./wos/wos11967-val{file_ext}", 'w') as f:
        for doc in doc_val:
            f.write(doc)
    with open(f"./wos/wos11967-test{file_ext}", 'w') as f:
        for doc in doc_test:
            f.write(doc)

    # write the number of docs in each part
    with open('./wos/wos11967_stat.txt', 'w') as f:
        f.write(str(len(doc_train)) + ' ' + str(len(doc_val)) + ' ' + str(len(doc_test)))


def prepare_wos(keep_meta: bool, is_sp_encode: bool, sp_model_path: str):
    """
    :param keep_meta: keep the meta tag <id, class> in each document
    :param is_sp_encode: if use sentencepeice to preprocess documents
    :param sp_model_path: path to sentencepeice model, only use when is_sp_encode is True
    """

    url = 'https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded'

    print('downloading...')
    r = requests.get(url)
    with open('./WebOfScience.zip', 'wb') as outfile:
        outfile.write(r.content)
    with zipfile.ZipFile('./WebOfScience.zip', 'r') as f:
        f.extractall('./WebOfScience')

    print('processing...')
    if is_sp_encode:
        if not sp_model_path:  # need to train sentencepeice
            # create plain txt file
            pathlib.Path('./wos').mkdir(parents=True, exist_ok=True)
            preprocess_wos(text_file='./WebOfScience/WOS11967/X.txt',
                           label_file='./WebOfScience/WOS11967/Y.txt',
                           train_p=0.6,
                           val_p=0.2,
                           keep_meta=False,
                           is_sp_encode=False,
                           sp_model_path=None)
            sp_model_path = train_sentencepiece(dataset_name='wos',
                                                txt_path='./wos/wos11967-train.txt')
            shutil.rmtree('./wos')

    pathlib.Path('./wos').mkdir(parents=True, exist_ok=True)
    preprocess_wos(text_file='./WebOfScience/WOS11967/X.txt',
                   label_file='./WebOfScience/WOS11967/Y.txt',
                   train_p=0.6,
                   val_p=0.2,
                   keep_meta=keep_meta,
                   is_sp_encode=is_sp_encode,
                   sp_model_path=sp_model_path)

    # remove unused files and folders
    shutil.rmtree('./WebOfScience')
    os.remove('./WebOfScience.zip')


############################# 20news #########################
def preprocess_20news(is_train: bool, keep_meta: bool, is_sp_encode: bool, sp_model_path: str):
    """
    :param is_train: True if using train part, False if using test part
    :param keep_meta: keep the meta tag <id, class> in each document
    :param is_sp_encode: if use sentencepeice to preprocess documents
    :param sp_model_path: path to sentencepeice model, only use when is_sp_encode is True
    """

    if is_sp_encode:
        file_ext = '.sp'
        if sp_model_path:
            sp.load(sp_model_path)
    else:
        file_ext = '.txt'

    if is_train:
        base_dir = "./20news-bydate/20news-bydate-train"
    else:
        base_dir = "./20news-bydate/20news-bydate-test"

    # get folders in 20_newsgroup corpus
    folders = os.listdir(base_dir)
    # print(folders)

    doc_list = []
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
                        doc += l
            doc_n_meta = ''
            if keep_meta:
                doc_n_meta += "<doc id="+file_ids[i]+" class="+folder+">\n"
            if is_sp_encode:
                doc_n_meta += ' '.join([wp for wp in sp.encode_as_pieces(doc)])+'\n'
            else:
                doc_n_meta += doc
            if keep_meta:
                doc_n_meta += "</doc>\n"
            doc_list.append(doc_n_meta)

    if is_train:  # split the original train set to form validation set
        random.seed(RANDOM_SEED)
        random.shuffle(doc_list)

        # validation set
        with open(f'./20news-bydate/20news-bydate-val{file_ext}', 'w') as f:
            for doc in doc_list[8000:]:
                f.writelines(doc)

        # new training set
        with open(f'./20news-bydate/20news-bydate-train{file_ext}', 'w') as f:
            for doc in doc_list[:8000]:
                f.writelines(doc)

    else:
        with open(f'./20news-bydate/20news-bydate-test{file_ext}', 'w') as f:
            for doc in doc_list:
                f.writelines(doc)


def prepare_20news(keep_meta, is_sp_encode, sp_model_path):
    """
    :param keep_meta: keep the meta tag <id, class> in each document
    :param is_sp_encode: if use sentencepeice to preprocess documents
    :param sp_model_path: path to sentencepeice model, only use when is_sp_encode is True
    """

    url = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'
    extract_dir = './20news-bydate'

    print('downloading...')
    tar_path, _ = urllib.request.urlretrieve(url)
    tar = tarfile.open(tar_path, "r:gz")
    tar.extractall(extract_dir)
    tar.close()
    urllib.request.urlcleanup()

    print('processing...')
    if is_sp_encode:
        if not sp_model_path:  # need to train sentencepeice
            # create plain txt file
            preprocess_20news(is_train=True,
                              keep_meta=False,
                              is_sp_encode=False,
                              sp_model_path=None)
            sp_model_path = train_sentencepiece(dataset_name='20news',
                                                txt_path='./20news-bydate/20news-bydate-train.txt')
            os.remove('./20news-bydate/20news-bydate-train.txt')
            os.remove('./20news-bydate/20news-bydate-val.txt')

    preprocess_20news(is_train=True, keep_meta=keep_meta,
                      is_sp_encode=is_sp_encode, sp_model_path=sp_model_path)
    preprocess_20news(is_train=False, keep_meta=keep_meta,
                      is_sp_encode=is_sp_encode, sp_model_path=sp_model_path)

    # remove unused files and folders
    shutil.rmtree('./20news-bydate/20news-bydate-train')
    shutil.rmtree('./20news-bydate/20news-bydate-test')


if __name__ == '__main__':
    args = docopt(__doc__, version='Fruit Fly Hashing, prepare_datasets 0.1')
    random.seed(RANDOM_SEED)
    sp = spm.SentencePieceProcessor()

    if args['--no_meta']:
        keep_meta = False
        print('Do not keep meta data')
    else:
        keep_meta = True
        print('Keep meta data <id, class>')

    if args['--spm']:
        is_sp_encode = True
        if args['--spm'] == 'per_dataset':
            sp_model_path = None
            print('Train sentencepeice model from scratch')
        else:
            sp_model_path = args['--spm']
            print('Use available sentencepeice model from ', sp_model_path)
    else:
        is_sp_encode = False
        sp_model_path = None
        print('Do not apply sentencepeice model')

    print('\nDataset: Wikipedia')
    prepare_wikipedia(keep_meta=keep_meta,
                      is_sp_encode=is_sp_encode,
                      sp_model_path=sp_model_path)

    print('\nDataset: Web of Science')
    prepare_wos(keep_meta=keep_meta,
                is_sp_encode=is_sp_encode,
                sp_model_path=sp_model_path)

    print('\nDataset: 20newsgroups-bydate')
    prepare_20news(keep_meta=keep_meta,
                   is_sp_encode=is_sp_encode,
                   sp_model_path=sp_model_path)

