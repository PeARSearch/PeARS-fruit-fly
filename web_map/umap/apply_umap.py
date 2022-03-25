"""Analysis of PN space
Usage:
  apply_umap.py --dataset=<str> 
  apply_umap.py (-h | --help)
  apply_umap.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<str>              Name of file the dataset, either wiki, 20news, or wos  (processed by sentencepiece)
"""


import os
from os.path import join, exists
import re
import umap
import joblib
import pickle
from glob import glob
import sentencepiece as spm
import numpy as np
from datetime import datetime
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.cluster import Birch
from collections import Counter

from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack, lil_matrix, coo_matrix
from utils import read_vocab, hash_dataset_, read_n_encode_dataset
import matplotlib.pyplot as plt


def train_clustering(m):
    print('--- Training Birch ---')
    brc = Birch(n_clusters=None)
    brc.fit(m[:20000,:]) #train on first 20k
    
    cfile = dataset.split('/')[-1].replace('.sp','.birch')
    filename = './models/umap/'+cfile
    joblib.dump(brc, filename)


def apply_clustering(brc,m,m_cats,spf):
    print('--- Cluster matrix using pretrained Birch ---')
    #Cluster points in matrix m, in batches of 20k
    idx2clusters = list(brc.predict(m[:20000,:]))
    clusters2idx = {}
    m = m.todense()

    for i in range(20000,m.shape[0],20000):
        print("Clustering",i,"to",i+20000)
        idx2clusters.extend(list(brc.predict(m[i:i+20000,:])))

    print('--- Save Birch output in cl2cats and cl2idx pickled files (./processed folder) ---')
    #Count items in each cluster, using labels for whole data
    cluster_counts = Counter(idx2clusters)
    print(len(idx2clusters),cluster_counts)

    #Make dictionary clusters to idx
    for cl in cluster_counts:
        clusters2idx[cl] = []
    for idx,cl in enumerate(idx2clusters):
        clusters2idx[cl].append(idx)
    
    #Make a dictionary clusters to list of categories
    clusters2cats = {}
    for cl,idx in clusters2idx.items():
        cats = [m_cats[i] for i in idx]
        clusters2cats[cl] = cats
    
    pklf = spf.replace('sp','cl2cats.pkl')
    with open(pklf, 'wb') as f:
        pickle.dump(clusters2cats,f)
    
    pklf = spf.replace('sp','cl2idx.pkl')
    with open(pklf, 'wb') as f:
        pickle.dump(clusters2idx,f)


def label_clusters():
    #Merge all cl2cats dictionary files
    cl2cats_files = glob(join('./processed','*.cl2cats.pkl'))
    clusters2cats = pickle.load(open(cl2cats_files[0],'rb'))

    for f in cl2cats_files[1:]:
        tmp = pickle.load(open(f,'rb'))
        for cl,cats in tmp.items():
            clusters2cats[cl].extend(cats)

    #Associate a single category label with each cluster
    cluster_cats = {}
    for k,v in clusters2cats.items():
        keywords = []
        for cat in v:
            keywords.extend(cat.split())
        c = Counter(keywords)
        category = ' '.join([pair[0]+' ('+str(pair[1])+')' for pair in c.most_common()[:5]])
        #print(k,category, np.sum(m[clusters[k]], axis=0) / len(clusters[k]))
        print(k,category)
        cluster_cats[k] = category

    return cluster_cats


#The default values here are from the BO on our Wikipedia dataset. Alternative in 2D for plotting.
#def train_umap(logprob_power=7, umap_nns=5, umap_min_dist=0.1, umap_components=2):
def train_umap(logprob_power=7, umap_nns=16, umap_min_dist=0.0, umap_components=31):
    print('--- Training UMAP ---')
    train_set, train_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    train_set = train_set.todense()[:20000]
    train_labels = train_labels[:20000]
    scaler = preprocessing.MinMaxScaler().fit(train_set)
    train_set = scaler.transform(train_set)
    umap_model = umap.UMAP(n_neighbors=umap_nns, min_dist=umap_min_dist, n_components=umap_components, metric='hellinger', random_state=32).fit(train_set)

    dfile = dataset.split('/')[-1].replace('.sp','.umap')
    filename = './models/umap/'+dfile
    joblib.dump(umap_model, filename)

    return csr_matrix(umap_model.transform(train_set)),train_labels


def apply_umap(umap_model,dataset):
    print('\n---Applying UMAP to ',dataset)
    logprob_power=7
    data_set, data_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    scaler = preprocessing.MinMaxScaler().fit(data_set.todense())
    data_set = scaler.transform(data_set.todense())
    m = csr_matrix(umap_model.transform(data_set[:20000,:]))

    for i in range(20000,data_set.shape[0],20000):
        print("Reducing",i,"to",i+20000)
        m2=csr_matrix(umap_model.transform(data_set[i:i+20000,:]))
        #print(m.shape,m2.shape)
        m = vstack((m,m2))
        #print("New m shape:",m.shape)
    data_set = np.nan_to_num(m)
    
    dfile = dataset.replace('.sp','.umap.m')
    joblib.dump(data_set, dfile)
    return data_set, data_labels


def train_models():
    m, _ = train_umap()
    train_clustering(m)


def plot_clusters():
    idx2clusters, cluster_cats = cluster(train_set,train_label)
    print(len(idx2clusters))

    plt.figure(figsize=(12,8))
    train_set = np.array(train_set)
    cl_names = list(set([cluster_cats[cl] for cl in idx2clusters]))
    scatter = plt.scatter(train_set[:, 0], train_set[:, 1], s= 5, c=idx2clusters, cmap='nipy_spectral')
    plt.title('Embedding of the '+dataset+' training set by UMAP', fontsize=14)
    #plt.legend(handles=scatter.legend_elements()[0], labels=cl_names, title="Categories")
    plt.savefig("./umap.png")



def run_fly():
    fly = joblib.load('./models/flies/'+dataset+'.fly.m')
    fly.eval_method = 'similarity'
    fly.hyperparameters['num_nns']=100
    print("\nEvaluating on test set with similarity:")
    fly.evaluate(train_set,test_set,train_label,test_label)

    hash_train, _, _ = hash_dataset_(dataset_mat=train_set, weight_mat=fly.projections, percent_hash=fly.wta)



if __name__ == '__main__':
    args = docopt(__doc__, version='Wikipedia clustering, 0.1')
    dataset = args["--dataset"]
    train = False
    spm_vocab = "../../spm/spm.wiki.vocab"
    
    # global variables
    print('Dataset name:', dataset)
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')

    if train:
        train_models()
   
    dataset_id = dataset.split('/')[-1].replace('.sp','')
    model_path = "./models/umap/"+dataset_id
    umap_model = joblib.load(model_path+'.umap')
    birch_model = joblib.load(model_path+'.birch')
    
    sp_files = glob(join('./processed','*.sp'))

    for spf in sp_files:
        if not exists(spf.replace('.sp','.umap.m')):
            m, m_labels = apply_umap(umap_model,spf)
            print("Output matrix shape:", m.shape)
            apply_clustering(birch_model,m,m_labels,spf)

    label_clusters()
