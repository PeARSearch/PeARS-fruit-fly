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
from os.path import join
import re
import umap
import joblib
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


def apply_clustering(brc,m,m_cats):
    #Cluster points in matrix m, in batches of 20k
    m = m.todense()
    idx2clusters = list(brc.predict(m[:20000,:]))

    for i in range(20000,m.shape[0],20000):
        print("Clustering",i,"to",i+20000)
        idx2clusters.extend(list(brc.predict(m[i:i+20000,:])))

    #Count items in each cluster, using labels for whole data
    cluster_counts = Counter(idx2clusters)
    print(len(idx2clusters),cluster_counts)

    #Make a dictionary of clusters
    #Key = cluster, value = list of all doc categories for that cluster
    clusters2cats = {}
    clusters = {}
    for cl in cluster_counts:
        clusters2cats[cl] = []
        clusters[cl] = []
    for idx,cl in enumerate(idx2clusters):
        clusters2cats[cl].append(m_cats[idx])
        clusters[cl].append(idx)

    #Associate a single category label with each cluster
    cluster_cats = {}
    just_counts = []
    for k,v in clusters2cats.items():
        #print(k,cluster_counts[k],' | '.join(s for s in v))
        keywords = []
        for cat in v:
            keywords.extend(cat.split())
        c = Counter(keywords)
        category = ' '.join([pair[0] for pair in c.most_common()[:5]])
        #print(k,cluster_counts[k],c.most_common()[:5])
        print(k,cluster_counts[k],category, np.sum(m[clusters[k]], axis=0) / len(clusters[k]))
        cluster_cats[k] = category
        just_counts.append(cluster_counts[k])

    #Rewrite idx2clusters to use the category names
    #idx2clusters = [cluster_cats[cl] for cl in idx2clusters]

    print("MEAN CLUSTER SIZE:",np.mean(just_counts), np.std(just_counts))
    return idx2clusters, cluster_cats


def train_umap(logprob_power=7, umap_nns=5, umap_min_dist=0.1, umap_components=2):
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
    data_set = np.nan_to_num(m).todense()
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
    train = True
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

    m, m_labels = apply_umap(umap_model,sp_files[0])
    for spf in sp_files[1:2]:
        print('\n---Applying UMAP to ',spf)
        m2, m2_labels = apply_umap(umap_model,spf)
        m = vstack((m,m2))
        m_labels.extend(m2_labels)
        print(m.shape)

    apply_clustering(birch_model,m,m_labels)

