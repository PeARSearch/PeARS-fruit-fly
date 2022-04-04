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
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from collections import Counter

from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack, lil_matrix, coo_matrix
from utils import read_vocab, hash_dataset_, read_n_encode_dataset
import matplotlib.pyplot as plt
from fly import Fly

def train_linalg(X,Y):
    w = np.linalg.lstsq(X,Y)[0] # obtaining the parameters
    return w

def test_linalg(X,params,Y_labels,brc):
    pred = np.dot(params.T,X)
    idx2clusters = list(brc.predict(pred))
    return accuracy_score(idx2clusters,Y_labels)

def train_clustering(m):
    print('--- Training Birch ---')
    brc = Birch(threshold=0.3,n_clusters=None)
    brc.fit(m[:50000,:]) #train on first 50k
    
    cfile = dataset.split('/')[-1].replace('.sp','.birch')
    filename = './models/umap/'+cfile
    joblib.dump(brc, filename)


def apply_clustering(brc=None, spf=None, save=True):
    print('--- Cluster matrix using pretrained Birch ---')
    #Cluster points in matrix m, in batches of 20k
    data_set, data_titles, data_labels = read_n_encode_dataset(spf, vectorizer, logprobs, logprob_power)
    m = joblib.load(spf.replace('.sp','.umap.m'))
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
    clusters2titles = {}
    for cl,idx in clusters2idx.items():
        cats = [data_titles[i] for i in idx]
        clusters2titles[cl] = cats
   
    if save:
        pklf = spf.replace('sp','cl2titles.pkl')
        with open(pklf, 'wb') as f:
            pickle.dump(clusters2titles,f)
        
        pklf = spf.replace('sp','cl2idx.pkl')
        with open(pklf, 'wb') as f:
            pickle.dump(clusters2idx,f)


def label_clusters():
    #Merge all cl2titles dictionary files
    stopwords = ['of', 'in', 'and', 'the', 'at', 'from', 'by', 'with', 'for', 'to', 'de', 'a']
    cl2titles_files = glob(join('./processed','*.cl2titles.pkl'))
    clusters2titles = pickle.load(open(cl2titles_files[0],'rb'))

    for f in cl2titles_files[1:]:
        tmp = pickle.load(open(f,'rb'))
        for cl,titles in tmp.items():
            if cl in clusters2titles:
                clusters2titles[cl].extend(titles)
            else:
                clusters2titles[cl] = titles

    #Associate a single category label with each cluster
    cluster_titles = {}
    for k,v in clusters2titles.items():
        keywords = []
        for title in v:
            keywords.extend([w for w in title.split() if w not in stopwords])
        c = Counter(keywords)
        category = ' '.join([pair[0]+' ('+str(pair[1])+')' for pair in c.most_common()[:5]])
        print('\n',k,len(v),category,'\n',v[:20])
        cluster_titles[k] = category

    return cluster_titles


#The default values here are from the BO on our Wikipedia dataset. Alternative in 2D for plotting.
#def train_umap(logprob_power=7, umap_nns=5, umap_min_dist=0.1, umap_components=2):
def train_umap(logprob_power=7, umap_nns=16, umap_min_dist=0.0, umap_components=31):
    print('--- Training UMAP ---')
    train_set, train_titles, train_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    train_set = train_set.todense()[:50000]
    train_labels = train_labels[:50000]
    scaler = preprocessing.MinMaxScaler().fit(train_set)
    train_set = scaler.transform(train_set)
    umap_model = umap.UMAP(n_neighbors=umap_nns, min_dist=umap_min_dist, n_components=umap_components, metric='hellinger', random_state=32).fit(train_set)

    dfile = dataset.split('/')[-1].replace('.sp','.umap')
    filename = './models/umap/'+dfile
    joblib.dump(umap_model, filename)

    return csr_matrix(umap_model.transform(train_set)),train_labels


def apply_umap(umap_model,dataset, save=True):
    print('\n---Applying UMAP to ',dataset)
    data_set, data_titles, data_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
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
    
    if save:
        dfile = dataset.replace('.sp','.umap.m')
        joblib.dump(data_set, dfile)
    return data_set, data_titles, data_labels


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

def umap_prec_at_k(k):
    train_set, train_titles, train_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    cl2idx = pickle.load(open(dataset.replace('sp','cl2idx.pkl'),'rb'))
    idx2cl = {}
    for cl,idx in cl2idx.items():
        for i in idx:
            idx2cl[i] = cl
    umap_labels = [idx2cl[i] for i in range(len(idx2cl))]
    cosines = 1-pairwise_distances(train_set.todense(), metric="cosine")
    scores = []
    for i in range(len(cosines)):
        i_sim = np.array(cosines[i])
        i_label = umap_labels[i]
        ranking = np.argsort(-i_sim)
        neighbours = [umap_labels[n] for n in ranking][1:k+1] #don't count first neighbour which is itself
        score = sum([1 if n == i_label else 0 for n in neighbours]) / k
        scores.append(score)
    return np.mean(scores)

def sanity_check(hashed_data, titles, cats):
    hammings = 1-pairwise_distances(hashed_data.todense(), metric="hamming")
    for i in range(hashed_data.shape[0])[:500]:
        print('\n ***')
        i_sim = np.array(hammings[i])
        ranking = np.argsort(-i_sim)
        neighbours = [(i_sim[n],titles[n],cats[n]) for n in ranking][:11] #don't count first neighbour which is itself
        for n in neighbours:
            print(n)


def apply_fly(kc_size, wta, proj_size, k, cluster_labels):
    train_set, train_titles, train_labels = read_n_encode_dataset(dataset, vectorizer, logprobs, logprob_power)
    umap_mat = joblib.load(dataset.replace('.sp','.umap.m'))
    cl2idx = pickle.load(open(dataset.replace('sp','cl2idx.pkl'),'rb'))
    idx2cl = {}
    for cl,idx in cl2idx.items():
        for i in idx:
            idx2cl[i] = cluster_labels[cl]
    umap_labels = [idx2cl[i] for i in range(len(idx2cl))]
    pn_size = umap_mat.shape[1]
    top_words = pn_size
    init_method = "random"
    eval_method = "similarity"
    proj_store = None
    hyperparameters = {'C':100,'num_iter':200,'num_nns':k}
    fly = Fly(pn_size, kc_size, wta, proj_size, top_words, init_method, eval_method, proj_store, hyperparameters)
    score, hashed_data = fly.evaluate(umap_mat,umap_mat,umap_labels,umap_labels)
    sanity_check(hashed_data,train_titles,umap_labels)
    return score


def evaluate_fly():
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
    linalg = False
    cluster = False
    evaluate = True
    spm_vocab = "../../spm/spm.wiki.vocab"
    
    # global variables
    print('Dataset name:', dataset)
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    logprob_power=7

    if train:
        train_models()

    elif linalg:
        dataset_id = dataset.split('/')[-1].replace('.sp','')
        model_path = "./models/umap/"+dataset_id
        umap_model = joblib.load(model_path+'.umap')
        birch_model = joblib.load(model_path+'.birch')
        
        m, m_labels = apply_umap(umap_model,spf,True)
        print("Output matrix shape:", m.shape)
        apply_clustering(birch_model,m,m_labels,spf,True)

    elif cluster:
        cluster_labels = label_clusters()
        for k,v in cluster_labels.items():
            print(k,v)

    elif evaluate:
        cluster_labels = label_clusters()
        k = 20
        umap_score = umap_prec_at_k(k)
        print("UMAP SCORE AT ",k,":",umap_score)
        dfile = dataset.replace('.sp','.umap.m')
        fly_score = apply_fly(512, 10, 10, k, cluster_labels)
        print("FLY SCORE AT ",k,":",fly_score)

    else:
        dataset_id = dataset.split('/')[-1].replace('.sp','')
        model_path = "./models/umap/"+dataset_id
        umap_model = joblib.load(model_path+'.umap')
        birch_model = joblib.load(model_path+'.birch')
        
        sp_files = glob(join('./processed','*.sp'))

        for spf in sp_files:
            if exists(spf.replace('.sp','.umap.m')):
                apply_clustering(birch_model,spf,True)
            #if not exists(spf.replace('.sp','.umap.m')):
            #    m, m_titles, m_labels = apply_umap(umap_model,spf,True)
            #    print("Output matrix shape:", m.shape)
