"""Process Wikipedia with UMAP + Fruit Fly
Usage:
  apply_umap_fly.py train --dataset=<path>
  apply_umap_fly.py reduce --model=<path>
  apply_umap_fly.py fly --model=<path> --dataset=<path>
  apply_umap_fly.py label --lang=<lang>
  apply_umap_fly.py (-h | --help)
  apply_umap_fly.py --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --dataset=<path>             Path to wiki dump file (preprocessed with sentencepiece).
  --model=<path>               Path to pretrained UMAP model (in models/umap).
  --lang=<lang>                Language of documents from wikipedia. e.g. pt
"""


from os.path import join, exists
import umap
import joblib
from pathlib import Path
import pickle
from glob import glob

import nltk
import numpy as np
from docopt import docopt
from joblib import Parallel, delayed
import multiprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.cluster import Birch
from sklearn.metrics import pairwise_distances
from collections import Counter
from nltk.corpus import stopwords

from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from utils import read_vocab, hash_dataset_, read_n_encode_dataset, encode_docs
import matplotlib.pyplot as plt
from fly import Fly


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


def generate_cluster_centroids():
    cl2idx_files = glob(join('./processed','*.cl2idx.pkl'))
    clusters2size = {}
    clusters2vecs = {}

    for f in cl2idx_files:
        cl2idx = pickle.load(open(f,'rb'))
        m = joblib.load(f.replace('.cl2idx.pkl','.umap.m'))
        for cl,idx in cl2idx.items():
            if cl in clusters2size:
                clusters2size[cl]+=len(idx)
            else:
                clusters2size[cl] = len(idx)
                clusters2vecs[cl] = np.zeros(umap_dim)

            for i in idx:
                clusters2vecs[cl]+=m[i]

    cm = np.zeros((len(clusters2vecs),umap_dim))
    for cl in clusters2size:
        cm[cl] = clusters2vecs[cl] / clusters2size[cl]

    return cm



def generate_cluster_labels(verbose=False):
    #Merge all cl2titles dictionary files
    print('--- Generating cluster labels ---')
    # stop_words = ['of', 'in', 'and', 'the', 'at', 'from', 'by', 'with', 'for', 'to', 'de', 'a']
    txt_f = open("langs_list.txt", "r")
    dic={}
    for line in txt_f:
        line = line.rstrip("\n").split("\t")
        dic[line[0]]=line[1]
    stop_words=set(stopwords.words(dic[lang]))
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
            keywords.extend([w for w in title.split() if w not in stop_words])
        c = Counter(keywords)
        #category = ' '.join([pair[0]+' ('+str(pair[1])+')' for pair in c.most_common()[:5]])
        category = ' '.join([pair[0] for pair in c.most_common()[:5]])
        if verbose:
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
    Path("./models/umap").mkdir(exist_ok=True, parents=True)
    filename = './models/umap/'+dfile
    joblib.dump(umap_model, filename)

    return csr_matrix(umap_model.transform(train_set)),train_labels


def apply_umap(umap_model, dataset, save=True):
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


def plot_cluster_centroids(centroids, cluster_labels):
    scaler = preprocessing.MinMaxScaler().fit(centroids)
    centroids = scaler.transform(centroids)
    cl_names = [cluster_labels[cl] for cl in range(centroids.shape[0])]
    cosines = 1-pairwise_distances(centroids, metric="cosine")
    
    for i in range(len(cosines)):
        i_sim = np.array(cosines[i])
        i_label = cl_names[i]
        ranking = np.argsort(-i_sim)
        neighbours = [cl_names[n] for n in ranking][1:11] #don't count first neighbour which is itself
        # print(i_label,neighbours)
    
    centroids_2d = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='hellinger', random_state=32).fit_transform(centroids)

    plt.figure(figsize=(20,40))
    scatter = plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s= 5, cmap='nipy_spectral')
    plt.title('Centroids', fontsize=14)
    for i, txt in enumerate(cl_names):
        plt.annotate(i, (centroids_2d[i][0], centroids_2d[i][1]))
        # print(i,txt)
    plt.savefig("centroids.png")

def umap_prec_at_k(spf, k):
    '''Compute precision at k using cluster IDs from Birch model'''
    train_set, train_titles, train_labels = read_n_encode_dataset(spf, vectorizer, logprobs, logprob_power)
    cl2idx = pickle.load(open(spf.replace('sp','cl2idx.pkl'),'rb'))
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

def train_fly(dataset, kc_size, wta, proj_size, k, cluster_labels, num_trial):
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
    #fly = Fly(pn_size, kc_size, wta, proj_size, top_words, init_method, eval_method, proj_store, hyperparameters)
    fly_list = [Fly(pn_size, kc_size, wta, proj_size, top_words, init_method, eval_method, proj_store, hyperparameters) for _ in range(num_trial)]
    '''Compute precision at k using cluster IDs from Birch model'''
    #score, hashed_data = fly.evaluate(umap_mat,umap_mat,umap_labels,umap_labels)
    with Parallel(n_jobs=max_thread, prefer="threads") as parallel:
        delayed_funcs = [delayed(lambda x:x.evaluate(umap_mat,umap_mat,umap_labels,umap_labels))(fly) for fly in fly_list]
        scores = parallel(delayed_funcs)
    score_list = np.array([p[0] for p in scores])
    print(score_list)
    best = np.argmax(score_list)
    #sanity_check(hashed_data,train_titles,umap_labels)
    return score_list[best], fly_list[best]

def apply_fly(spf,fly_path):
    data_set, data_titles, data_labels = read_n_encode_dataset(spf, vectorizer, logprobs, logprob_power)
    umap_mat = joblib.load(spf.replace('.sp','.umap.m'))
    cl2idx = pickle.load(open(spf.replace('sp','cl2idx.pkl'),'rb'))
    idx2cl = {}
    for cl,idx in cl2idx.items():
        for i in idx:
            idx2cl[i] = cluster_labels[cl]
    umap_labels = [idx2cl[i] for i in range(len(idx2cl))]
    fly = joblib.load(fly_path)
    #Compute precision at k using cluster IDs from Birch model
    score, hashed_data = fly.evaluate(umap_mat,umap_mat,umap_labels,umap_labels)

    #Save hashes 
    title2hash = {}
    for i in range(hashed_data.shape[0]):
        b = hashed_data[i][0].todense()
        #Transform long binary array into an int
        bin_id = b.dot(2**np.arange(b.size)[::-1])[0,0]
        #print(bin_id,data_titles[i],umap_labels[i])
        title2hash[data_titles[i]] = bin_id
    hfile = spf.replace('.sp','.fh')
    joblib.dump(title2hash, hfile)
    return score


if __name__ == '__main__':
    args = docopt(__doc__, version='Wikipedia clustering, 0.1')
    if args["--dataset"]:
        dataset = args["--dataset"]+".sp"
        lang = dataset.split("/")[-1][:2]
    if args["--model"]:
        model_path = args["--model"]+".umap"
        lang = model_path.split("/")[-1][:2]
    if args["--lang"]:
        lang=args["--lang"]
    train = True if args["train"] else False
    reduce_and_cluster = True if args["reduce"] else False
    fly = True if args["fly"] else False
    label_clusters = True if args["label"] else False
    
    # global variables
    spm_vocab = f"../../spm/spm.{lang}wiki.vocab"
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    logprob_power=7 #From experiments on wiki dataset
    umap_dim=31
    max_thread = int(multiprocessing.cpu_count() * 0.2)

    if train:
        print('Dataset name:', dataset)
        train_models()

    elif reduce_and_cluster:
        umap_model = joblib.load(model_path)
        birch_model = joblib.load(model_path.replace('.umap','.birch'))
        
        sp_files = glob(join('./processed','*.sp'))
        for spf in sp_files:
            if not exists(spf.replace('.sp','.umap.m')):
                m, _, _ = apply_umap(umap_model,spf,True)
                print("Output matrix shape:", m.shape)
                apply_clustering(birch_model,spf,True)

    elif label_clusters:
        cluster_labels = generate_cluster_labels(verbose=True)
        centroids = generate_cluster_centroids()
        # for k,v in cluster_labels.items():
        #     print(k,v,centroids[k], "\n")
        plot_cluster_centroids(centroids, cluster_labels)

    elif fly:
        k = 20
        umap_model = joblib.load(model_path)
        print('\n--- Generating cluster labels for fly training ---')
        cluster_labels = generate_cluster_labels(verbose=False)
        print('\n---Training fruit fly ---')
        fly_score, fly = train_fly(dataset, 256, 50, 4, k, cluster_labels, 10)
        print("FLY SCORE FOR TRAIN SET ",k,":",fly_score)
        fly_path = './models/flies/'+dataset.split('/')[-1].replace('sp','fly.m')
        joblib.dump(fly,fly_path)
        
        sp_files = glob(join('./processed','*.sp'))
        for spf in sp_files:
            print("\n--- Putting",spf,"through trained fly ---")
            umap_score = umap_prec_at_k(spf,k)
            print("UMAP SCORE AT ",k,":",umap_score)
            fly_score = apply_fly(spf,fly_path) 
            print("FLY SCORE AT ",k,":",fly_score)

    else:
        print("You shouldn't be here.")

