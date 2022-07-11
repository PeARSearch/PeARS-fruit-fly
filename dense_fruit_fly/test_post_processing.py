"""Test init strategy
Usage:
  fly_search.py --dataset=<str>
  fly_search.py (-h | --help)
  fly_search.py --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --dataset=<str>              Name of dataset, either wiki, 20news, or wos.
"""

import random
import joblib
import pathlib
import numpy as np
import multiprocessing
import sentencepiece as spm
from docopt import docopt
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import lil_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.stats import entropy

from classify import train_model
from sklearn.metrics import pairwise_distances
from utils import read_vocab, hash_dataset_, read_n_encode_dataset
from vectorizer import vectorize_scale
import warnings
warnings.filterwarnings("ignore")


def dim_reduction(X_train, X_val, n_dim, method=None):
    if method == 'pca':
        pca = PCA(n_components=n_dim)
        pca.fit(X_train)
        X_train_tf = pca.transform(X_train)
        X_val_tf = pca.transform(X_val)
        return X_train_tf, X_val_tf
    if method == 'tsne':
        tsne = TSNE(n_components=n_dim, method='exact', n_iter=300)
        X_train_tf = tsne.fit_transform(X_train)
        X_val_tf = tsne.fit_transform(X_val)
        return X_train_tf, X_val_tf
    if method == 'umap':
        umap = UMAP(n_neighbors=10, min_dist=0, n_components=n_dim, metric='hellinger').fit(X_train)
        X_train_tf = umap.transform(X_train)
        X_val_tf = umap.transform(X_val)
        return X_train_tf, X_val_tf
    if method == 'zero':
        zero_stat = np.count_nonzero(X_train, axis=0)
        keep_idx = np.argpartition(zero_stat, n_dim)[:n_dim]
        X_train_prune = X_train[:, keep_idx]
        X_val_prune = X_val[:, keep_idx]
        return X_train_prune, X_val_prune
    if method == 'std':
        std_stat = np.std(X_train, axis=0)
        keep_idx = np.argpartition(std_stat, -n_dim)[-n_dim:]
        X_train_prune = X_train[:, keep_idx]
        X_val_prune = X_val[:, keep_idx]
        return X_train_prune, X_val_prune
    if method == 'entropy':
        entropy_stat = entropy(X_train + np.finfo(float).eps, base=2, axis=0)
        keep_idx = np.argpartition(entropy_stat, -n_dim)[-n_dim:]
        X_train_prune = X_train[:, keep_idx]
        X_val_prune = X_val[:, keep_idx]
        return X_train_prune, X_val_prune
    return X_train, X_val


# def imp_classify():  # imp = iterative magnitude pruning



class FlyTestPostProcessing:
    def __init__(self, pn_size=None, kc_size=None, wta=None, proj_size=None, dim_reduction_method=None, n_dim_reduction=None,
                 eval_method=None, proj_store=None, hyperparameters=None):
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.wta = wta
        self.proj_size = proj_size
        self.eval_method = eval_method
        self.hyperparameters = hyperparameters
        weight_mat, self.shuffled_idx = self.create_projections(proj_size=self.proj_size)
        # weight_mat, self.shuffled_idx = self.create_projections_3(proj_size=self.proj_size, favor_order=std_rank)
        self.projections = lil_matrix(weight_mat)
        self.val_score_c = 0
        self.val_score_s = 0
        self.is_evaluated = False
        self.kc_use_sorted = None
        self.kc_in_hash_sorted = None
        # print("INIT",self.kc_size,self.proj_size,self.wta,self.get_coverage())

        self.dim_reduction_method = dim_reduction_method
        self.n_dim_reduction = n_dim_reduction


    def create_projections(self, proj_size):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        idx = list(range(self.pn_size))
        random.shuffle(idx)
        used_idx = idx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(0, len(idx), proj_size):
                p = idx[i:i + proj_size]
                for j in p:
                    weight_mat[c][j] = 1
                c += 1
                if c >= self.kc_size:
                    break
            random.shuffle(idx)  # reshuffle if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]

    def create_projections_3(self, proj_size, favor_order):  # favor high variance dimensions
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        idx = list(np.random.permutation(favor_order[0:1000])) + list(np.random.permutation(favor_order[1000:]))
        used_idx = idx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(0, len(idx), proj_size):
                p = idx[i:i + proj_size]
                for j in p:
                    weight_mat[c][j] = 1 if np.random.random() < 0.5 else -1
                c += 1
                if c >= self.kc_size:
                    break
            # reshuffle if needed -- if all KCs are not filled
            idx = list(np.random.permutation(favor_order[0:1000])) + list(np.random.permutation(favor_order[1000:]))
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]


    def evaluate(self, train_set, val_set, train_label, val_label):
        # # dim reduction
        # train_set, val_set = dim_reduction(X_train=train_set, X_val=val_set,
        #                                    n_dim=1000, method='pca')

        hash_val, kc_use_val, kc_sorted_val = hash_dataset_(dataset_mat=val_set, weight_mat=self.projections,
                                                            percent_hash=self.wta)
        # if self.eval_method == "classification":
        # We only need the train set for classification, not similarity
        hash_train, kc_use_train, kc_sorted_train = hash_dataset_(dataset_mat=train_set,
                                                                  weight_mat=self.projections,
                                                                  percent_hash=self.wta)

        # dim reduction
        hash_train = hash_train.toarray()
        hash_val = hash_val.toarray()
        hash_train, hash_val = dim_reduction(X_train=hash_train, X_val=hash_val,
                                             n_dim=self.n_dim_reduction, method=self.dim_reduction_method)
        hash_train = (hash_train > 0).astype(np.int_)
        hash_val = (hash_val > 0).astype(np.int_)

        self.val_score_c, _ = train_model(m_train=hash_train, classes_train=train_label,
                                        m_val=hash_val, classes_val=val_label,
                                        C=self.hyperparameters['C'], num_iter=self.hyperparameters['num_iter'])
        #if self.eval_method == "similarity":
        self.val_score_s, kc_in_hash_sorted = self.prec_at_k(m_val=hash_val, classes_val=val_label_prec,
                                                             k=self.hyperparameters['num_nns'])
        self.kc_in_hash_sorted = list(kc_in_hash_sorted)
        self.is_evaluated = True
        # print("\nCOVERAGE:",self.get_coverage())
        # print("SCORE:", self.val_score)
        # self.kc_use = kc_use_val / np.sum(kc_use_val)
        self.kc_use_sorted = list(kc_sorted_val)
        # print("PROJECTIONS:",self.print_projections())
        # print("KC USE:",np.sort(self.kc_use)[::-1][:20])
        return self.val_score_c, self.val_score_s#, self.kc_use_sorted, self.kc_in_hash_sorted
        # return np.random.random(), np.random.random()

    def compute_nearest_neighbours(self, hammings, labels, i, num_nns):
        i_sim = np.array(hammings[i])
        i_label = labels[i]
        ranking = np.argsort(-i_sim)
        neighbours = [labels[n] for n in ranking][1:num_nns + 1]  # don't count first neighbour which is itself
        n_sum = 0
        # print("neighbours: ", neighbours)
        # print("i_label", i_label)
        for n in neighbours:
            for lab in n:
                if lab in i_label:
                    n_sum += 1
                    break
        score = n_sum / num_nns
        return score, neighbours

    def prec_at_k(self, m_val=None, classes_val=None, k=None):
        hammings = 1 - pairwise_distances(m_val, metric="hamming")
        kc_hash_use = np.zeros(m_val.shape[1])
        scores = []
        for i in range(hammings.shape[0]):
            score, neighbours = self.compute_nearest_neighbours(hammings, classes_val, i, k)
            for idx in np.nonzero(m_val[i] == 0)[0]: #m_val[i].indices:
                kc_hash_use[idx] += 1
            scores.append(score)
        kc_hash_use = kc_hash_use / sum(kc_hash_use)
        kc_sorted_hash_use = np.argsort(kc_hash_use)[
                             :-kc_hash_use.shape[0] - 1:-1]  # Give sorted list from most to least used KCs
        return np.mean(scores), kc_sorted_hash_use


def fruitfly_pipeline(kc_size, proj_size, wta, knn, num_trial, save):
    # Below parameters are needed to init the fruit fly, even if not used here
    dim_reduction_list = [None]
    n_dim_list = [64]
    eval_method = ''
    proj_store = None
    hyperparameters = {'C': 100, 'num_iter': 200, 'num_nns': knn}

    fly_list = [FlyTestPostProcessing(PN_SIZE, kc_size, wta, proj_size, dim_reduction_method, n_dim, eval_method, proj_store, hyperparameters)
                for dim_reduction_method in dim_reduction_list for n_dim in n_dim_list for _ in range(num_trial)]
    print('total', len(fly_list), 'flies')
    print('evaluating...')

    with joblib.Parallel(n_jobs=max_thread, prefer="threads") as parallel:
        delayed_funcs = [joblib.delayed(lambda x: x.evaluate(train_set, val_set, train_label, val_label))(fly) for fly in
                         fly_list]
        scores = parallel(delayed_funcs)

    score_c_list = np.array([p[0] for p in scores]).reshape([len(dim_reduction_list), len(n_dim_list), num_trial])
    score_s_list = np.array([p[1] for p in scores]).reshape([len(dim_reduction_list), len(n_dim_list), num_trial])

    # average the validation acc
    avg_score_c = np.mean(score_c_list, axis=2)
    std_score_c = np.std(score_c_list, axis=2)
    # print('average score c:')
    # for i in avg_score_c: print(round(i, 3))

    # best_score_c = max(score_c_list)
    # best_fly_c = fly_list[score_c_list.index(best_score_c)]
    # if save:
    #     filename = './models/flies/' + dataset + '.fly_c.m'
    #     joblib.dump(best_fly_c, filename)

    # average the validation acc
    avg_score_s = np.mean(score_s_list, axis=2)
    std_score_s = np.std(score_s_list, axis=2)
    # print('average score s:')
    # for i in avg_score_s: print(round(i, 3))

    # best_score_s = max(score_s_list)
    # best_fly_s = fly_list[score_s_list.index(best_score_s)]
    # if save:
    #     filename = './models/flies/' + dataset + '.fly_s.m'
    #     joblib.dump(best_fly_s, filename)

    print(avg_score_c)
    print(avg_score_s)
    return avg_score_c, avg_score_s


if __name__ == '__main__':
    args = docopt(__doc__, version='Test post processing strategy, ver 0.1')
    dataset = args["--dataset"]
    power = 1

    if dataset == "wiki" or dataset == "enwiki":
        train_path = "../datasets/wikipedia/wikipedia-train.sp"
        spm_model = "../spm/spm.enwiki.model"
        spm_vocab = "../spm/spm.enwiki.vocab"
        # umap_model = joblib.load("./models/umap/wiki.umap")
    if dataset == "20news":
        train_path = "../datasets/20news-bydate/20news-bydate-train.sp"
        spm_model = "../spm/spm.20news.model"
        spm_vocab = "../spm/spm.20news.vocab"
        # umap_model = joblib.load("./models/umap/20news.umap")
    if dataset == "wos":
        train_path = "../datasets/wos/wos11967-train.sp"
        spm_model = "../spm/spm.wos.model"
        spm_vocab = "../spm/spm.wos.vocab"
        # umap_model = joblib.load("./models/umap/wos.umap")
    if dataset == "tmc":
        train_path = "../datasets/tmc/tmc-train.sp"
        spm_model = "../spm/spm.tmc.model"
        spm_vocab = "../spm/spm.tmc.vocab"
    if dataset == "reuters":
        train_path = "../datasets/reuters/reuters-train.sp"
        spm_model = "../spm/spm.reuters.model"
        spm_vocab = "../spm/spm.reuters.vocab"
    if dataset == "agnews":
        train_path = "../datasets/agnews/agnews-train.sp"
        spm_model = "../spm/spm.agnews.model"
        spm_vocab = "../spm/spm.agnews.vocab"

    dataset_name = train_path.split('/')[2].split('-')[0]
    print('Dataset name:', dataset_name)

    pathlib.Path('./log').mkdir(parents=True, exist_ok=True)

    # global variables
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')

    print('reading dataset...')
    train_set, train_label = read_n_encode_dataset(train_path, vectorizer, logprobs, power)
    val_set, val_label = read_n_encode_dataset(train_path.replace('train', 'val'), vectorizer, logprobs, power)
    val_label_prec = val_label

    if dataset == "tmc" or dataset == "reuters":
        labels = set()
        for split in [train_label, val_label]:
            for i in split:
                for lab in i:
                    labels.add(lab)
        onehotencoder = MultiLabelBinarizer(classes=sorted(labels))
        train_label = onehotencoder.fit_transform(train_label)  # .tolist()
        val_label = onehotencoder.fit_transform(val_label)  # .tolist()

    # scaler = preprocessing.MinMaxScaler().fit(train_set.todense())
    # train_set = scaler.transform(train_set.todense())
    # val_set = scaler.transform(val_set.todense())

    # umap_model = joblib.load("./models/umap/enwiki-latest-pages-articles.train.hacked.umap")
    # train_set = umap_model.predict(train_set)
    # val_set = umap_model.predict(val_set)
    input_train, _ = vectorize_scale(lang=dataset, spf=train_path, logprob_power=7, top_words=500)
    input_val, _ = vectorize_scale(lang=dataset, spf=train_path.replace('train', 'val'), logprob_power=7, top_words=500)
    ridge_model = joblib.load(f'./models/umap/{dataset}/ridge_{dataset}')
    train_set = ridge_model.predict(input_train)
    val_set = ridge_model.predict(input_val)
    # std_rank = np.argsort(train_set.std(axis=0))[::-1]  # standard deviation from highest to lowest

    PN_SIZE = train_set.shape[1]

    max_thread = int(multiprocessing.cpu_count() * 0.7)

    print('testing...')
    tmp = fruitfly_pipeline(kc_size=10000, proj_size=5, wta=50, knn=100, num_trial=5, save=False)
