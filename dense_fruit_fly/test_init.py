"""Test init strategy
Usage:
  fly_search.py --dataset=<str> --logprob=<n>
  fly_search.py (-h | --help)
  fly_search.py --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --dataset=<str>              Name of dataset, either wiki, 20news, or wos.
  --logprob=<n>                Power of word log probabilities when vectorizing documents.
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

from classify import train_model
from sklearn.metrics import pairwise_distances
from utils import read_vocab, hash_dataset_, read_n_encode_dataset
import warnings
warnings.filterwarnings("ignore")


class FlyTestInit:
    def __init__(self, pn_size=None, kc_size=None, wta=None, proj_size=None, init_method=None, eval_method=None,
                 proj_store=None, hyperparameters=None):
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.wta = wta
        self.proj_size = proj_size
        self.init_method = init_method
        self.eval_method = eval_method
        self.hyperparameters = hyperparameters
        if self.init_method == "original":
            weight_mat, self.shuffled_idx = self.create_projections(self.proj_size)
        elif self.init_method == "minus_1":
            weight_mat, self.shuffled_idx = self.create_projections_0(self.proj_size)
        elif self.init_method == "ach":
            weight_mat, self.shuffled_idx = self.create_projections_1()
        elif self.init_method == "ach_3":
            weight_mat, self.shuffled_idx = self.create_projections_2(scale=3)
        elif self.init_method == "ach_kc":
            weight_mat, self.shuffled_idx = self.create_projections_2(scale=self.kc_size)
        elif self.init_method == "ach_sqrt_pn":
            weight_mat, self.shuffled_idx = self.create_projections_2(scale=np.sqrt(self.pn_size))
        else:
            weight_mat, self.shuffled_idx = self.projection_store(proj_store)

        self.projections = lil_matrix(weight_mat)
        self.val_score_c = 0
        self.val_score_s = 0
        self.is_evaluated = False
        self.kc_use_sorted = None
        self.kc_in_hash_sorted = None
        # print("INIT",self.kc_size,self.proj_size,self.wta,self.get_coverage())


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

    def create_projections_0(self, proj_size):  # introduce -1
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        idx = list(range(self.pn_size))
        random.shuffle(idx)
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
            random.shuffle(idx)  # reshuffle if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]

    def create_projections_1(self):  # Achlioptas without scale
        weight_mat = np.random.choice([-1, 0, 1], size=(self.kc_size, self.pn_size), p=[1/6, 2/3, 1/6])
        return weight_mat, None

    def create_projections_2(self, scale):  # Achlioptas with scale
        proportion = [1/(2*scale), 1 - 1/scale, 1/(2*scale)]
        weight_mat = np.random.choice([-1, 0, 1], size=(self.kc_size, self.pn_size), p=proportion)
        weight_mat = weight_mat * np.sqrt(scale/self.kc_size)
        return weight_mat, None


    def projection_store(self, proj_store):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        self.proj_store = proj_store.copy()
        proj_size = len(self.proj_store[0])
        random.shuffle(self.proj_store)
        sidx = [pn for p in self.proj_store for pn in p]
        idx = list(range(self.pn_size))
        not_in_store_idx = list(set(idx) - set(sidx))
        # print(len(not_in_store_idx),'IDs not in store')
        used_idx = sidx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(len(self.proj_store)):
                p = self.proj_store[i]
                for j in p:
                    weight_mat[c][j] = 1
                c += 1
                if c >= self.kc_size:
                    break
            random.shuffle(idx)  # add random if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]

    def get_coverage(self):
        ps = self.projections.toarray()
        vocab_cov = (self.pn_size - np.where(~ps.any(axis=0))[0].shape[0]) / self.pn_size
        kc_cov = (self.kc_size - np.where(~ps.any(axis=1))[0].shape[0]) / self.kc_size
        return vocab_cov, kc_cov

    # def get_fitness(self):
    #     if not self.is_evaluated:
    #         return 0
    #     if DATASET == "all":
    #         return np.mean(self.val_scores)
    #     else:
    #         return np.sum(self.val_scores)

    def evaluate(self, train_set, val_set, train_label, val_label):
        hash_val, kc_use_val, kc_sorted_val = hash_dataset_(dataset_mat=val_set, weight_mat=self.projections,
                                                            percent_hash=self.wta)
        # if self.eval_method == "classification":
        # We only need the train set for classification, not similarity
        hash_train, kc_use_train, kc_sorted_train = hash_dataset_(dataset_mat=train_set,
                                                                  weight_mat=self.projections,
                                                                  percent_hash=self.wta)
        self.val_score_c, _ = train_model(m_train=hash_train, classes_train=train_label,
                                        m_val=hash_val, classes_val=val_label,
                                        C=self.hyperparameters['C'], num_iter=self.hyperparameters['num_iter'])
        #if self.eval_method == "similarity":
        self.val_score_s, kc_in_hash_sorted = self.prec_at_k(m_val=hash_val, classes_val=val_label,
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

    def compute_nearest_neighbours(self, hammings, labels, i, num_nns):
        i_sim = np.array(hammings[i])
        i_label = labels[i]
        ranking = np.argsort(-i_sim)
        neighbours = [labels[n] for n in ranking][1:num_nns + 1]  # don't count first neighbour which is itself
        n_sum = 0
        print("neighbours: ", neighbours)
        print("i_label", i_label)
        for n in neighbours:
            for lab in n:
                if lab in i_label:
                    print("lab", lab)
                    XXX
                    """
                    now it's not working for multilabel documents because of 
                    the one hot encoded labels. Check that!!!!
                    """
                    n_sum += 1
                    break
        score = n_sum / num_nns
        return score, neighbours

    def prec_at_k(self, m_val=None, classes_val=None, k=None):
        hammings = 1 - pairwise_distances(m_val.todense(), metric="hamming")
        kc_hash_use = np.zeros(m_val.shape[1])
        scores = []
        for i in range(hammings.shape[0]):
            score, neighbours = self.compute_nearest_neighbours(hammings, classes_val, i, k)
            for idx in m_val[i].indices:
                kc_hash_use[idx] += 1
            scores.append(score)
        kc_hash_use = kc_hash_use / sum(kc_hash_use)
        kc_sorted_hash_use = np.argsort(kc_hash_use)[
                             :-kc_hash_use.shape[0] - 1:-1]  # Give sorted list from most to least used KCs
        return np.mean(scores), kc_sorted_hash_use

    def print_projections(self):
        words = ''
        for row in self.projections[:10]:
            cs = np.where(row.toarray()[0] == 1)[0]
            for i in cs:
                words += reverse_vocab[i] + ' '
            words += '|'
        return words


def fruitfly_pipeline(kc_size, proj_size, wta, knn, num_trial, save):
    # Below parameters are needed to init the fruit fly, even if not used here
    init_method_list = ['original', 'minus_1', 'ach', 'ach_3', 'ach_kc', 'ach_sqrt_pn']
    eval_method = ''
    proj_store = None
    hyperparameters = {'C': 100, 'num_iter': 200, 'num_nns': knn}

    fly_list = [FlyTestInit(PN_SIZE, kc_size, wta, proj_size, init_method, eval_method, proj_store, hyperparameters)
                for init_method in init_method_list for _ in range(num_trial)]
    print('evaluating...')

    with joblib.Parallel(n_jobs=max_thread, prefer="threads") as parallel:
        delayed_funcs = [joblib.delayed(lambda x: x.evaluate(train_set, val_set, train_label, val_label))(fly) for fly in
                         fly_list]
        scores = parallel(delayed_funcs)

    score_c_list = np.array([p[0] for p in scores]).reshape(len(init_method_list), num_trial)
    score_s_list = np.array([p[1] for p in scores]).reshape(len(init_method_list), num_trial)

    # average the validation acc
    avg_score_c = np.mean(score_c_list, axis=1)
    std_score_c = np.std(score_c_list, axis=1)
    print('average score c:')
    for i in avg_score_c: print(round(i, 3))

    # best_score_c = max(score_c_list)
    # best_fly_c = fly_list[score_c_list.index(best_score_c)]
    # if save:
    #     filename = './models/flies/' + dataset + '.fly_c.m'
    #     joblib.dump(best_fly_c, filename)

    # average the validation acc
    avg_score_s = np.mean(score_s_list, axis=1)
    std_score_s = np.std(score_s_list, axis=1)
    print('average score s:')
    for i in avg_score_s: print(round(i, 3))

    # best_score_s = max(score_s_list)
    # best_fly_s = fly_list[score_s_list.index(best_score_s)]
    # if save:
    #     filename = './models/flies/' + dataset + '.fly_s.m'
    #     joblib.dump(best_fly_s, filename)

    return avg_score_c, avg_score_s


if __name__ == '__main__':
    args = docopt(__doc__, version='Test init strategy, ver 0.1')
    dataset = args["--dataset"]
    power = int(args["--logprob"])

    if dataset == "wiki":
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

    if dataset == "tmc" or dataset=="reuters":
        labels = set()
        for split in [train_label, val_label]:
            for i in split:
                for lab in i:
                    labels.add(lab)
        onehotencoder = MultiLabelBinarizer(classes=sorted(labels))
        train_label = onehotencoder.fit_transform(train_label) #.tolist()
        val_label = onehotencoder.fit_transform(val_label) #.tolist()

    scaler = preprocessing.MinMaxScaler().fit(train_set.todense())
    train_set = scaler.transform(train_set.todense())
    val_set = scaler.transform(val_set.todense())
    # train_set = umap_model.transform(train_set)
    # val_set = umap_model.transform(val_set)
    PN_SIZE = train_set.shape[1]

    max_thread = int(multiprocessing.cpu_count() * 0.7)

    print('testing...')
    for kc_size in [64, 128, 256]:
        print('kc size =', kc_size)
        tmp = fruitfly_pipeline(kc_size=kc_size, proj_size=5, wta=50, knn=100, num_trial=10, save=False)
        # print(tmp)