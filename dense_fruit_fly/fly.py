"""The main Fly class"""


import random
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack, lil_matrix, coo_matrix

from classify import train_model
from sklearn.metrics import pairwise_distances
from bo_utils import read_vocab, hash_dataset_, append_as_json, get_stats

class Fly:
    def __init__(self, pn_size=None, kc_size=None, wta=None, proj_size=None, top_words=None, init_method=None, eval_method=None, proj_store=None, hyperparameters=None):
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.wta = wta
        self.proj_size = proj_size
        self.top_words = top_words
        self.init_method = init_method
        self.eval_method = eval_method
        self.hyperparameters = hyperparameters
        if self.init_method == "random":
            weight_mat, self.shuffled_idx = self.create_projections(self.proj_size)
        else:
            weight_mat, self.shuffled_idx = self.projection_store(proj_store)

        self.projections = lil_matrix(weight_mat)
        self.val_score = 0
        self.kc_score = 1 / np.log10(int(self.kc_size * self.wta / 100))
        self.is_evaluated = False
        self.kc_use = None
        self.kc_sorted = None
        #print("INIT",self.kc_size,self.proj_size,self.wta,self.get_coverage())

    def create_projections(self,proj_size):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        idx = list(range(self.pn_size))
        random.shuffle(idx)
        used_idx = idx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(0,len(idx),proj_size):
                p = idx[i:i+proj_size]
                for j in p:
                    weight_mat[c][j] = 1
                c+=1
                if c >= self.kc_size:
                    break
            random.shuffle(idx) #reshuffle if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]


    def projection_store(self,proj_store):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        self.proj_store = proj_store.copy()
        proj_size = len(self.proj_store[0])
        random.shuffle(self.proj_store)
        sidx = [pn for p in self.proj_store for pn in p]
        idx = list(range(self.pn_size))
        not_in_store_idx = list(set(idx) - set(sidx))
        print(len(not_in_store_idx),'IDs not in store')
        used_idx = sidx.copy()
        c = 0
        
        while c < self.kc_size:
            for i in range(len(self.proj_store)):
                p = self.proj_store[i]
                for j in p:
                    weight_mat[c][j] = 1
                c+=1
                if c >= self.kc_size:
                    break
            random.shuffle(idx) #add random if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]

    def get_coverage(self):
        ps = self.projections.toarray()
        vocab_cov = (self.pn_size - np.where(~ps.any(axis=0))[0].shape[0]) / self.pn_size
        kc_cov = (self.kc_size - np.where(~ps.any(axis=1))[0].shape[0]) / self.kc_size
        return vocab_cov, kc_cov

    def get_fitness(self):
        if not self.is_evaluated:
            return 0
        if DATASET == "all":
            return np.mean(self.val_scores) 
        else:
            return np.sum(self.val_scores)

    def evaluate(self,train_set,val_set,train_label,val_label):
        hash_val, kc_use_val, kc_sorted_val = hash_dataset_(dataset_mat=val_set, weight_mat=self.projections,
                                 percent_hash=self.wta, top_words=self.top_words)
        if self.eval_method == "classification":
            #We only need the train set for classification, not similarity
            hash_train, kc_use_train, kc_sorted_train = hash_dataset_(dataset_mat=train_set, weight_mat=self.projections,
                                   percent_hash=self.wta, top_words=self.top_words)
            self.val_score, _ = train_model(m_train=hash_train, classes_train=train_label,
                                       m_val=hash_val, classes_val=val_label,
                                       C=self.hyperparameters['C'], num_iter=self.hyperparameters['num_iter'])
        if self.eval_method == "similarity":
            self.val_score, _ = self.prec_at_k(m_val=hash_val, classes_val=val_label, k=self.hyperparameters['num_nns'])
        self.is_evaluated = True
        print("\nCOVERAGE:",self.get_coverage())
        print("SCORE:",self.val_score)
        self.kc_use = kc_use_val / np.sum(kc_use_val)
        self.kc_sorted = list(kc_sorted_val)
        #print("PROJECTIONS:",self.print_projections())
        #print("KC USE:",np.sort(self.kc_use)[::-1][:20])
        return self.val_score,self.kc_sorted[:10]

    def compute_nearest_neighbours(self,cosines,labels,i,num_nns):
        i_cos = np.array(cosines[i])
        i_label = labels[i]
        ranking = np.argsort(-i_cos)
        neighbours = [labels[n] for n in ranking][1:num_nns+1] #don't count first neighbour which is itself
        score = sum([1 if n == i_label else 0 for n in neighbours]) / num_nns
        #print(i,i_label,neighbours,score)
        return score,neighbours

    def prec_at_k(self,m_val=None,classes_val=None,k=None):
        cosines = 1-pairwise_distances(m_val, metric="cosine") / 2 
        #Dividing by 2 to avoid negative values. See https://stackoverflow.com/questions/37454785/how-to-handle-negative-values-of-cosine-similarities
        scores = []
        for i in range(cosines.shape[0]):
            score, neighbours = self.compute_nearest_neighbours(cosines,classes_val,i,k)
            scores.append(score)
        return np.mean(scores), scores

    def print_projections(self):
        words = ''
        for row in self.projections[:10]:
            cs = np.where(row.toarray()[0] == 1)[0]
            for i in cs:
                words+=reverse_vocab[i]+' '
            words+='|'
        return words


