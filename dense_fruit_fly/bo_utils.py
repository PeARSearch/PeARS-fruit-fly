import re
import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import pairwise_distances
from os.path import exists
from hash import read_projections, projection_vectorized, wta_vectorized


def read_vocab(vocab_file):
    c = 0
    vocab = {}
    reverse_vocab = {}
    logprobs = []
    with open(vocab_file) as f:
        for l in f:
            l = l.rstrip('\n')
            wp = l.split('\t')[0]
            logprob = -(float(l.split('\t')[1]))
            #logprob = log(lp + 1.1)
            if wp in vocab or wp == '':
                continue
            vocab[wp] = c
            reverse_vocab[c] = wp
            logprobs.append(logprob**3)
            c+=1
    return vocab, reverse_vocab, logprobs


def read_n_encode_dataset(path, vectorizer, logprobs):
    # read
    doc_list, label_list = [], []
    doc = ""
    with open(path) as f:
        for l in f:
            l = l.rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*class=([^ ]*)>", l)
                label = m.group(1)
                label_list.append(label)
            elif l[:5] == "</doc":
                doc_list.append(doc)
                doc = ""
            else:
                doc += l + ' '

    # encode
    X = vectorizer.fit_transform(doc_list)
    X = csr_matrix(X)
    X = X.multiply(logprobs)

    return X, label_list


def write_as_json(dic, f):
    output_file = open(f, 'w', encoding='utf-8')
    json.dump(dic, output_file)


def append_as_json(dic, f):
    output_file = open(f, 'a', encoding='utf-8')
    json.dump(dic, output_file)
    output_file.write("\n")


def hash_input_vectorized_(pn_mat, weight_mat, percent_hash):
    kc_mat = pn_mat.dot(weight_mat.T)
    #print(pn_mat.shape,weight_mat.shape,kc_mat.shape)
    sums = kc_mat.toarray().sum(axis=0,keepdims=1);
    sums[sums==0] = 1
    kc_mat_norm = (kc_mat.toarray()/sums).T # Make columns (KC vectors) sum to 1 and transpose for next step
    cosines = pairwise_distances(kc_mat_norm,metric='cosine') # Distances between KC vectors
    kc_use = np.squeeze(np.asarray(np.mean(cosines,axis=0))) # Mean distance across rows (KCs)
    kc_sorted_ids = np.argsort(kc_use)[:-kc_use.shape[0]-1:-1] #Give sorted list from most to least used KCs
    m, n = kc_mat.shape
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(kc_mat[i: i+2000].toarray(), k=percent_hash)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hashed_kenyon = wta_csr[1:]
    return hashed_kenyon, kc_use, kc_sorted_ids


def hash_dataset_(dataset_mat, weight_mat, percent_hash, top_words):
    m, n = dataset_mat.shape
    dataset_mat = csr_matrix(dataset_mat)
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(dataset_mat[i: i+2000].toarray(), k=top_words, percent=False)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hs, kc_use, kc_sorted_ids = hash_input_vectorized_(wta_csr[1:], weight_mat, percent_hash)
    hs = (hs > 0).astype(np.int_)
    return hs, kc_use, kc_sorted_ids


def get_stats(pop: list):
    """
    Get the average stats of a population
    Compare and get the best fly on multiple criteria
    """
    stats = {}
    count_nonzero, num_col, num_row, wta, kc_score, val_score, fitness = [], [], [], [], [], [], []
    for individual in pop:
        num_row.append(individual.projections.shape[0])
        num_col.append(individual.projections.shape[1])
        count_nonzero.append(individual.projections.count_nonzero() /
                             (individual.projections.shape[0] * individual.projections.shape[1]))
        wta.append(individual.wta)
        kc_score.append(individual.kc_score)
        val_score.append(individual.val_scores)
        fitness.append(individual.get_fitness())
    val_score = np.array(val_score)

    # population stats
    stats['fitness'] = np.mean(fitness)
    stats['non_zero'] = np.mean(count_nonzero)
    stats['val_score'] = np.mean(val_score, axis=0).tolist()
    stats['kc_size'] = np.mean(num_row)
    stats['wta'] = np.mean(wta)
    stats['kc_score'] = np.mean(kc_score)

    # get best individual
    best_score_file = './models/evolution/best_scores.json'
    if exists(best_score_file):
        with open('./models/evolution/best_scores.json') as f:
            best_scores = json.load(f)
    else:
        best_scores = {'fitness':0.0, 'avg_val_score':0.0, 'kc_score':0.0, 'val_wos':0.0, 'val_wikipedia':0.0, 'val_20news':0.0}
    # best fitness
    if max(fitness) > best_scores['fitness']:
        best_scores['fitness'] = max(fitness)
        with open('./models/evolution/best_fitness', "wb") as f:
            pickle.dump(pop[np.argmax(fitness)], f)
    # best avg_val_score
    avg_val_score = np.mean(val_score, axis=1)
    if np.max(avg_val_score) > best_scores['avg_val_score']:
        best_scores['avg_val_score'] = np.max(avg_val_score)
        with open('./models/evolution/best_val_score', "wb") as f:
            pickle.dump(pop[np.argmax(avg_val_score)], f)
    # best kc_score
    if max(kc_score) > best_scores['kc_score']:
        best_scores['kc_score'] = max(kc_score)
        with open('./models/evolution/best_kc_score', "wb") as f:
            pickle.dump(pop[np.argmax(kc_score)], f)
    # best val wos
    if max(val_score[:, 0]) > best_scores['val_wos']:
        best_scores['val_wos'] = max(val_score[:, 0])
        with open('./models/evolution/best_val_wos', "wb") as f:
            pickle.dump(pop[np.argmax(val_score[:, 0])], f)
    # best val wiki
    if max(val_score[:, 1]) > best_scores['val_wikipedia']:
        best_scores['val_wikipedia'] = max(val_score[:, 1])
        with open('./models/evolution/best_val_wikipedia', "wb") as f:
            pickle.dump(pop[np.argmax(val_score[:, 1])], f)
    # best val 20news
    if max(val_score[:, 2]) > best_scores['val_20news']:
        best_scores['val_20news'] = max(val_score[:, 2])
        with open('./models/evolution/best_val_20news', "wb") as f:
            pickle.dump(pop[np.argmax(val_score[:, 2])], f)
    # update best json
    with open('./models/evolution/best_scores.json', 'w') as f:
        json.dump(best_scores, f)

    return stats


