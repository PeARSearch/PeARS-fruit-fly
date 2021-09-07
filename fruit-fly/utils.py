import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack

from hash import wta_vectorized
from evolve_flies import genetic_alg

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events


def write_as_json(dic, f):
    output_file = open(f, 'w', encoding='utf-8')
    json.dump(dic, output_file)


def append_as_json(dic, f):
    output_file = open(f, 'a', encoding='utf-8')
    json.dump(dic, output_file)
    output_file.write("\n")


def hash_input_vectorized_(pn_mat, weight_mat, percent_hash):
    kc_mat = pn_mat.dot(weight_mat.T)
    m, n = kc_mat.shape
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(kc_mat[i: i+2000].toarray(), k=percent_hash)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hashed_kenyon = wta_csr[1:]
    return hashed_kenyon


def hash_dataset_(dataset_mat, weight_mat, percent_hash, top_words):
    m, n = dataset_mat.shape
    dataset_mat = csr_matrix(dataset_mat)
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(dataset_mat[i: i+2000].toarray(), k=top_words, percent=False)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hs = hash_input_vectorized_(wta_csr[1:], weight_mat, percent_hash)
    return hs


def get_stats(pop: list):
    """
    Get the average stats of a population
    Compare and get the best fly on multiple criteria
    """
    stats = {}
    count_nonzero, num_col, num_row, wta, kc_score, val_score, fitness = [], [], [], [], [], [], []
    for individual in pop:
        num_row.append(individual.projection.shape[0])
        num_col.append(individual.projection.shape[1])
        count_nonzero.append(individual.projection.count_nonzero() /
                             (individual.projection.shape[0] * individual.projection.shape[1]))
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
    with open('./models/evolution/best_scores.json') as f:
        best_scores = json.load(f)
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
    if max(val_score[:, 1]) > best_scores['val_wiki']:
        best_scores['val_wiki'] = max(val_score[:, 1])
        with open('./models/evolution/best_val_wiki', "wb") as f:
            pickle.dump(pop[np.argmax(val_score[:, 1])], f)
    # best val 20news
    if max(val_score[:, 2]) > best_scores['val_20news']:
        best_scores['val_wos'] = max(val_score[:, 2])
        with open('./models/evolution/best_val_20news', "wb") as f:
            pickle.dump(pop[np.argmax(val_score[:, 2])], f)
    # update best json
    with open('./models/evolution/best_scores.json', 'w') as f:
        json.dump(best_scores, f)

    return stats


def bayesian_optimization():
    def _evolve_bayes(pop_size: int, crossover_prob: float, select_percent: float,
                      mutate_prob_proj: float, mutate_scale_wta: float):
        pop_size = int(pop_size)
        if pop_size % 2:  # the pop size should be even
            pop_size -= 1  # otherwise after the 1st generation it will decrease by 1
        return genetic_alg(pop_size, crossover_prob, select_percent,
                           mutate_prob_proj, mutate_scale_wta)

    optimizer = BayesianOptimization(f=_evolve_bayes,
                                     pbounds={"pop_size": (400, 2000), 'select_percent': (0.1, 0.9),
                                              'crossover_prob': (0.3, 0.8),
                                              'mutate_prob_proj': (0.01, 0.1), 'mutate_scale_wta': (1, 5)},
                                     # random_state=123,
                                     verbose=2)
    logger = JSONLogger(path="./models/evolution/bayes_ga_log.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(n_iter=100)

    dic = {'pop_size': int(optimizer.max['params']['pop_size']),
           'select_percent': optimizer.max['params']['select_percent'],
           'cross_prob': optimizer.max['params']['crossover_prob'],
           'mutate_prob_proj': optimizer.max['params']['mutate_prob_proj'],
           'mutate_scale_wta': optimizer.max['params']['mutate_scale_wta']}
    print("Final result:", optimizer.max)
    append_as_json(dic, "./models/evolution/bayes_results.json")
