"""Genetic Algorithm for fruit-fly projection
Usage:
  evolve_flies.py
  evolve_flies.py (-h | --help)
  evolve_flies.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
"""

import numpy as np
import joblib
import multiprocessing
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, lil_matrix
from docopt import docopt
import time
from datetime import datetime
from copy import deepcopy

from hyperparam_search import read_n_encode_dataset
from classify import train_model
from hash import read_vocab
from utils import hash_dataset_, append_as_json, get_stats


class Fly:
    def __init__(self):
        self.kc_size = np.random.randint(low=MIN_KC, high=MAX_KC)
        self.wta = np.random.uniform(low=MIN_WTA, high=MAX_WTA)
        weight_mat = np.zeros((self.kc_size, PN_SIZE))
        for i in range(self.kc_size):
            num_proj = np.random.randint(low=MIN_PROJ, high=MAX_PROJ)
            for j in np.random.randint(PN_SIZE, size=num_proj):
                weight_mat[i, j] = 1
        self.projection = lil_matrix(weight_mat)
        self.val_scores = [0, 0, 0]
        self.kc_score = 1 / np.log10(int(self.kc_size * self.wta / 100))
        self.is_evaluated = False

    def get_fitness(self):
        if not self.is_evaluated:
            return 0
        return np.mean(self.val_scores) + self.kc_score

    def evaluate(self):
        val_score_list = []
        for i in range(len(train_set_list)):
            hash_train = hash_dataset_(dataset_mat=train_set_list[i], weight_mat=self.projection,
                                       percent_hash=self.wta, top_words=top_word)
            hash_val = hash_dataset_(dataset_mat=val_set_list[i], weight_mat=self.projection,
                                     percent_hash=self.wta, top_words=top_word)
            val_score, _ = train_model(m_train=hash_train, classes_train=train_label_list[i],
                                       m_val=hash_val, classes_val=val_label_list[i],
                                       C=C, num_iter=num_iter)
            val_score_list.append(val_score)
        self.val_scores = val_score_list
        self.is_evaluated = True


# components of genetic algorithm
def init_pop(pop_size: int):
    """
    Generate a random population
    """
    population = joblib.Parallel(n_jobs=max_thread, prefer="threads")(
        joblib.delayed(Fly)() for _ in range(pop_size))
    return population


def eval_pop(population: list):
    """
    Calculate the fitness of every chromosome
    :param population: the list that contains every chromosomes
    """
    def _eval_individual(fly: Fly):
        if not fly.is_evaluated:
            fly.evaluate()
        else:
            pass
    joblib.Parallel(n_jobs=max_thread, prefer="threads")(
        joblib.delayed(_eval_individual)(fly) for fly in population)


def select_elite_tournament(fitness_list: list, select_percent: float):
    """
    Tournament + elitism selection.
    Return the index of genes that has been selected.
    """
    selected = set()
    num_select = round(select_percent * len(fitness_list)) - 2
    if num_select < 2:
        num_select = 2  # mother, father

    fitness2idx = {k: v for v, k in enumerate(fitness_list)}
    elite_chroms = 2
    if num_select < elite_chroms:
        elite_chroms = num_select
    elites = sorted(fitness_list, reverse=True)[:elite_chroms]
    for eli in elites:
        selected.add(fitness2idx[eli])

    while len(selected) < num_select:
        first_cand, second_cand = np.random.choice(range(len(fitness_list)), 2)
        if fitness_list[first_cand] > fitness_list[second_cand]:
            if first_cand not in selected:
                selected.add(first_cand)
        else:
            if second_cand not in selected:
                selected.add(second_cand)
    return list(selected)


def crossover(parent1: Fly, parent2: Fly):
    """
    Crossover two flies
    Input is a Fly(); Return two new offsprings
    Do crossover on two elements: the projection and the wta
    Projection: randomly truncate rows in the projection matrix that has higher number of row
    until the two matrices have the same number of row, then split and swap vertically
    Wta: take randomly two values between the range of two wta values
    """
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    child1.is_evaluated = False
    child2.is_evaluated = False

    # first, crossover projection matrices
    # truncate
    if parent1.kc_size > parent2.kc_size:
        random_indices = np.random.choice(parent1.kc_size, size=int(parent2.kc_size), replace=False)
        child1.projection = parent1.projection[random_indices, :]
        child1.kc_size = child1.projection.shape[0]
    else:
        random_indices = np.random.choice(parent2.kc_size, size=int(parent1.kc_size), replace=False)
        child2.projection = parent2.projection[random_indices, :]
        child2.kc_size = child2.projection.shape[0]
    # swap
    col_idx = int(child1.projection.shape[1]/2)
    new_proj_1 = hstack([child1.projection[:, :col_idx], child2.projection[:, col_idx:]])
    new_proj_2 = hstack([child1.projection[:, col_idx:], child2.projection[:, :col_idx]])
    child1.projection, child2.projection = lil_matrix(new_proj_1), lil_matrix(new_proj_2)

    # then, crossover wta
    wta_low, wta_high = sorted([child1.wta, child2.wta])
    child1.wta = np.random.uniform(low=wta_low, high=wta_high)
    child2.wta = np.random.uniform(low=wta_low, high=wta_high)

    # update kc_score
    child1.kc_score = 1 / np.log10(int(child1.kc_size * child1.wta / 100))
    child2.kc_score = 1 / np.log10(int(child2.kc_size * child2.wta / 100))

    return child1, child2


def mutate(individual: Fly, mutate_prob_proj: float, mutate_scale_wta: float):
    """
    Mutate the projection matrix and the wta
    Projection matrix: randomly choose some elements and flip them (0 -> 1, 1 -> 0)
    The number of chosen element is controlled by mutate_prob_proj
    Wta: construct a Gaussian distribution with mean is the old wta, standard deviation is
    the mutate_scale_wta, then draw the new wta from the distribution
    Modify the wta if it is out of the min or max value of WTA
    """
    mutated_indiv = deepcopy(individual)
    mutated_indiv.is_evaluated = False

    # first, mutate the projection
    row_mutate = np.random.choice(individual.kc_size, int(individual.kc_size * mutate_prob_proj))
    for i in row_mutate:
        new_row = np.zeros(PN_SIZE)
        num_proj = np.random.randint(low=MIN_PROJ, high=MAX_PROJ)
        new_row[np.random.randint(0, PN_SIZE, num_proj)] = 1
        mutated_indiv.projection[i] = new_row
        mutated_indiv.projection = lil_matrix(mutated_indiv.projection)

    # then, mutate the wta
    new_wta = np.random.normal(loc=individual.wta, scale=mutate_scale_wta)
    if new_wta < MIN_WTA:
        new_wta = MIN_WTA
    if new_wta > MAX_WTA:
        new_wta = MAX_WTA
    mutated_indiv.wta = new_wta

    return mutated_indiv


def evolve(population: list, select_percent: float,
           crossover_prob: float, mutate_prob_proj: float, mutate_scale_wta: float):
    """
    Create next generation by selecting, crossing-over and mutating.
    """
    def _select_crossover_mutate():
        # select
        selected = select_elite_tournament(fitness_list, select_percent)
        mother_choice = np.random.choice(selected)
        selected.remove(mother_choice)
        father_choice = np.random.choice(selected)
        mother = population[mother_choice]
        father = population[father_choice]

        # crossover
        if np.random.random() < crossover_prob:
            child1, child2 = crossover(mother, father)
        else:
            child1 = mother
            child2 = father

        # mutation
        child1 = mutate(child1, mutate_prob_proj, mutate_scale_wta)
        child2 = mutate(child2, mutate_prob_proj, mutate_scale_wta)

        return child1, child2

    fitness_list = [individual.get_fitness() for individual in population]

    children_list = joblib.Parallel(n_jobs=max_thread, prefer="threads")(
        joblib.delayed(_select_crossover_mutate)() for _ in range(len(population) // 2))
    new_population = children_list[0] + children_list[1]

    return new_population


def genetic_alg(pop_size: int, crossover_prob: float, select_percent: float,
                mutate_prob_proj: float, mutate_scale_wta: float):
    """
    Genetic Algorithms, main function.
    """
    # create log
    log_file = './models/evolution/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.json'
    append_as_json({'pop_size': pop_size, 'crossover_prob': crossover_prob,
                    'select_percent': select_percent, 'mutate_prob_proj': mutate_prob_proj,
                    'mutate_scale_wta': mutate_scale_wta}, log_file)
    append_as_json({'max_generation': MAX_GENERATION, 'min_wta': MIN_WTA, 'max_wta': MAX_WTA,
                    'min_kc': MIN_KC, 'max_kc': MAX_KC, 'min_proj': MIN_PROJ, 'max_proj': MAX_PROJ,
                    'top_word': top_word, 'C': C, 'num_iter': num_iter}, log_file)

    # generate the first random population
    print(f'generate the first generation of {pop_size} individuals')
    start_time = time.time()
    population = init_pop(pop_size)
    print('time to generate the first generation: {}'.format(time.time() - start_time))

    print('evolving')
    # total_improvement = []
    # last_fitness = 0
    for g in range(MAX_GENERATION):
        start_time = time.time()
        # new generation, including selection, crossover, mutation
        if g > 0:  # do not process this step in the 1st generation, since every fitness = 0
            population = evolve(population, select_percent,
                                crossover_prob, mutate_prob_proj, mutate_scale_wta)

        # evaluate the population
        eval_pop(population)
        # fitness_list = [individual.get_fitness() for individual in population]

        # print progress
        # avg_fitness = np.mean(fitness_list)
        # if g == 0:
        #     last_fitness = avg_fitness
        # improvement_fitness = avg_fitness - last_fitness
        # last_fitness = avg_fitness
        # total_improvement.append(improvement_fitness)
        # avg_fitness_list.append(avg_fitness)
        stats = get_stats(population)
        stats['gen'] = g
        stats['time'] = time.time() - start_time

        append_as_json(stats, log_file)
        print(stats)

    # print("sum improvement:", sum(total_improvement))
    # return sum(total_improvement)


if __name__ == '__main__':
    args = docopt(__doc__, version='Genetic Algorithm of the fruit-fly projection, ver 0.1')

    MAX_GENERATION = 20
    MIN_WTA, MAX_WTA = 1, 30
    MIN_KC, MAX_KC = 1000, 10000
    MIN_PROJ, MAX_PROJ = 5, 20
    top_word = 700
    C = 100
    num_iter = 2000  # wikipedia and wos only need 50 steps
    max_thread = int(multiprocessing.cpu_count() * 0.7)
    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')
    vocab, reverse_vocab, logprobs = read_vocab()
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
    PN_SIZE = len(vocab)

    print('reading datasets')
    num_dataset = 3
    train_set_list, train_label_list = [None] * 3, [None] * 3
    val_set_list, val_label_list = [None] * 3, [None] * 3
    train_set_list[0], train_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-train.sp', vectorizer, logprobs)
    val_set_list[0], val_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-val.sp', vectorizer, logprobs)
    train_set_list[1], train_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-train.sp', vectorizer, logprobs)
    val_set_list[1], val_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-val.sp', vectorizer, logprobs)
    train_set_list[2], train_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-train.sp', vectorizer, logprobs)
    val_set_list[2], val_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-val.sp', vectorizer, logprobs)

    genetic_alg(pop_size=1000, crossover_prob=0.5, select_percent=0.2, mutate_prob_proj=0.02, mutate_scale_wta=2)
