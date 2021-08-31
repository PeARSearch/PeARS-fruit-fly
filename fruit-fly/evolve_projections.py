
"""Genetic Algorithm for fruit-fly projection
Usage:
  evolve_projection.py --train_path=<filename>
  evolve_projection.py (-h | --help)
  evolve_projection.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<filename>         Name of file to train (processed by sentencepeice)
"""


import numpy as np
import joblib
import pathlib
import pickle
import multiprocessing
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, vstack, hstack, lil_matrix
from docopt import docopt
import time
import utils
import scipy.stats as ss

from hyperparam_search import read_n_encode_dataset
from classify import train_model
from hash import wta_vectorized, read_vocab


# helpers
def generate_proj():
    """
    Generate a random projection
    """
    kc_size = np.random.randint(low=MIN_KC, high=MAX_KC)
    weight_mat = np.zeros((kc_size, PN_SIZE))
    for i in range(kc_size):
        # size_ = np.random.randint(low=2, high=10, size=1)[0]
        for j in np.random.randint(PN_SIZE, size=NUM_PROJ):
            weight_mat[i, j] = 1
    weight_mat = lil_matrix(weight_mat)
    return weight_mat


def get_stats(pop):
    """
    Count the percentage of non-zero elements
    """
    stats = {}
    count_nonzero, num_col, num_row, kc_score = [], [], [], []
    for individual in pop:
        num_row.append(individual.shape[0])
        num_col.append(individual.shape[1])
        count_nonzero.append(individual.count_nonzero() / (individual.shape[0] * individual.shape[1]))
        kc_score.append(1 / np.log10(individual.shape[0]))
    # count_nonzero = [individual.count_nonzero() for individual in pop]
    # m, n = pop[0].shape[0], pop[0].shape[1]
    stats['nonzero'] = np.mean(count_nonzero)
    stats['kc_size'] = np.mean(num_row)
    stats['kc_score'] = np.mean(kc_score)
    return stats


def get_best(population, fitness_list):
    """
    Return the best gene and its fitness.
    :param population: the list that contains every chromosomes
    """
    best_fit = max(fitness_list)
    best_individual = population[fitness_list.index(best_fit)]
    return best_individual, best_fit


# components of genetic algorithm
def init_pop():
    """
    Generate a random population
    """
    population = joblib.Parallel(n_jobs=max_thread, prefer="threads")(
        joblib.delayed(generate_proj)() for _ in range(POP_SIZE))
    fitness_list = [-1] * POP_SIZE
    return population, fitness_list


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


def fitness(weight_mat, prev_fitness):
    """
    Measure the fitness of one projection
    """
    kc_size = weight_mat.shape[0]
    kc_score = 1 / np.log10(kc_size)
    if prev_fitness != -1:
        sum_val_score = prev_fitness - kc_score
        return prev_fitness, sum_val_score, kc_score, kc_size

    val_score_list = []
    for i in range(len(train_set_list)):
        hash_train = hash_dataset_(dataset_mat=train_set_list[i], weight_mat=weight_mat,
                                   percent_hash=percent_hash, top_words=top_word)
        hash_val = hash_dataset_(dataset_mat=val_set_list[i], weight_mat=weight_mat,
                                 percent_hash=percent_hash, top_words=top_word)
        # print('training and evaluating')
        val_score, _ = train_model(m_train=hash_train, classes_train=train_label_list[i],
                                   m_val=hash_val, classes_val=val_label_list[i],
                                   C=C, num_iter=num_iter)
        val_score_list.append(val_score)

    # print(kc_size)
    fitness_score = np.sum(val_score_list) + kc_score
    return fitness_score, np.sum(val_score_list), kc_score, kc_size
    # return weight_mat.count_nonzero()


def validate(population, fitness_list):
    """
    Calculate the fitness of every chromosome
    :param population: the list that contains every chromosomes
    :param fitness_list: the list that contains fitness score of the corresponding chromosome
    """
    new_fitness_list = joblib.Parallel(n_jobs=max_thread, prefer="threads")(
        joblib.delayed(fitness)(population[i], fitness_list[i]) for i in range(POP_SIZE))
    return new_fitness_list


def select(fitness_list):
    """
    Tournament selection.
    Return the index of genes that has been selected.
    """
    selected = []
    num_select = round(SELECT_PERCENT * POP_SIZE)
    if num_select < 2:
        num_select = 2  # mother, father
    for i in range(num_select):
        first_cand, second_cand = np.random.choice(range(POP_SIZE), 2)
        if fitness_list[first_cand] > fitness_list[second_cand]:
            selected.append(first_cand)
        else:
            selected.append(second_cand)
    return selected


def elitist_selection(fitness_list):
    """
    Tournament selection + elitist selection
    Return the index of genes that has been selected.
    """
    selected = set()
    num_select = round(SELECT_PERCENT * POP_SIZE)
    if num_select < 2:
        num_select = 2  # mother, father

    fitness2idx={k:v for v, k in enumerate(fitness_list)}
    global ELITE_CHROMS
    if num_select<ELITE_CHROMS:
        ELITE_CHROMS=num_select
    elites=sorted(fitness_list, reverse=True)[:ELITE_CHROMS]
    for eli in elites:
        selected.add(fitness2idx[eli])

    while len(selected)<num_select:
        first_cand, second_cand = np.random.choice(range(len(fitness_list)), 2)
        if fitness_list[first_cand] > fitness_list[second_cand]:
            if first_cand not in selected:
                selected.add(first_cand)
        else:
            if second_cand not in selected:
                selected.add(second_cand)
    return list(selected)


def roulette_wheel_selection_updated(fitness_list):
    '''
    New roulette wheel selection function.
    Returns the index of the chromosomes that have been selected.
    '''
    num_select = round(SELECT_PERCENT * POP_SIZE)
    s=sum(fitness_list)
    weights = []
    for i in range(len(fitness_list)) :
        weights.append(fitness_list[i]/s)

    random_num=np.random.choice(fitness_list,size=num_select,p=weights)
    return len([fitness_list.index(i) for i in random_num])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def rank_selection(fitness_list):
    '''
    Similar method to the roulette wheel selection, the difference is that the probability
    is calculated according to the rank of the fitness values. 
    Returns the index of chromosomes that have been selected.
    '''
    num_select = round(SELECT_PERCENT * POP_SIZE)

    ranks=ss.rankdata(fitness_list)
    for i, rank in enumerate(ranks):
      ranks[i]= int(rank) / len(fitness_list)+1

    ranks=softmax(ranks)
    print(sum(ranks))

    random_num=np.random.choice(fitness_list,size=num_select,p=ranks)
    return [fitness_list.index(i) for i in random_num]


def rank_selection_2(fitness_list):
    '''
    Similar method to the roulette wheel selection, the difference is that the probability
    is calculated according to the rank of the fitness values.
    Returns the index of chromosomes that have been selected.
    '''
    num_select = round(SELECT_PERCENT * POP_SIZE)
    len_rank = len(fitness_list)
    rank_sum = len_rank * (len_rank + 1) / 2

    ranks=ss.rankdata(fitness_list)
    for i, rank in enumerate(ranks):
        ranks[i]= int(rank) / rank_sum

    random_num=np.random.choice(fitness_list,size=num_select,p=ranks)
    return [fitness_list.index(i) for i in random_num]


def stochastic_universal_sampling_selection(fitness_list):
    '''
    "SUS uses a single random value to sample all of the solutions by choosing
    them at evenly spaced intervals."
    Returns the index of chromosomes that have been selected.
    '''
    total_fitness = 0
    fitness_scale = []
    for index, individual in enumerate(fitness_list):
        total_fitness += individual
        if index == 0:
            fitness_scale.append(individual)
        else:
            fitness_scale.append(individual + fitness_scale[index - 1])

    selected = []
    num_select=round(SELECT_PERCENT * POP_SIZE)
    fitness_step = total_fitness / num_select
    random_offset = np.random.uniform(0, fitness_step)

    current_fitness_pointer = random_offset
    last_fitness_scale_position = 0
    for index in range(len(fitness_list)):
        for fitness_scale_position in range(last_fitness_scale_position, len(fitness_scale)):
            if fitness_scale[fitness_scale_position] >= current_fitness_pointer:
                selected.append(fitness_scale_position)
                last_fitness_scale_position = fitness_scale_position
                break
        current_fitness_pointer += fitness_step

    return selected


def crossover(parent1, parent2):
    """
    Crossover two genes. Return two new offsprings.
    split by column, then merge
    requirement: two parents have the number of row
    """
    col_idx=int(parent1.shape[1]/2)
    child1=hstack([parent1[:, :col_idx], parent2[:, col_idx:]])
    child2=hstack([parent1[:, col_idx:], parent2[:, :col_idx]])

    return lil_matrix(child1), lil_matrix(child2)


def crossover_v1(parent1, parent2):
    """
    split by row, then merge
    requirement: two parents have the number of row
    """
    row_idx=int(parent1.shape[0]/2)
    child1=hstack([parent1[:row_idx, :], parent2[row_idx:, :]])
    child2=hstack([parent1[row_idx:, :], parent2[:row_idx, :]])

    return lil_matrix(child1), lil_matrix(child2)


def crossover_v2(parent1, parent2):
    """
    split by column
    truncate the parent with more number of row to be the same as the other parent
    """
    row_idx = min(parent1.shape[0], parent2.shape[0])
    # or combine the two matrices by row, then split
    col_idx = parent1.shape[1] // 2
    child1 = hstack([parent1[:row_idx, :col_idx], parent2[:row_idx, col_idx:]])
    child2 = hstack([parent1[:row_idx, col_idx:], parent2[:row_idx, :col_idx]])

    return lil_matrix(child1), lil_matrix(child2)


def crossover_v3(parent1, parent2):
    """
    merge by row first, then spilt by row equally, then split by column
    """
    row_idx = (parent1.shape[0] + parent2.shape[0]) // 2
    merge_parent = lil_matrix(vstack([parent1, parent2]))
    parent1 = merge_parent[:row_idx, :]
    parent2 = merge_parent[row_idx:row_idx*2, :]
    # or combine the two matrices by row, then split
    col_idx = parent1.shape[1] // 2
    child1 = hstack([parent1[:, :col_idx], parent2[:, col_idx:]])
    child2 = hstack([parent1[:, col_idx:], parent2[:, :col_idx]])

    return lil_matrix(child1), lil_matrix(child2)


def crossover_v4(parent1, parent2):
    """
    Crossover two genes. Return two new offsprings.
    Split by column, then merge.
    Requirement: two parents have the number of row, for that, matrix with highest number of rows 
    is reduced by randomly picking its rows to be the same size of smallest matrix.
    """

    if parent1.shape[0]>parent2.shape[0]:
        random_indices = np.random.choice(parent1.shape[0], size=int(parent2.shape[0]), replace=False)
        parent1 = parent1[random_indices, :]
    else:
        random_indices = np.random.choice(parent2.shape[0], size=int(parent1.shape[0]), replace=False)
        parent2 = parent2[random_indices, :]

    col_idx=int(parent1.shape[1]/2)
    child1=hstack([parent1[:, :col_idx], parent2[:, col_idx:]])
    child2=hstack([parent1[:, col_idx:], parent2[:, :col_idx]])

    return lil_matrix(child1), lil_matrix(child2)


def mutate(chrom):
    """
     (Decide on the best number of random indices for flipping the bit.)
    """
    row_col=set()
    for i in range(int(MUTATE_PROB * chrom.shape[0] * chrom.shape[1])):   #or: int(MUTATE_PROB * chrom.count_nonzero())
        if np.random.random() < MUTATE_PROB:
            row = np.random.choice(chrom.shape[0])
            col = np.random.choice(chrom.shape[1])

            while (row, col) in row_col:    #avoid repetition of indexes in the matrix
                row = np.random.choice(chrom.shape[0])
                col = np.random.choice(chrom.shape[1])

            row_col.add((row, col))
            if chrom[row, col]==0:
                chrom[row, col]=1
            else:
                chrom[row, col]=0

    return chrom


def evolve(population, fitness_list):
    """
    Create next generation by selecting, crossing-over and mutating.
    """
    new_population, new_fitness_list = [], []

    for i in range(len(population) // 2):
        # selection
        selected = rank_selection(fitness_list)
        mother_choice = np.random.choice(selected)
        selected.remove(mother_choice)
        father_choice = np.random.choice(selected)
        mother = population[mother_choice]
        father = population[father_choice]

        # crossover
        if np.random.random() < CROSSOVER_PROB:
            child1, child2 = crossover_v4(mother, father)
            new_fitness_list.append(-1)
            new_fitness_list.append(-1)
        else:
            child1 = mother
            child2 = father
            new_fitness_list.append(fitness_list[mother_choice])
            new_fitness_list.append(fitness_list[father_choice])

        # mutation
        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.append(child1)
        new_population.append(child2)

    return new_population, new_fitness_list


def genetic_alg():
    """
    Genetic Algorithms, main function.
    """
    # generate the first random population
    print(f'generate the first generation of {POP_SIZE} individuals')
    start_time = time.time()
    population, fitness_list = init_pop()
    print('time to generate the first generation: {}'.format(time.time() - start_time))

    print('evolving')
    # get the init solution
    solution, max_fitness = get_best(population, fitness_list)
    avg_fitness_list, stat_list = [], []
    stat_list.append(get_stats(population))
    for g in range(MAX_GENERATION):
        dic={}
        start_time = time.time()
        # new generation, including selection, crossover, mutation
        if g > 0:  # do not process this step in the 1st generation, since every fitness = -1
            population, fitness_list = evolve(population, fitness_list)

        # validate the population
        fitness_tuple = validate(population, fitness_list)
        fitness_list = [i[0] for i in fitness_tuple]
        val_score_list = [i[1] for i in fitness_tuple]
        kc_score_list = [i[2] for i in fitness_tuple]
        kc_size_list= [i[3] for i in fitness_tuple]
        dic['fitness_score']= fitness_list
        dic['val_score']=val_score_list
        dic['kc_score']=kc_score_list
        dic['kc_number']=kc_size_list
        utils.append_as_json(dic, './models/evolution/generations.json')

        # find the solution
        temp_sol, temp_fit = get_best(population, fitness_list)
        if temp_fit > max_fitness:
            solution = temp_sol
            max_fitness = temp_fit

        # print progress
        avg_fitness = sum(fitness_list) / POP_SIZE
        avg_fitness_list.append(avg_fitness)
        stat_list.append(get_stats(population))
        print('gen {}, avg fitness {}, avg val {}, stats {}, time {}'.format(
            g, avg_fitness, sum(val_score_list) / POP_SIZE,
            stat_list[-1], time.time() - start_time))
        # save progress
        with open('./models/evolution/best_solution', "wb") as f:
            pickle.dump(solution, f)
        with open('./models/evolution/best_fitness', "wb") as f:
            pickle.dump(max_fitness, f)
        with open('./models/evolution/avg_fitness_list', "wb") as f:
            pickle.dump(avg_fitness_list, f)
        with open('./models/evolution/stat_list', "wb") as f:
            pickle.dump(stat_list, f)

    return solution, max_fitness, avg_fitness_list, stat_list


if __name__ == '__main__':
    args = docopt(__doc__, version='Genetic Algorithm for fruit-fly projection, ver 0.1')
    train_path = args["--train_path"]
    dataset_name = train_path.split('/')[2].split('-')[0]
    print('Dataset name:', dataset_name)
    pathlib.Path(f'./models/evolution/{dataset_name}').mkdir(parents=True, exist_ok=True)

    # global
    POP_SIZE = 400  # hyperparam (400 - 2000)
    if POP_SIZE % 2:  # the pop size should be even
        POP_SIZE -= 1  # otherwise after the 1st generation it will decrease by 1
    MAX_GENERATION = 10  # 4
    SELECT_PERCENT = 0.2  # hyperparam (0.1 - 0.9)
    CROSSOVER_PROB = 0.7  # hyperparam (0.3 - 0.8)
    ELITE_CHROMS = 2  # hyperparam (2 - 10)

    # KC_SIZE = 8834
    MIN_KC, MAX_KC = 1000, 10000
    # MIN_PROJ, MAX_PROJ = 2, 10
    NUM_PROJ = 10
    top_word = 242
    percent_hash = 15
    C = 93
    num_iter = 2000#50
    # if dataset_name == '20news':
    #     num_iter = 2000

    max_thread = int(multiprocessing.cpu_count() * 0.7)

    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')
    vocab, reverse_vocab, logprobs = read_vocab()
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

    print('reading datasets')
    train_set_list, train_label_list = [None, None, None], [None, None, None]
    val_set_list, val_label_list = [None, None, None], [None, None, None]
    train_set_list[0], train_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-train.sp', vectorizer, logprobs)
    val_set_list[0], val_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-val.sp', vectorizer, logprobs)
    train_set_list[1], train_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-train.sp', vectorizer, logprobs)
    val_set_list[1], val_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-val.sp', vectorizer, logprobs)
    train_set_list[2], train_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-train.sp', vectorizer, logprobs)
    val_set_list[2], val_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-val.sp', vectorizer, logprobs)

    PN_SIZE = len(vocab)
    MUTATE_PROB = NUM_PROJ / PN_SIZE * 0.8  # hyperparam

    # main
    best_solution, best_fitness, avg_fitness_list, stat_list = genetic_alg()
    print('MAX FITNESS', best_fitness)
    with open('./models/evolution/best_solution', "wb") as f:
        pickle.dump(best_solution, f)
    with open('./models/evolution/best_fitness', "wb") as f:
        pickle.dump(best_fitness, f)
    with open('./models/evolution/avg_fitness_list', "wb") as f:
        pickle.dump(avg_fitness_list, f)
    with open('./models/evolution/stat_list', "wb") as f:
        pickle.dump(stat_list, f)