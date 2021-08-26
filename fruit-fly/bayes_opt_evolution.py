"""Genetic Algorithm for fruit-fly projection
Usage:
  bayes_opt_evolution.py --train_path=<filename>
  bayes_opt_evolution.py (-h | --help)
  bayes_opt_evolution.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<filename>         Name of file to train (processed by sentencepeice)
"""

from bayes_opt import BayesianOptimization
import utils
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
import scipy.stats as ss

from hyperparam_search import read_n_encode_dataset
from classify import train_model
from hash import wta_vectorized, read_vocab


# helpers
def generate_proj():
    """
    Generate a random projection
    """
    kc_size = 4000# np.random.randint(low=MIN_KC, high=MAX_KC)
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
        val_score = prev_fitness - kc_score
        return prev_fitness, val_score, kc_score

    hash_train = hash_dataset_(dataset_mat=train_set, weight_mat=weight_mat,
                               percent_hash=percent_hash, top_words=top_word)
    hash_val = hash_dataset_(dataset_mat=val_set, weight_mat=weight_mat,
                             percent_hash=percent_hash, top_words=top_word)
    # print('training and evaluating')
    val_score, _ = train_model(m_train=hash_train, classes_train=train_label,
                               m_val=hash_val, classes_val=val_label,
                               C=C, num_iter=num_iter)

    # print(kc_size)
    fitness_score = val_score + kc_score
    return fitness_score, val_score, kc_score, kc_size
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


def elitist_selection(fitness_list, ELITE_CHROMS, SELECT_PERCENT, POP_SIZE):
    """
    Tournament selection + elitist selection
    Return the index of chromosomes that have been selected.
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


def crossover_v4_list(parent1_list, parent2_list):
    """
    Crossover two genes. 
    Input is a list containing one matrix in position 0 and a vector in position 1
    Return two new offsprings which is a list with a matrix in position 0 and a vector in position 1.
    """
    child1_list=[]
    child2_list=[]
    for parent1, parent2 in zip(parent1_list, parent2_list):
      if parent1.ndim > 1:
        if parent1.shape[0]>parent2.shape[0]:
          random_indices = np.random.choice(parent1.shape[0], size=int(parent2.shape[0]), replace=False)
          parent1 = parent1[random_indices, :]
        else:
          random_indices = np.random.choice(parent2.shape[0], size=int(parent1.shape[0]), replace=False)
          parent2 = parent2[random_indices, :]

        col_idx=int(parent1.shape[1]/2)
        child1=hstack([parent1[:, :col_idx], parent2[:, col_idx:]])
        child2=hstack([parent1[:, col_idx:], parent2[:, :col_idx]])
        child1_list.append(child1)
        child2_list.append(child2)
      else:
        col_idx=int(parent1.shape[0]/2)
        child1=np.concatenate([parent1[:col_idx], parent2[col_idx:]])
        child2=np.concatenate([parent1[col_idx:], parent2[:col_idx]])
        child1_list.append(child1)
        child2_list.append(child2)

    return child1_list, child2_list


def mutate_list(chrom_list, MUTATE_PROB, MUTATE_PROB_VEC):
    """
    Modifies the chromosome by flipping the bit of indexes that 
    have probability lower that mutation rate. 
    Takes as input a list with a matrix in position 0 and a vector in position 1. 
    Return a list with a mutated matrix in position 0 and a mutated vector in position 1.
    """
    mutated_chrom=[]
    for chrom in chrom_list:
      if chrom.ndim>1:
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
        mutated_chrom.append(chrom)
      else:
        for i in range(chrom.shape[0]):
          if np.random.random() < MUTATE_PROB_VEC: # a hyperpameter
            print(i, chrom[i])
            if chrom[i]==0:
              chrom[i]=1
            else:
              chrom[i]=0
        mutated_chrom.append(chrom)

    return mutated_chrom


def evolve(population, fitness_list, ELITE_CHROMS, SELECT_PERCENT, POP_SIZE, 
				CROSSOVER_PROB, MUTATE_PROB, MUTATE_PROB_VEC):
    """
    Create next generation by selecting, crossing-over and mutating.
    """
    new_population, new_fitness_list = [], []

    for i in range(len(population) // 2):
        # selection
        selected = elitist_selection(fitness_list, ELITE_CHROMS, SELECT_PERCENT, POP_SIZE)
        mother_choice = np.random.choice(selected)
        selected.remove(mother_choice)
        father_choice = np.random.choice(selected)
        mother = population[mother_choice]
        father = population[father_choice]

        # crossover
        if np.random.random() < CROSSOVER_PROB:
            child1, child2 = crossover_v4_list(mother, father)
            new_fitness_list.append(-1)
            new_fitness_list.append(-1)
        else:
            child1 = mother
            child2 = father
            new_fitness_list.append(fitness_list[mother_choice])
            new_fitness_list.append(fitness_list[father_choice])

        # mutation
        child1 = mutate_list(child1, MUTATE_PROB, MUTATE_PROB_VEC)
        child2 = mutate_list(child2, MUTATE_PROB, MUTATE_PROB_VEC)

        new_population.append(child1)
        new_population.append(child2)

    return new_population, new_fitness_list


def genetic_alg(POP_SIZE, CROSSOVER_PROB, SELECT_PERCENT, MUTATE_PROB, MUTATE_PROB_VEC, ELITE_CHROMS):
    """
    Genetic Algorithms, main function.
    """
    # generate the first random population

    MAX_GENERATION=4

    print(f'generate the first generation of {POP_SIZE} individuals')
    start_time = time.time()
    population, fitness_list = init_pop()
    print('time to generate the first generation: {}'.format(time.time() - start_time))

    print('evolving')
    # get the init solution
    solution, max_fitness = get_best(population, fitness_list)
    avg_fitness_list, stat_list = [], []
    stat_list.append(get_stats(population))
    total_improvement=[]
    last_fitness=0
    for g in range(MAX_GENERATION):
        dic={}
        start_time = time.time()
        # new generation, including selection, crossover, mutation
        if g > 0:  # do not process this step in the 1st generation, since every fitness = -1
            population, fitness_list = evolve(population, fitness_list, ELITE_CHROMS, SELECT_PERCENT, 
            				POP_SIZE, CROSSOVER_PROB, , MUTATE_PROB, MUTATE_PROB_VEC)

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
        utils.append_as_json(dic, './models/evolution/generations_bayes.json')

        # find the solution
        temp_sol, temp_fit = get_best(population, fitness_list)
        if temp_fit > max_fitness:
            solution = temp_sol
            max_fitness = temp_fit

        # print progress
        avg_fitness = sum(fitness_list) / POP_SIZE
        if last_fitness == 0:
        	last_fitness=avg_fitness
        improvement_fitness=avg_fitness-last_fitness
        last_fitness=avg_fitness
        total_improvement.append(improvement_fitness)
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

    return sum(total_improvement)


def bayesian_optimization(NUM_PROJ, PN_SIZE, WTA_DIM):
  def evolve_bayes(POP_SIZE, SELECT_PERCENT, CROSSOVER_PROB, ELITE_CHROMS, MUTATE_PROB, MUTATE_PROB_VEC):
  	POP_SIZE = int(POP_SIZE) 
    if POP_SIZE % 2:  # the pop size should be even
        POP_SIZE -= 1  # otherwise after the 1st generation it will decrease by 1
    MUTATE_PROB = NUM_PROJ / PN_SIZE * MUTATE_PROB
    MUTATE_PROB_VEC = MUTATE_PROB_VEC/WTA_DIM.shape[0]
    ELITE_CHROMS = int(ELITE_CHROMS)

    return genetic_alg(POP_SIZE, CROSSOVER_PROB, SELECT_PERCENT, MUTATE_PROB, 
    					MUTATE_PROB_VEC, ELITE_CHROMS)

  optimizer = BayesianOptimization(
      f=evolve_bayes,
      pbounds={"POP_SIZE": (400, 2000), 'SELECT_PERCENT': (0.1, 0.9), 
      			'CROSSOVER_PROB':(0.3, 0.8), 'ELITE_CHROMS': (2, 10), 
      			'MUTATE_PROB':(0.2, 0.8), 'MUTATE_PROB_VEC':(0.5, 2)},
      random_state=123,
      verbose=2
  )

  optimizer.maximize(n_iter=100)
  dic = {}
  dic['pop_size']=int(optimizer.max['params']['POP_SIZE'])
  dic['select_percent']=optimizer.max['params']['SELECT_PERCENT']
  dic['cross_prob'] = optimizer.max['params']['CROSSOVER_PROB']
  dic['elite_chroms'] = int(optimizer.max['params']['ELITE_CHROMS'])
  dic['mutate_prob'] = optimizer.max['params']['MUTATE_PROB']
  dic['mutate_prob_vec'] = optimizer.max['params']['MUTATE_PROB_VEC']

  print("Final result:", optimizer.max)
  utils.append_as_json(dic, "./models/evolution/bayes_results.json")


if __name__ == '__main__':
    args = docopt(__doc__, version='Bayesian Optimization for the Genetic Algorithm of the fruit-fly projection, ver 0.1')
    train_path = args["--train_path"]
    dataset_name = train_path.split('/')[2].split('-')[0]
    print('Dataset name:', dataset_name)
    pathlib.Path(f'./models/evolution/{dataset_name}').mkdir(parents=True, exist_ok=True)

    # KC_SIZE = 8834
    MIN_KC, MAX_KC = 1000, 10000
    # MIN_PROJ, MAX_PROJ = 2, 10
    NUM_PROJ = 10
    WTA_DIM =  # to be set
    top_word = 242
    percent_hash = 15
    C = 93
    num_iter = 50
    if dataset_name == '20news':
        num_iter = 2000

    max_thread = int(multiprocessing.cpu_count() * 0.7)

    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')
    vocab, reverse_vocab, logprobs = read_vocab()
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
    train_set, train_label = read_n_encode_dataset(train_path, vectorizer, logprobs)
    val_set, val_label = read_n_encode_dataset(train_path.replace('train', 'val'), vectorizer, logprobs)

    PN_SIZE = train_set.shape[1]

	bayesian_optimization(NUM_PROJ, PN_SIZE, WTA_DIM)