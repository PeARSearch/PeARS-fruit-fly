"""Genetic Algorithm for fruit-fly projection
Usage:
  evolve_on_budget.py [--dataset=<wos|wiki|20news>]
  evolve_on_budget.py (-h | --help)
  evolve_on_budget.py --version

Options:
  --dataset=<wos|wiki|news>       Name of dataset to be tested. If flag is unused, all datasets are tested.
  -h --help                       Show this screen.
  --version                       Show version.
"""

import os
import pickle
import numpy as np
import joblib
import multiprocessing
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack, vstack, lil_matrix, coo_matrix
from docopt import docopt
import time
from datetime import datetime
from copy import deepcopy

from hyperparam_search import read_n_encode_dataset
from classify import train_model
from hash import read_vocab
from utils import hash_dataset_, append_as_json, get_stats
import itertools

class Fly:
    def __init__(self):
        self.kc_size = np.random.randint(low=MIN_KC, high=MAX_KC)
        self.wta = np.random.uniform(low=MIN_WTA, high=MAX_WTA)
        weight_mat = np.zeros((self.kc_size, PN_SIZE))
        for i in range(self.kc_size):
            num_proj = np.random.randint(low=MIN_PROJ, high=MAX_PROJ)
            for j in np.random.randint(PN_SIZE, size=num_proj):
                weight_mat[i, j] = 1
        self.projections = lil_matrix(weight_mat)
        self.val_scores = [0, 0, 0]
        self.kc_score = 1 / np.log10(int(self.kc_size * self.wta / 100))
        self.is_evaluated = False

    def get_fitness(self):
        if not self.is_evaluated:
            return 0
        if DATASET == "all":
            return np.mean(self.val_scores) 
        else:
            return np.sum(self.val_scores)

    def evaluate(self):
        start_time = time.time()
        val_score_list = []
        for i in range(len(train_set_list)):
            if train_set_list[i] is None:
                val_score_list.append(0)
                continue
            hash_train = hash_dataset_(dataset_mat=train_set_list[i], weight_mat=self.projections,
                                       percent_hash=self.wta, top_words=TOP_WORDS)
            hash_val = hash_dataset_(dataset_mat=val_set_list[i], weight_mat=self.projections,
                                     percent_hash=self.wta, top_words=TOP_WORDS)
            val_score, _ = train_model(m_train=hash_train, classes_train=train_label_list[i],
                                       m_val=hash_val, classes_val=val_label_list[i],
                                       C=C, num_iter=NUM_ITER)
            val_score_list.append(val_score)
        self.val_scores = val_score_list
        self.is_evaluated = True
        #return val_score_list, time.time() - start_time
        return val_score_list, self.kc_size, self.wta

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
            stats = fly.evaluate()
            print("FLY STATS:",stats)
        else:
            pass
    joblib.Parallel(n_jobs=max_thread, prefer="threads")(
        joblib.delayed(_eval_individual)(fly) for fly in population)


def select_elite_tournament(fitness_list: list, elite: int, select_percent: float):
    """
    Tournament + elitism selection.
    Return the index of genes that has been selected.
    """
    selected = []
    num_select = round(select_percent * len(fitness_list)) - 2
    if num_select < 2:
        num_select = 2  # mother, father

    #fitness2idx = {k: v for v, k in enumerate(fitness_list)} #This is problemative with duplicate values
    elite_chroms = elite
    if num_select < elite_chroms:
        elite_chroms = num_select
    elites = sorted(fitness_list, reverse=True)[:elite_chroms]
    for eli in elites:
        idx = [index for index, value in enumerate(fitness_list) if value == eli]
        #selected.add(fitness2idx[eli])
        selected.extend(idx)
    selected = set(selected)

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
    Do crossover on two components: the projection and the wta
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
        child1.projections = parent1.projections[random_indices, :]
        child1.kc_size = child1.projections.shape[0]
    else:
        random_indices = np.random.choice(parent2.kc_size, size=int(parent1.kc_size), replace=False)
        child2.projections = parent2.projections[random_indices, :]
        child2.kc_size = child2.projections.shape[0]
    # swap
    col_idx = int(child1.projections.shape[1]/2)
    new_proj_1 = hstack([child1.projections[:, :col_idx], child2.projections[:, col_idx:]])
    new_proj_2 = hstack([child1.projections[:, col_idx:], child2.projections[:, :col_idx]])
    
    child1.projections, child2.projections = lil_matrix(new_proj_1), lil_matrix(new_proj_2)

    # then, crossover wta
    wta_low, wta_high = sorted([child1.wta, child2.wta])
    child1.wta = np.random.uniform(low=wta_low, high=wta_high)
    child2.wta = np.random.uniform(low=wta_low, high=wta_high)

    # update kc_score
    child1.kc_score = 1 / np.log10(int(child1.kc_size * child1.wta / 100))
    child2.kc_score = 1 / np.log10(int(child2.kc_size * child2.wta / 100))

    return child1, child2



def mutate(individual: Fly, mutate_prob_proj: float, mutate_scale_wta: float, grow: bool):
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
        mutated_indiv.projections[i] = new_row
        mutated_indiv.projections = lil_matrix(mutated_indiv.projections)
        
    # add a few new rows
    if grow:
        num_new_row = np.random.randint(low=5, high=10)
        new_mat = np.zeros((num_new_row, PN_SIZE))
        for i in range(num_new_row):
            num_proj = np.random.randint(low=MIN_PROJ, high=MAX_PROJ)
            for j in np.random.randint(PN_SIZE, size=num_proj):
                new_mat[i, j] = 1
        # concat the old part with the new part
        mutated_indiv.projections = vstack([mutated_indiv.projections, lil_matrix(new_mat)])
        mutated_indiv.projections = lil_matrix(mutated_indiv.projections)

    # then, mutate the wta
    new_wta = np.random.normal(loc=individual.wta, scale=mutate_scale_wta)
    if new_wta < MIN_WTA:
        new_wta = MIN_WTA
    if new_wta > MAX_WTA:
        new_wta = MAX_WTA
    mutated_indiv.wta = new_wta

    return mutated_indiv

def evolve(population: list, elite: int, select_percent: float,
        crossover_prob: float, mutate_prob_proj: float, mutate_scale_wta: float, grow: bool):
    """
    Create next generation by selecting, crossing-over and mutating.
    """
    def _select_crossover_mutate():
        # select
        selected = select_elite_tournament(fitness_list, elite, select_percent)
        #print("SELECTED INDIVIDUALS:",selected)
        mother_choice = np.random.choice(selected)
        selected.remove(mother_choice)
        father_choice = np.random.choice(selected)
        mother = population[mother_choice]
        father = population[father_choice]
        

        # crossover
        if np.random.random() < crossover_prob:
            #print("Generating two children...")
            child1, child2 = crossover(mother, father)
        else:
            #print("Children will be identical to parents")
            child1 = mother
            child2 = father

        # mutation
        child1 = mutate(child1, mutate_prob_proj, mutate_scale_wta, grow)
        child2 = mutate(child2, mutate_prob_proj, mutate_scale_wta, grow)
        child1.kc_size = child1.projections.shape[0]
        child2.kc_size = child2.projections.shape[0]

        #print("CROSSOVER - MOTHER:",mother_choice, mother.kc_size, "FATHER:",father_choice, father.kc_size, "MUTATION - CHILDREN KC SIZES",child1.kc_size,child2.kc_size)
        return child1, child2

    fitness_list = [individual.get_fitness() for individual in population]
    print("FITNESS LIST:",fitness_list)

    children_list = joblib.Parallel(n_jobs=max_thread, prefer="threads")(
        joblib.delayed(_select_crossover_mutate)() for _ in range(len(population) // 2))

    new_population = []
    for pair in children_list:
        new_population.append(pair[0])
        new_population.append(pair[1])

    #print("NEW POPULATION:",[i.kc_size for i in new_population])
    return new_population


def genetic_alg(pop_size: int, crossover_prob: float, elite: int, select_percent: float,
        mutate_prob_proj: float, mutate_scale_wta: float, grow: bool):
    """
    Genetic Algorithms, main function.
    """
    # create log
    log_file = './models/evolution/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.json'
    best_fly_file = './models/evolution/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.fly'
    append_as_json({'pop_size': pop_size, 'max_generation': MAX_GENERATION, 'crossover_prob': crossover_prob,
                    'select_percent': select_percent, 'mutate_prob_proj': mutate_prob_proj,
                    'mutate_scale_wta': mutate_scale_wta}, log_file)
    append_as_json({'min_wta': MIN_WTA, 'max_wta': MAX_WTA,
                    'min_kc': MIN_KC, 'max_kc': MAX_KC, 'min_proj': MIN_PROJ, 'max_proj': MAX_PROJ,
                    'TOP_WORDS': TOP_WORDS, 'C': C, 'NUM_ITER': NUM_ITER}, log_file)

    # generate the first random population
    print(f'generate the first generation of {pop_size} individuals')
    start_time = time.time()
    population = init_pop(pop_size)
    print('time to generate the first generation: {}'.format(time.time() - start_time))
    fitness_list = [individual.get_fitness() for individual in population]
    overall_best_fly = None

    def _return_best_fly():
        best_fly = population[0]
        for individual in population:
            if individual.get_fitness() > best_fly.get_fitness():
                best_fly = individual
        return best_fly, best_fly.get_fitness()

    print('evolving')
    total_improvement = []
    last_fitness = 0
    for g in range(MAX_GENERATION):
        print("\n\nGENERATION",g)
        start_time = time.time()
        # new generation, including selection, crossover, mutation
        if g > 0:  # do not process this step in the 1st generation, since every fitness = 0
            population = evolve(population, elite, select_percent,
                                crossover_prob, mutate_prob_proj, mutate_scale_wta, grow)

        # evaluate the population
        eval_pop(population)
        fitness_list = [individual.get_fitness() for individual in population]
        # print progress
        avg_fitness = np.mean(fitness_list)
        var_fitness = np.var(fitness_list)
        print("GEN",g,"AVG FITNESS:",avg_fitness,"VAR:",var_fitness)
        if g == 0:
             last_fitness = avg_fitness
        improvement_fitness = avg_fitness - last_fitness
        last_fitness = avg_fitness
        total_improvement.append(improvement_fitness)
        stats = get_stats(population)
        stats['gen'] = g
        stats['time'] = time.time() - start_time

        append_as_json(stats, log_file)
        print(stats)
        best_fly, best_fitness = _return_best_fly()
        print("CURRENT BEST FLY:",best_fitness,best_fly.kc_size,best_fly.wta)
        if overall_best_fly == None or overall_best_fly.get_fitness() < best_fitness:
            print("UPDATING OVERALL BEST FLY:",best_fitness,best_fly.kc_size,best_fly.wta)
            overall_best_fly = best_fly
    print("sum improvement:", sum(total_improvement))
    return overall_best_fly


if __name__ == '__main__':
    args = docopt(__doc__, version='Genetic Algorithm of the fruit-fly projection, with budgeting, ver 0.1')

    if not os.path.isdir('./models/evolution'):
        os.makedirs('./models/evolution')
    
    BUDGET_HS_SIZE = int(input("Please enter the max hash size for your model: "))
    print(BUDGET_HS_SIZE)

    BUDGET_ITERS = int(input("Please enter the max number of models to train: "))
    print(BUDGET_ITERS)

    sp = spm.SentencePieceProcessor()
    
    DATASET = "all"
    
    if args['--dataset']:
        DATASET = args['--dataset']
        print("VOCAB: "+DATASET+".spm.vocab")
        sp.load("../spm/spm."+DATASET+".model")
        vocab, reverse_vocab, logprobs = read_vocab("../spm/spm."+DATASET+".vocab")
    else:
        print("VOCAB: spmcc.vocab")
        sp.load('../spm/spmcc.model')
        vocab, reverse_vocab, logprobs = read_vocab("../spm/spmcc.vocab")

    #Hyperparameters for GA
    POP_SIZE = 50
    MAX_GENERATION = int(BUDGET_ITERS / POP_SIZE)
    GROW = False
    ELITE = 6
    CROSSOVER_PROB = 0.5
    PERCENT_SELECTED = 0.4
    MUTATE_PROJ_PROB = 0.04
    MUTATE_WTA_SCALE = 2


    #Hyperparameters for document representation
    TOP_WORDS = 150
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
   
    #Hyperparameters for fly model
    #Apart from PN_SIZE, dependent on length of vocab, others are optimized by GA
    MIN_WTA, MAX_WTA = 5, 20
    if GROW:
        MIN_KC, MAX_KC = 20, int(BUDGET_HS_SIZE / (MAX_WTA / 100))
    else:
        MIN_KC, MAX_KC = 20, int(BUDGET_HS_SIZE / (MAX_WTA / 100) - 90) #To account for growth at a rate of max 10 extra KCs, for 10 generations
    MIN_PROJ, MAX_PROJ = 5, 20
    PN_SIZE = len(vocab)

    #Hyperparameters for classifier, kept fixed
    C = 100
    NUM_ITER = 50
    
    #Hyperparameters for parallel processing
    max_thread = int(multiprocessing.cpu_count() * 0.2)

    generation_fitnesses = []

    print('reading datasets')
    num_dataset = 3
    train_set_list, train_label_list = [None] * num_dataset, [None] * num_dataset
    val_set_list, val_label_list = [None] * num_dataset, [None] * num_dataset

    if DATASET in ["all","wos"]:
        train_set_list[0], train_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-train.sp', vectorizer, logprobs)
        val_set_list[0], val_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-val.sp', vectorizer, logprobs)
        train_set_list
    if DATASET in ["all","wiki"]:
        train_set_list[1], train_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-train.sp', vectorizer, logprobs)
        val_set_list[1], val_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-val.sp', vectorizer, logprobs)
    if DATASET in ["all","20news"]:
        train_set_list[2], train_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-train.sp', vectorizer, logprobs)
        val_set_list[2], val_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-val.sp', vectorizer, logprobs)

    param_grid = {'GROW': [True,False], 'CROSSOVER_PROB' : [0.5,0.7,0.9], 'ELITE' : [2,4,6,8], 'PERCENT_SELECTED' : [0.1, 0.3], 'MUTATE_PROJ_PROB' : [0.05, 0.1, 0.2], 'TOP_WORDS' : [50,100,150,200]}
    grid = list(itertools.product(*param_grid.values()))

    for params in grid:
        GROW, CROSSOVER_PROB, ELITE, PERCENT_SELECTED, MUTATE_PROJ_PROB, TOP_WORDS = params
        print('\n\npop_size:',POP_SIZE, 'max_generation:', MAX_GENERATION, 'crossover_prob:', CROSSOVER_PROB, 'elite:', ELITE,  
                'select_percent:', PERCENT_SELECTED, 'mutate_prob_proj:', MUTATE_PROJ_PROB, 'growth:', GROW, 'top_words:',TOP_WORDS, 
                'mutate_scale_wta:', MUTATE_WTA_SCALE, 'MIN_KC:', MIN_KC, 'MAX_KC:', MAX_KC) 
        overall_best_fly = genetic_alg(pop_size=POP_SIZE, crossover_prob=CROSSOVER_PROB, elite=ELITE, select_percent=PERCENT_SELECTED, mutate_prob_proj=MUTATE_PROJ_PROB, mutate_scale_wta=MUTATE_WTA_SCALE, grow=GROW)
        print('\n\nRESULTS FOR PARAMS: pop_size:',POP_SIZE, 'max_generation:', MAX_GENERATION, 'crossover_prob:', CROSSOVER_PROB, 'elite:', ELITE, 
                'select_percent:', PERCENT_SELECTED, 'mutate_prob_proj:', MUTATE_PROJ_PROB, 'growth:', GROW, 'top_words:',TOP_WORDS, 
                'mutate_scale_wta:', MUTATE_WTA_SCALE, 'best_fly_fitness:', overall_best_fly.get_fitness(), 
                'best_fly_kc_size:', overall_best_fly.kc_size, 'best_fly_wta:', overall_best_fly.wta)
