import pickle
import numpy as np

from evolve_flies import Fly


def print_res(criteria):
    with open(f'./models/evolution/{criteria}', 'rb') as f:
        best = pickle.load(f)
    print('{: <20} kc {}, wta {: .3f}%, val score wos-wiki-20news {}'.format(criteria, best.kc_size, best.wta, best.val_scores))


if __name__ == '__main__':
    print_res('best_fitness')
    print_res('best_val_score')
    print_res('best_kc_score')
    print_res('best_val_wos')
    print_res('best_val_wikipedia')
    print_res('best_val_20news')
