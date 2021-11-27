"""Create projections for the fruit fly

Usage:
  mkprojections.py --kc=<n> --size=<n>
  mkprojections.py (-h | --help)
  mkprojections.py --version

Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --kc=<n>                        Number of KC cells.
  --size=<n>                      Size of projections

"""

import os
import random
import numpy as np
from docopt import docopt

def read_vocab():
    c = 0
    vocab = {}
    reverse_vocab = {}
    logprobs = []
    with open("../spm/spmcc.vocab") as f:
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


def create_projections(PN_size, KC_size, proj_size,d, trial):
    # print("Creating",KC_size,"projections...")
    projection_functions = {}
    idx = list(range(PN_size))
    potential_projections = []

    while len(potential_projections) < KC_size:
        random.shuffle(idx)

        for i in range(0,len(idx),proj_size):
            p = idx[i:i+proj_size]
            potential_projections.append(p)

    f=open(os.path.join(d,"spmcc_"+str(trial)+".projs"),'w')
    for cell in range(KC_size):
        p = np.array(potential_projections[cell])
        projection_functions[cell] = p
        pw = ""
        for i in p:
            pw+=str(i)+' '
        f.write(pw[:-1]+'\n')
    f.close()
    return str(os.path.join(d,"spmcc_"+str(trial)+".projs")) #projection_functions


if __name__ == '__main__':
    args = docopt(__doc__, version='Fruit Fly Projections 0.1')

    vocab, reverse_vocab, logprobs = read_vocab()

    PN_size = len(vocab)
    KC_size = int(args["--kc"])
    proj_size = int(args["--size"])

    d = "models/projection/kc"+str(KC_size)+"-p"+str(proj_size)
    os.mkdir(d)
    projection_functions = create_projections(PN_size, KC_size, proj_size,d, trial=0)

