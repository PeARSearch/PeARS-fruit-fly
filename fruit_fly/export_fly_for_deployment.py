"""Export fly for deployment

Usage:
  export_fly_for_deployment.py --fly=<path> --dpath=<path>
  export_fly_for_deployment.py (-h | --help)
  export_fly_for_deployment.py --version
Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --fly=<path>              Path to selected fly model.
  --dpath=<path>            Path to deployment folder.

"""

from docopt import docopt
import pickle
import numpy as np
from os.path import join
from evolve_flies import Fly

class DeployedFly:
    def __init__(self):
        self.kc_size = None
        self.wta = None
        self.projection = None
        self.val_scores = []

def export(fly):
    dfly = DeployedFly()
    dfly.kc_size = fly.kc_size
    dfly.wta = fly.wta
    dfly.projection = fly.projection
    dfly.val_scores = fly.val_scores
    return dfly

args = docopt(__doc__, version='Hashing a document, ver 0.1')
deployment_path = args['--dpath']
fly_model = args['--fly']

with open(fly_model, 'rb') as f:  # modified the name of the fruit-fly here
    fly_model = pickle.load(f)

deployment_fly = export(fly_model)

with open(join(deployment_path,"fly.m"), "wb") as f:
    pickle.dump(deployment_fly, f)
