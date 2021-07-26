"""Common Crawl processor - training of the LDA model with the OCTIS library

Usage:
  lda_octis.py --pathdataset=<foldername> --outputfile=<filename>
  lda_octis.py (-h | --help)
  lda_octis.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --pathdataset=<foldername>	Name of the folder where the preprocessed documents are.
  --outputfile=<filename>		Name of the file where the output of the lda model will be saved.

"""

from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
import numpy as np
import csv
import sys
from docopt import docopt
csv.field_size_limit(sys.maxsize)

def load_dataset(dataset_folder):
	dataset = Dataset()
	dataset.load_custom_dataset_from_folder("./"+dataset_folder)

	return dataset

def train_lda(dataset, out_file):

	model = LDA(num_topics=100, random_state=2345)  # Create model
	model_output = model.train_model(dataset) # Train the model

	np.savez(out_file+".npz",**model_output)
	print(f"Output (a dictionary of array) was saved as '{out_file}'.")


if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Processor')

    dataset_folder = args['--pathdataset']
    out_file = "./"+dataset_folder+"/"+args['--outputfile']

    dataset_loaded=load_dataset(dataset_folder)
    train_lda(dataset_loaded, out_file)
