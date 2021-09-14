"""Fruit fly classification of wikipedia meta-categories
Usage:
  classify_wiki.py --flypath=<foldername> --C=<n> --num_iter=<n>
  classify_wiki.py (-h | --help)
  classify_wiki.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --flypath=<foldername>          Name of folder where best performing fly is located
  --C=<n>                         C parameter
  --num_iter=<n>                  Number of iterations
"""

from evolve_flies import Fly
import pickle
from hyperparam_search import read_n_encode_dataset
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from hash import read_vocab
from utils import hash_dataset_
from docopt import docopt
from classify import train_model


def evaluate(train_set_list, projection, wta, top_word):
    hash_train = hash_dataset_(dataset_mat=train_set_list, weight_mat=projection,
                               percent_hash=wta, top_words=top_word)
    hash_val = hash_dataset_(dataset_mat=val_set_list, weight_mat=projection,
                             percent_hash=wta, top_words=top_word)
    print("Running classification...")
    val_score, _ = train_model(m_train=hash_train, classes_train=train_label_list,
                               m_val=hash_val, classes_val=val_label_list,
                               C=C, num_iter=num_iter)
    print(val_score)


if __name__ == '__main__':
    args = docopt(__doc__, version='Ideal Words 0.1')
    C = int(args["--C"])
    num_iter = int(args["--num_iter"])
    flypath = args["--flypath"]

    with open(flypath+'best_val_score', 'rb') as f:  # modified the name of the fruit-fly here
      best = pickle.load(f)
        
    projection=best.projection  # get the projection
    wta = best.wta
    top_word=700

    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')
    vocab, reverse_vocab, logprobs = read_vocab()
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

    train_set_list, train_label_list = read_n_encode_dataset('../wikipedia_categories/wiki_cats/wiki_cats_train.sp', vectorizer, logprobs)
    val_set_list, val_label_list = read_n_encode_dataset('../wikipedia_categories/wiki_cats/wiki_cats_val.sp', vectorizer, logprobs)

    evaluate(train_set_list, projection, wta, top_word)



