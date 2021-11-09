"""Hash documents with selected fly

Usage:
  show_proj.py --docfile=<filename> --fly=<pathfly>
  show_proj.py (-h | --help)
  show_proj.py --version
Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --docfile=<path>          Path of file containing documents and information about each doc such as URL and label.
  --fly=<path>              Path to selected fly model.

"""

import pickle
import numpy as np
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
# from evolve_flies import Fly
from scipy.sparse import csr_matrix, lil_matrix
from docopt import docopt
from hash import wta, return_keywords, read_vocab
from hyperparam_search import read_n_encode_dataset
from classify import train_model
from hash import read_vocab
from utils import hash_dataset_


class Fly:
    def __init__(self):
        self.kc_size = 9000
        self.wta = 3
        weight_mat = np.zeros((self.kc_size, PN_SIZE))
        for i in range(self.kc_size):
            num_proj = 3
            for j in np.random.randint(PN_SIZE, size=num_proj):
                weight_mat[i, j] = 1
        self.projection = lil_matrix(weight_mat)
        self.val_scores = [0, 0, 0]
        self.kc_score = 1 / np.log10(int(self.kc_size * self.wta / 100))
        self.is_evaluated = False

    def get_fitness(self):
        if not self.is_evaluated:
            return 0
        return np.mean(self.val_scores)

    def evaluate(self):
        val_score_list = []
        classifier = None
        for i in range(len(train_set_list)):
            hash_train = hash_dataset_(dataset_mat=train_set_list[i], weight_mat=self.projection,
                                       percent_hash=self.wta, top_words=top_word)
            hash_val = hash_dataset_(dataset_mat=val_set_list[i], weight_mat=self.projection,
                                     percent_hash=self.wta, top_words=top_word)
            val_score, lg_model = train_model(m_train=hash_train, classes_train=train_label_list[i],
                                       m_val=hash_val, classes_val=val_label_list[i],
                                       C=2, num_iter=2000)
            val_score_list.append(val_score)
            if i == 1:
                classifier = lg_model
        self.val_scores = val_score_list
        self.is_evaluated = True
        return val_score_list, classifier


def sort_nonzeros(x):
    sidx = np.argsort(x)
    out_idx = sidx[np.in1d(sidx, np.flatnonzero(x!=0))][::-1]
    out_x = x[out_idx]
    return out_x, out_idx


def inspect_hash(f_dataset, fly):
    with open(f_dataset, 'r') as f:
        for l in f:
            print('--- input: ', l.strip())
            # encode
            ll = sp.encode_as_pieces(l)
            X = vectorizer.fit_transform([" ".join(ll)])
            X = csr_matrix(X)
            X = X.multiply(logprobs)

            # keep a few hundred of important words
            vec = wta(X.toarray()[0], top_word, percent=False)
            kwds = [reverse_vocab[w] for w in return_keywords(vec)]
            print('--- top 10 word: ', kwds)

            # projection
            mat_mul = lil_matrix(vec).dot(fly.projection.T)

            # wta
            wta_vec = wta(mat_mul.toarray()[0], fly.wta, percent=True)

            # binarization
            bin_vec = (wta_vec > 0).astype(np.int_)
            class_idx = classifier.predict(bin_vec.reshape(1, -1))
            print('--- predict class: ', class_idx[0])

            # inspect
            values, ids = sort_nonzeros(wta_vec)
            print('--- num activations: ', len(ids))
            for i in ids:
                words, logprob = [], []
                pn_ids = fly.projection[i].nonzero()[1]
                for j in pn_ids:
                    words.append(reverse_vocab[j])
                    logprob.append(X.toarray()[0][j])
                print('   proj', i, words, logprob)

            print('--------------------------------------\n')


if __name__ == '__main__':
    args = docopt(__doc__, version='Hashing a document, ver 0.1')
    f_dataset = args['--docfile']
    # fly_model = args['--fly']

    top_word = 100
    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')
    vocab, reverse_vocab, logprobs = read_vocab()
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
    PN_SIZE = len(vocab)

    print('reading datasets')
    num_dataset = 3
    train_set_list, train_label_list = [None] * num_dataset, [None] * num_dataset
    val_set_list, val_label_list = [None] * num_dataset, [None] * num_dataset
    train_set_list[0], train_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-train.sp', vectorizer,
                                                                   logprobs)
    val_set_list[0], val_label_list[0] = read_n_encode_dataset('../datasets/wos/wos11967-val.sp', vectorizer, logprobs)
    train_set_list[1], train_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-train.sp',
                                                                   vectorizer, logprobs)
    val_set_list[1], val_label_list[1] = read_n_encode_dataset('../datasets/wikipedia/wikipedia-val.sp', vectorizer,
                                                               logprobs)
    train_set_list[2], train_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-train.sp',
                                                                   vectorizer, logprobs)
    val_set_list[2], val_label_list[2] = read_n_encode_dataset('../datasets/20news-bydate/20news-bydate-val.sp',
                                                               vectorizer, logprobs)

    # with open(fly_model, 'rb') as f:  # modified the name of the fruit-fly here
    #     fly_model = pickle.load(f)
    fly_model = Fly()

    # evaluate the fly
    print('evaluating the fly')
    val_scores, classifier = fly_model.evaluate()
    print('wos wiki 20news: ', val_scores)

    inspect_hash(f_dataset, fly=fly_model)
