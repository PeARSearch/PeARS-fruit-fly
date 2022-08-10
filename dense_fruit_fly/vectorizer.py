import numpy as np
import re
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from utils import read_vocab, wta_vectorized
from sklearn import preprocessing


def encode_docs(doc_list, vectorizer, logprobs, power, top_words):
    logprobs = np.array([logprob ** power for logprob in logprobs])
    X = vectorizer.fit_transform(doc_list)
    X = X.multiply(logprobs)
    X = wta_vectorized(X.toarray(), top_words / X.shape[1])
    X = csr_matrix(X)
    return X


def read_n_encode_dataset(path=None, vectorizer=None, logprobs=None, power=None, top_words=None):
    # read
    doc_list, label_list = [], []
    doc = ""
    with open(path) as f:
        for l in f:
            l = l.rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*class=([^ ]*)>", l)
                label = m.group(1)
                label = label.split("|")
                label_list.append(label)
            elif l[:5] == "</doc":
                doc_list.append(doc)
                doc = ""
            else:
                doc += l + ' '

    # encode
    X = encode_docs(doc_list, vectorizer, logprobs, power, top_words)
    return X, None, label_list


def init_vectorizer(lang): 
    spm_vocab = f"../spm/spm.{lang}.vocab"
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')
    return vectorizer, logprobs

def vectorize(lang, spf, logprob_power, top_words):
    '''Takes input file and return vectorized /scaled dataset'''
    vectorizer, logprobs = init_vectorizer(lang)
    dataset, wikititles, wikicats = read_n_encode_dataset(spf, vectorizer, logprobs, logprob_power, top_words)
    dataset = dataset.todense()
    return dataset, wikititles, wikicats

def scale(dataset):
    #scaler = preprocessing.MinMaxScaler().fit(dataset)
    scaler = preprocessing.Normalizer(norm='l2').fit(dataset)
    return scaler.transform(dataset)

def vectorize_scale(lang, spf, logprob_power, top_words):
    dataset, titles, _ = vectorize(lang,spf,logprob_power,top_words)
    return scale(dataset), titles
