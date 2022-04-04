"""Fruit fly classification
Usage:
  classify.py --file=<filename> --C=<n> --num_iter=<n>
  classify.py (-h | --help)
  classify.py --version
Options:
  -h --help          Show this screen.
  --version          Show version.
  --filename         Name of training file
  --C=<n>            C parameter
  --num_iter=<n>     Number of iterations
"""

from docopt import docopt
import numpy as np
from sklearn import linear_model
import pickle

#print(device)
#random.seed(77)


def get_single_classes(classes):
    classes = list(classes.values())
    single_classes = list(set(classes))
    class_ids = {}
    i = 0
    for cl in single_classes:
        class_ids[cl] = i
        i+=1
    print(class_ids)
    return class_ids


def make_output(classes,class_ids):
    classes = list(classes.values())
    return [class_ids[c] for c in classes]


def train_model(m_train,classes_train,m_val,classes_val,C,num_iter):
    lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear',
                                         max_iter=num_iter, C=C, verbose=0)
    lm.fit(m_train, classes_train)
    score = lm.score(m_val,classes_val)
    # print(lm.predict(m_val))
    # print(score)
    return score, lm


def prepare_data(tr_file):
    dev_file = tr_file.replace("train","val")
    # print("Reading dataset...")
    m_train = pickle.load(open(tr_file,'rb')).todense()
    #print(m_train)
    #m_train = preprocessing.normalize(m_train, norm='l1')
    m_val = pickle.load(open(dev_file,'rb')).todense()
    #print(m_val)
    #m_val = preprocessing.normalize(m_val, norm='l1')

    class_ids = get_single_classes(pickle.load(open(tr_file.replace("hs","cls"),'rb')))
    classes_train = np.array(make_output(pickle.load(open(tr_file.replace("hs","cls"),'rb')),class_ids))
    #print(classes_train)
    classes_val = np.array(make_output(pickle.load(open(dev_file.replace("hs","cls"),'rb')),class_ids))

    ids_train = np.array([i for i in range(m_train.shape[0])])
    #print(ids_train)
    ids_val = np.array([i for i in range(m_val.shape[0])])

    return m_train,classes_train,m_val,classes_val,ids_train,ids_val


if __name__ == '__main__':
    args = docopt(__doc__, version='Ideal Words 0.1')
    tr_file = args["--file"]
    C = int(args["--C"])
    num_iter = int(args["--num_iter"])

    m_train,classes_train,m_val,classes_val,ids_train,ids_val = prepare_data(tr_file)
    score, lm = train_model(m_train,classes_train,m_val,classes_val,C,num_iter)
    print("SCORE:",score)
