"""Fruit fly classification

Usage:
  classify.py --file=<filename> --lr=<n> --batch=<n> --epochs=<n> --hidden=<n> --wdecay=<n> [--checkpoint=<dir>]
  classify.py (-h | --help)
  classify.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --filename    Name of training file
  --lr=<n>      Learning rate.
  --batch=<n>   Batch size.
  --hidden=<n>  Hidden layer size.
  --wdecay=<n>  Weight decay for Adam.
  --checkpoint=<dir>        Save best model to dir.

"""

import sys
import itertools
from docopt import docopt
from scipy.stats import spearmanr
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as td
import numpy as np
import random
import os
from math import sqrt
from sklearn.metrics import r2_score
from sklearn import preprocessing
import pickle

#print(device)
#random.seed(77)



class MLP(nn.Module):
    def __init__(self,d_in,hiddensize,d_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in,hiddensize)
        self.fc2 = nn.Linear(hiddensize,hiddensize)
        self.fc3 = nn.Linear(hiddensize,d_out)
        #self.drop1 = nn.Dropout(p=0.5)
    def forward(self, x):
        x1 = torch.relu(self.fc1(x[0]))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.softmax(self.fc3(x2), dim=1)
        return x3

def make_output(classes):
    classes = list(classes.values())
    outm = np.zeros((len(classes), len(set(classes)))) ###########################################
    single_classes = list(set(classes))
    for i in range(len(classes)):
        outm[i][single_classes.index(classes[i])] = 1
    return outm


def delete_checkpoints(checkpointsdir):
    filelist = os.listdir(checkpointsdir)
    for f in filelist:
        os.remove(os.path.join(checkpointsdir, f))

def validate(net,ids_val,m_val,classes_val,batch_size):
    #print("VALIDATION...............")
    #batch_size = 1
    val_predictions = []
    val_golds = []
    for i in range(0,len(ids_val), batch_size):
        prediction = net([Variable(torch.FloatTensor(m_val[ids_val[i:i+batch_size]]))])
        #predictions.append(prediction.data.numpy()[0])
        predictions = prediction.data.cpu().numpy()

        for j in range(predictions.shape[0]):
            val_predictions.append(np.argmax(predictions[j]))
            val_golds.append(np.argmax(classes_val[ids_val[i+j]]))

    val_pred_score = sum(1 for x,y in zip(val_predictions,val_golds) if x == y) / len(val_golds)

    #print("VAL PRED SCORE:",val_pred_score)
    return val_pred_score

def train_model(m_train,classes_train,m_val,classes_val,ids_train,ids_val,hiddensize,lrate,wdecay,batchsize,epochs,checkpointsdir):
    '''Initialise network'''
    net = MLP(m_train.shape[1],hiddensize,classes_train.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=lrate, weight_decay=wdecay)
    criterion = nn.MSELoss()

    total_loss = 0.0
    c=0
    batch_size = batchsize
    validation_scores = []
    current_max_score = -10000

    for epoch in range(epochs):

        #print("TRAINING...............")
        #print("Epoch {}".format(epoch))
        random.shuffle(ids_train)
        #for i in ids_train:
        for i in range(0,len(ids_train), batch_size):
            X, Y = m_train[ids_train[i:i+batch_size]], classes_train[ids_train[i:i+batch_size]]
            X, Y = Variable(torch.FloatTensor(X), requires_grad=True), Variable(torch.FloatTensor(Y), requires_grad=False)
            net.zero_grad()
            output = net([X])
            #loss = criterion(output, Y, torch.Tensor([1]))
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            c+=1
            #print(total_loss / c)

        validation_score = validate(net,ids_val,m_val,classes_val,batchsize)
        validation_scores.append(validation_score)
        if np.argmax(validation_scores) == epoch and checkpointsdir != "":
            print("Saving checkpoint...")
            delete_checkpoints(checkpointsdir)
            torch.save(net, os.path.join(checkpointsdir,"e"+str(epoch)))
        #print(validation_scores)
        #print(current_max_score)
        if epoch % 100 == 0:
            if validation_scores[-1] > current_max_score:
                current_max_score = validation_scores[-1]
            else:
                print("Early stopping...")
                break

    print("MAX:", np.argmax(validation_scores), np.max(validation_scores))
    return np.max(validation_scores)

def prepare_data(tr_file):
    dev_file = tr_file.replace("train","test")

    print("Reading dataset...")
    m_train = pickle.load(open(tr_file,'rb')).todense()
    #print(m_train)
    #m_train = preprocessing.normalize(m_train, norm='l1')
    m_val = pickle.load(open(dev_file,'rb')).todense()
    #print(m_val)
    #m_val = preprocessing.normalize(m_val, norm='l1')

    classes_train = make_output(pickle.load(open(tr_file.replace("hs","cls"),'rb')))
    classes_val = make_output(pickle.load(open(dev_file.replace("hs","cls"),'rb')))

    ids_train = np.array([i for i in range(m_train.shape[0])])
    ids_val = np.array([i for i in range(m_val.shape[0])])

    return m_train,classes_train,m_val,classes_val,ids_train,ids_val

if __name__ == '__main__':
    checkpointsdir = ""
    args = docopt(__doc__, version='Ideal Words 0.1')
    if args["--checkpoint"]:
        checkpointsdir = args["--checkpoint"]
    tr_file = args["--file"]
    lrate = float(args["--lr"])
    batchsize = int(args["--batch"])
    epochs = int(args["--epochs"])
    hiddensize = int(args["--hidden"])
    wdecay = float(args["--wdecay"])

    if checkpointsdir != "":
        delete_checkpoints(checkpointsdir)
    m_train,classes_train,m_val,classes_val,ids_train,ids_val = prepare_data(tr_file)
    train_model(m_train,classes_train,m_val,classes_val,ids_train,ids_val,hiddensize,lrate,wdecay,batchsize,epochs,checkpointsdir)
