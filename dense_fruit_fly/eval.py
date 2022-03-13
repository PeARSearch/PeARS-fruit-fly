import numpy as np
from sklearn.metrics import pairwise_distances

def compute_nearest_neighbours(vecs,labels,i,num_nns):
    i_sim = np.array(vecs[i])
    i_label = labels[i]
    ranking = np.argsort(-i_sim)
    neighbours = [labels[n] for n in ranking][1:num_nns+1] #don't count first neighbour which is itself
    score = sum([1 if n == i_label else 0 for n in neighbours]) / num_nns
    #print(i,i_label,neighbours,score)
    return score,neighbours

def prec_at_k(m=None,classes=None,k=None,metric="cosine"):
    vecs = 1-pairwise_distances(m.todense(), metric=metric)
    scores = []
    for i in range(vecs.shape[0]):
        score, neighbours = compute_nearest_neighbours(vecs,classes,i,k)
        scores.append(score)
    return np.mean(scores)

