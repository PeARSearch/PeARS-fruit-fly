# The dense fruit fly

This folder contains code to generate 'dense' fruit flies, i.e. fruit flies with a small number of KCs, issued from a dimensionality-reduced UMAP model. These fruit flies are trained on the prec@k evaluation: given one document in the val set, i.e. the fraction of the first *k* nearest neighbours that belong the same class as the document. 

To train a UMAP model, do:

    umap_search.py --dataset=<wiki|20news|wos> 

Once the best UMAP model has been found and saved, we can proceed to the training of the fruit fly, doing:

    fly_search.py --dataset=<wiki|20news|wos> --logprob=<n>

The logprob figure should be the one returned by *umap_search.py* in the best set of hyperparameters. NB: ideally we should save this in the fly itself. TODO.

Finally, the best saved fly can be evaluated on the test set, using both the prec@k evaluation and a classification task:

    test_fly.py --dataset=<wiki|20news|wos> --logprob=<n>

(Same comment here about the logprob value.)
