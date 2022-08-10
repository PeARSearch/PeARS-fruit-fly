# The dense fruit fly

## Dense fruit fly with Uniform Manifold Approximation and Projection for Dimension Reduction (UMAP)

This folder contains code to generate 'dense' fruit flies, i.e. fruit flies with a small number of KCs, issued from a dimensionality-reduced UMAP model. These fruit flies are trained on the prec@k evaluation: given one document in the val set, i.e. the fraction of the first *k* nearest neighbours that belong the same class as the document. 

To train a UMAP model, do:

    umap_search.py --dataset=<wiki|20news|wos> 

Once the best UMAP model has been found and saved, we can proceed to the training of the fruit fly, doing:

    fly_search.py --dataset=<wiki|20news|wos> --logprob=<n>

The logprob figure should be the one returned by *umap_search.py* in the best set of hyperparameters. NB: ideally we should save this in the fly itself. TODO.

Finally, the best saved fly can be evaluated on the test set, using both the prec@k evaluation and a classification task:

    test_fly.py --dataset=<wiki|20news|wos> --logprob=<n>

(Same comment here about the logprob value.)


## Testing different methods of initialization the projection matrix

Please read the wiki [section 1.3](https://github.com/PeARSearch/PeARS-fruit-fly/wiki/1.3.-Initialization,-pre-and-post-processing)
 for the description of the task.

To run the initialization test:

    test_init.py --dataset=<wiki|20news|wos|reuters|tmc|agnews> 

The code will print the score of classification and pre@100 for 64 and 128 dimensions setting.
Each line of number respectively corresponds to the initialization method mentioned in the result table
in the wiki page above.

## Test different pre and post processing methods to reduce the number of KC

Please read the wiki [section 1.3](https://github.com/PeARSearch/PeARS-fruit-fly/wiki/1.3.-Initialization,-pre-and-post-processing)
 for the description of the task.

To run the pre and post processing test:

    test_pre_post_processing.py --dataset=<wiki|20news|wos|reuters|tmc|agnews> 

The code will print the matrix of scores for classification and pre@100 for 64 and 128 dimensions setting, with the first column
is for 64 dimensions, the second one is for 128 dimensions. The row corresponds to the methods in order:
10000 dimensions without any processing steps, 64/128 dimensions without any processing steps, applying PCA before projection step,
applying PCA after the projection step, applying PCA both before and after projection steps, pruning KC by removing nodes with high 
frequency of zero, or low standard deviation, or low entropy.
