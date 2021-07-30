# Fruit Fly algorithm (FFA) for text classification

## Run the FFA step by step on one dataset

1. Creating random projections for the fruit fly

The random projections used by the fly will be created and saved in a file named spmcc.projs in a directory under models/. For instance, running

    python mkprojections.py --kc=2000 --size=10

would create random projections of size 10, going into a Kenyon Cell layer of 2000 nodes, saved under *models/kc2000-p10/spmcc_0.projs*.

2. Compute hashes on the train and val sets

We will now compute document hashes with our random projections. Here is an example usage.

    python hash.py --file=../datasets/20news-bydate/20news-bydate-train.sp --dir=models/projection/kc2000-p10/spmcc_0.projs --topwords=100 --wta=10
    python hash.py --file=../datasets/20news-bydate/20news-bydate-val.sp --dir=models/projection/kc2000-p10/spmcc_0.projs --topwords=100 --wta=10

3. Train/test a network to classify documents

We now train a simple logistic regression model to classify documents on the training data, and evaluate on the test set. Here s an example usage:

    python classify.py --file=tmp/20news-bydate-train.kc2000.size10.trial0.top100.wta10.hs --C=10 --num_iter=1000

in which **C** is the the inverse of regularization term in logistic regression, *num_iter* is the number of iteration
in the optimization process.

## Run hyper-parameter search on three datasets

20newsgroups dataset:
  
    python hyperparam_search.py --train_path=../datasets/20news-bydate/20news-bydate-train.sp --continue_log=./log/logs_20newsgroups.json 

will let the Bayesian optimization runs for 50 times, each time the classification will run 5 times and average the
validation scores. The **--continue_log** argument, which is *optional*, takes the history of the Bayesian optimization and
continue to optimize. If you run for the first time, ignore this argument.

Web of Science dataset:

    python hyperparam_search.py --train_path=../datasets/wos/wos11967-train.sp

Wikipedia dataset:

    python hyperparam_search.py --train_path=../datasets/wikipedia/wikipedia-train.sp

The validation scores and the combinations of hyper-parameters are stored in the log folder.