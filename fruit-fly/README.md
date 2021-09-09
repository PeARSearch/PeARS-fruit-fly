# Fruit Fly algorithm (FFA) for text classification

Before running the code below, make sure to have downloaded the necessary datasets. More information is available from the *datasets* directory in the main repository.

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

in which **C** is the the inverse of regularization term in logistic regression, **num_iter** is the number of iteration
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

### Test the best hyper-parameters on test sets

Manually creating the best hyper-parameter settings in **models/best_models**. Please take a look in this folder for
examples. Run:

    python test_models.py --test_path=../datasets/wos/wos11967-test.sp --config_path=./models/best_models/wos_config.json

to get the average score on **--test_path** for all the hyper-parameter settings in **--config_path**.

## Run the evolution process

The evolution process helps to find the projection matrix and the winner-take-all rate that satisfy a pre-defined goal.
So far, the goal is searching for a fly that performs well on classification task (on both 3 datasets), and it also
has small number of non-zero elements in the hashing results.

To run the evolution process:

    python -W ignore evolve_flies.py 

The hyper-parameters, as well as the information in each generation can be found in a json file in *models/evolution*.
The best flies for each criterion (best overall fitness, best average validation accuracies on 3 datasets, lowest number
of non-zero elements, best validation accuracies on each datasets) can be found in the same directory.
