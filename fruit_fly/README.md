# A Fruit Fly algorithm (FFA) for text classification

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
  
    python hyperparam_search.py --train_path=../datasets/20news-bydate/20news-bydate-train.sp --continue_log=./log/logs_20news.json 

will let the Bayesian optimization runs for 50 times, each time the classification will run 5 times and average the
validation scores. The **--continue_log** argument, which is *optional*, takes the history of the Bayesian optimization and
continue to optimize. If you run for the first time, ignore this argument.

Web of Science dataset:

    python hyperparam_search.py --train_path=../datasets/wos/wos11967-train.sp

Wikipedia dataset:

    python hyperparam_search.py --train_path=../datasets/wikipedia/wikipedia-train.sp

The validation scores and the combinations of hyper-parameters are stored in the log folder.

## Test the best hyper-parameters on test sets

Manually creating the best hyper-parameter settings in **models/best_models**. Please take a look in this folder for
examples. Run:

    python test_models.py --test_path=../datasets/wos/wos11967-test.sp --config_path=./models/best_models/wos_config.json

to get the average score on **--test_path** for all the hyper-parameter settings in **--config_path**.

## Results

If you want to see how the FFA is doing without running the algorithm yourself, check out our results in the Wiki [here](https://github.com/PeARSearch/PeARS-fruit-fly/wiki/1.1.-Baselines).


# An evolutionary algorithm for the FFA

Having set up our basic FFA, we now turn to the problem of optimizing the projections inside the fly. We recall that each random projection in the architecture can be taken as capturing one semantic dimension of the document to classify. Thus, there should be better and worse projection sets, i.e. more or less discriminative ones. We propose the use of a genetic algorithm to find a fly individual with an ideal set of projections. For more information, see the Wiki page [here](https://github.com/PeARSearch/PeARS-fruit-fly/wiki/1.2-A-Genetic-Algorithm-for-optimizing-FFA).

## Running the evolutionary process

The evolution process helps to find the projection matrix and the winner-take-all rate that satisfy a pre-defined goal. Our goal is to find a fly that performs well on the classification task (on all 3 datasets), and also has a small number of non-zero elements in the final hash (i.e. is compact and efficient).

The evolution process can be run using the following script:

    python -W ignore evolve_flies.py 

Running the above will generate a json file in *models/evolution* containing the hyperparameters used in the evolutionary process, as well as performance information for each generation of flies. As the evolutionary process takes place, the best individual flies for each criterion are saved (best overall fitness, best average validation accuracies on 3 datasets, lowest number of non-zero elements, best validation accuracies on each datasets) in the same directory. It is possible to print high-level results for the evolution process by running:

    python print_best_flies.py

This will return a summary of the best flies obtained for each criterion.



## Running the best flies on the test sets

The evolution process evaluates flies on the development sets of our three datasets. Once the best individuals have been identified, we can check their performance on the test data. To do this for a selected fly, run:

    python test_evolution_model.py --fly=<selected fly>

So for instance, to test the fly with the overall best fitness:

    python test_evolution_model.py --fly=models/evolution/best_fitness


## Use a fly to hash documents

Finally, once the best fly has been selected, we can use it to hash documents. The code expects a file with text documents associated with a particular label such as our example file named 'docs_labels_example.txt' (You can use your './wiki_cats/n_metacategories.txt' document if you have already executed 'prepare_dataset_wiki.py' in 'wikipedia_categories'). Then, run the code below with the selected fly from the evolution process and find the output in the 'hashes' folder.

    python hash_with_best_proj.py --fly=<selected fly> --docfile=docs_labels_example.txt

The outputs are pickle files containing the hash (.hs), the keywords (.kwords), id (.ids), label (.cls) and finally, url (.url) of each document. The information from each web document is grouped by label, meaning that each pickle file contains a given type of information (e.g. hash, url,...) of all documents belonging to the same label. 
