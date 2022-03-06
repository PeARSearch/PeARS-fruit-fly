# The dense fruit fly

This folder contains code to generate 'dense' fruit flies, i.e. fruit flies with a high WTA which will make them suitable for similarity calculations.

To run a hyperparameter search, do:

    hyperparam_search.py --dataset=<str> [--continue_log=<filename>] (--random|--store) (--classification|--similarity)

For example:

    python3 hyperparam_search.py --dataset=wos --random --similarity

will run the optimization for the Web of Science dataset, with flies initialized randomly and evaluated using similarity.

When using the --store flag, flies will be created using pre-computed projections from the directory *projection_store* in the root folder. Make sure that the pbounds parameters set for the Bayesian Optimization use a range of projection sizes for which files exist in *projection_store*. (The pbounds have to be set manually inside *hyperparam_search.py*.
