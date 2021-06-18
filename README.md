# PeARS Fruit Fly

This repository integrates insights from the Fruit Fly Algorithm (FFA) in the PeARS framework. Browse the *references.txt* file for references to relevant scientific publications. There is also a little practical exercise on using the FFA [here](https://github.com/ml-for-nlp/fruit-fly).

## Install

We recommend installing the code in a virtual environment (under Python3.6):

    virtualenv -p python3.6 PeARS-fruit-fly

Install requirements:

    cd PeARS-fruit-fly/
    source bin activate
    pip install -r requirements.txt

## Dataset

See README in the dataset/ directory for details of the dataset to be downloaded.


## Splitting dataset into wordpieces

The train and test sections of the 20_newsgroup dataset will first be converted into wordpieces:

    python3 wordpiece.py

## Creating random projections for the fruit fly

The random projections used by the fly will be created and saved in a file named spmcc.projs in a directory under models/. For instance, running

    python3 mkprojections.py --kc=2000 --size=10

would create random projections of size 10, going into a Kenyon Cell layer of 2000 nodes, saved under *models/kc2000-p10/spmcc.projs*.

## Compute hashes on the train and test sets

We will now compute document hashes with our random projections. Here is an example usage.

    python3 hash.py --file=datasets/20news-bydate/20news-bydate-train.sp --dir=models/kc2000-p10/ --topwords=100 --wta=10

## Train/test a network to classify documents

We now train a simple neural net to classify documents on the training data, and evaluate on the test set. Here s an example usage:

    python3 classify.py --file=models/kc2000-p10/20news-bydate-train.top100.wta10.hs --lr=0.0002 --batch=2048 --epochs=1000 --hidden=100 --wdecay=0.0001
