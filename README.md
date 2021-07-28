# PeARS Fruit Fly

This repository integrates insights from the Fruit Fly Algorithm (FFA) in the PeARS framework. Browse the wiki for references to relevant scientific publications. There is also a little practical exercise on using the FFA [here](https://github.com/ml-for-nlp/fruit-fly).

## Install

We recommend installing the code in a virtual environment (under Python3.6):

    virtualenv -p python3.6 PeARS-fruit-fly

Install requirements:

    cd PeARS-fruit-fly/
    source bin activate
    pip install -r requirements.txt

## Repository structure

This repository contains three directories, as described below.

### Dataset

See README in the dataset/ directory for details of the dataset to be downloaded.

### CommonCrawl processor

Code for cleaning the CommonCrawl downloads, and clustering. Please read the README in the sub folder.

### Fruit-fly algorithm

Implementation of the FFA for text classification.

1. Implementation of a baseline with Bayesian hyper-parameters search.

2. A genetic algorithm to improve the performance of the FFA.

3. A multi-layers FFA to release.

Please read the README in the sub folder.
