# PeARS Fruit Fly

It is widely believed that Web search engines require immense resources to operate, making it impossible for individuals to explore alternatives to the dominant information retrieval paradigms. The [PeARS project](https://pearsproject.org/) aims at changing this view by providing search tools that can be used by anyone to index and share Web content on specific topics. The focus is specifically on designing algorithms that will run on entry-level hardware, producing compact but semantically rich representations of Web documents. In this project, we use a cognitively-inspired algorithm to produce queryable representations of Web pages in a highly efficient and transparent manner. The proposed algorithm is a hashing function inspired by the olfactory system of the fruit fly, which has already been used in [other computer science applications](https://science.sciencemag.org/content/358/6364/793.abstract) and is recognised for its simplicity and high efficiency. We will implement and evaluate the algorithm on the task of document retrieval. It will then be integrated into a Web application aimed at supporting the growing practice of 'digital gardening', allowing users to research and categorise Web content related to their interests, without requiring access to centralised search engines.

This repository contains all code necessary to run and replicate our work. Note that the present README provides a minimal overview of the repository. Please browse the Wiki for extensive information about the framework and our experiments so far.

**We gratefully acknowledge financial support from [NLnet](https://nlnet.nl/) under the [NGI Zero programme](https://nlnet.nl/NGI0/).**

## Install

We recommend installing the code in a virtual environment (under Python3.6):

    virtualenv -p python3.6 PeARS-fruit-fly

Install requirements:

    cd PeARS-fruit-fly/
    source bin activate
    pip install -r requirements.txt

## Repository structure

This repository contains three directories, as described below. Each directory contains its own README, which contains further details on each aspect of the framework.

### Dataset

The datasets/ directory contains data for evaluating document vectors.

### CommonCrawl processor

The common-crawl-processor/ directory contains code for extracting and cleaning documents from [CommonCrawl dumps](https://commoncrawl.org/the-data/get-started/).

### Fruit-fly algorithm

The fruit-fly/ directory contains our implementation of the FFA for text classification. This will eventually include:

1. An implementation of a baseline system with Bayesian hyper-parameters search. \[available now\]

2. A genetic algorithm to improve the performance of the FFA. \[todo\]

3. A multi-layer FFA. \[todo\]


