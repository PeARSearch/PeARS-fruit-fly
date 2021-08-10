# The dataset directory

This directory contains the datasets to be processed by the fruit fly. There are three datasets: Wikipedia, [Web
of Science](https://data.mendeley.com/datasets/9rw3vkcfy4/6) (medium-size version), and [20newsgroups](http://qwone.com/~jason/20Newsgroups/).

Simply run this command to download the three datasets, and pre-process them for training:

    python prepare_datasets.py

The process may take a few minutes. After finishing, there will be three new folders created in the **datasets** directory, containing the necessary data.

## How to re-create the Wikipedia dataset

If you are curious about how the Wikipedia dataset was created, please take a look at the **create_Wiki_dataset.ipynb** notebook.
It contains the procedures to create the ready-to-train Wikipedia dataset, which includes truncated articles from Wikipedia.

If you need the un-tokenized version, please download [the raw dataset](http://pearsproject.org/static/datasets/pears-fruit-fly-wikipedia-raw.zip).

