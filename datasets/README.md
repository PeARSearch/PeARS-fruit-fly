# The dataset directory

This directory contains the datasets to be processed by the fruit fly. There are three datasets: Wikipedia, [Web
of Science](https://data.mendeley.com/datasets/9rw3vkcfy4/6) (medium-size version), and [20newsgroups](http://qwone.com/~jason/20Newsgroups/).

The script `prepare_datasets.py` download the datasets and split them to train, validation, and test parts.

In the default setting, the datasets contains meta tag `<id, class>` attached to each original document:

    python prepare_datasets.py

To get plain text (without meta tag), run:

    python prepare_datasets.py --no_meta

To get the datasets pre-processed by sentencepeice models, run one of the two commands:

    python prepare_datasets.py --spm=../spm/spmcc.model
    python prepare_datasets.py --spm=per_dataset

The first command will process all datasets with a single sentencepiece model, trained on Common Crawl data.
The second command processes each dataset individually, with a sentencepiece model issued from that dataset's training set.

You should use the second command when training fruit flies from the **budgeting** directory, and the first command otherwise.

The process may take a few minutes. After finishing, there will be three new folders created in the **datasets** directory,
containing the necessary data.

## How to re-create the Wikipedia dataset

If you are curious about how the Wikipedia dataset was created, please take a look at the **create_Wiki_dataset.ipynb** notebook.
It contains the procedures to create the ready-to-train Wikipedia dataset, which includes truncated articles from Wikipedia.

If you need the un-tokenized version, please download [the raw dataset](http://pearsproject.org/static/datasets/pears-fruit-fly-wikipedia-raw.zip).

