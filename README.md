# PeARS Fruit Fly

## Install

We recommend installing the code in a virtual environment (under Python3.6):

    `virtualenv -p python3.6 PeARS-fruit-fly`

Install requirements:

    `cd PeARS-fruit-fly/`
    `source bin activate`
    `pip install -r requirements.txt`

## Dataset

See README in the dataset/ directory for details of the dataset to be downloaded.


## Splitting dataset into wordpieces

The train and test sections of the 20_newsgroup dataset will first be converted into wordpieces:

    `python3 wordpiece.py`
