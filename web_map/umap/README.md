# A map of knowledge - dimensionality reduced implementation

This folder contains code to process the whole of Wikipedia in any language of interest, to generate a 'topical map' of encyclopedic knowledge. This is a specific implementation of the idea of a 'knowledge map', explained [here](https://github.com/PeARSearch/PeARS-fruit-fly/tree/main/web_map). Implementation specifics are as follows:

* the code can be run for any language with a Wikipedia;
* it implements dimensionality reduction prior to feeding document representations to the fruit fly;
* the reduced representations are turned into binary vectors via the fly;
* evaluation is performed using *precision @ k*, a measure that computes whether the *k* nearest neighbours of a target document belong to the same class as the target;
* classes for evaluation are provided by prior clustering over the reduced representations (but prior to binarization).


**Warning:** Processing a Wikipedia dump takes a long time. Be prepared for it!



## Downloading and pre-processing Wikipedia

We provide a script to download and pre-process an entire Wikipedia dump, using a trained sentencepiece model. (One is provided for the English in the top directory of this repository, at *spm/spm.wiki.*.) As long as an appropriate sentencepiece model is available, the script can be run for any language, using the following command:

    python3 get_wiki_data.py --lang=<str> --spm=<path>

where the argument of *--lang* is the desired language (e.g. *en* for English, *ml* for Malayalam, etc). 


## Applying dimensionality reduction and fruit fly to the Wiki corpus

We use UMAP for dimensionality reduction and Birch for clustering. The first thing we will have to do is train a UMAP and Birch model from one subset of Wikipedia. This can be done with the following command:

    python3 apply_umap_fly.py train --dataset=processed/enwiki-latest-pages-articles1.xml-p1p41242.sp

(Here, we are training on the first file of the dump. This is usually a good choice, as older articles cover a range of fundamental topics.)

Next, we will dimensionality-reduce and cluster the Wikipedia data, file by file, using the models we have trained: 

    python3 apply_umap_fly.py reduce --model=models/umap/enwiki-latest-pages-articles1.xml-p1p41242.umap

(The script figures out the path for the Birch model from the UMAP path.)

If desired, it is possible to get an interpretable representation of the UMAP clusters using:

    python3 apply_umap_fly.py label

This will gather documents from all dump files, together with their respective cluster IDs, and derive keywords from them to describe each cluster.

The next and final step is to put the UMAP representations through the fruit fly. To do this, run e.g.:

    python3 apply_umap_fly.py fly --dataset=processed/enwiki-latest-pages-articles1.xml-p1p41242.sp --model=models/umap/enwiki-latest-pages-articles1.xml-p1p41242.umap

where the argument of *--dataset* is the dump file the fly should be trained on (probably again the first dump file), and the argument of *--model* is the path to the previously trained UMAP model. In principle, it is not necessary to train the fly on the file that UMAP was trained on, but it makes good sense. 
