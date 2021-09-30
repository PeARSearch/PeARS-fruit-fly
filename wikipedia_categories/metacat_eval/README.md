# Wikipedia category extraction code (for development purposes)

This directory contains code to evaluate the spatial configuration of Wikipedia categories after hashing with the Fruit Fly Algorithm. This folder is for development purposes only. It contains scripts to parse a Wikipedia snapshot, extract categories from it, cluster them into 'meta-categories', and link external Web documents to each category. The resulting dataset can then be used for hashing and evaluating the hashed space.


## Download external links from wikipedia

We first need to extract links from a Wikipedia snapshot. To do this, we need the URLs corresponding to the many zipped files which, together, constitute the snapshot. An example Wikipedia dump can be found here: [https://dumps.wikimedia.org/enwiki/latest/](https://dumps.wikimedia.org/enwiki/latest/). Out of the many files listed on such page, we want to inspect the ones in the format *https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-...*, which contain the actual text of the document, including Web links. So our first task is to produce a list of such XML files, which we will save in a .txt file. You can use the following script to extract a list of wiki files for the latest dump:

    python3 create_wiki_dump_links.py 

This will create a file *wiki_dump_links.txt* in your directory, showing one wiki URL per line. 

Once we have all URLs for the Wikipedia dump, we will extract hyperlinks from the documents' content:

    python3 get_external_links.py 

This will take a while, as the Wikipedia dump is rather large. So get yourself a coffee! Your links will be saved in the *links* directory (one link file per Wikipedia snapshot file).

At this stage, the content of the extracted links should be scraped from the Web. The scraped content is saved in the *links* directory in files ending *.links.txt.gz*. The actual format of the file should be similar to that shown [here](https://github.com/PeARSearch/PeARS-fruit-fly/blob/main/fruit_fly/docs_labels_example.txt).



## Get category distribution

In order to output the number of documents that have been scraped by category, run:

    python3 wiki_cats.py --linksfolder=./links/ --function=0
    
The output of this execution can be found in the file 'distrib_categories.txt' in the newly created 'wiki_cats' folder. The first column represents the name of the category whereas the second shows the number of pages extracted for each category. 


## Create meta-categories
Given that Wikipedia categories can be overwhelmingly specific, we need to merge similar categories into a single 'meta-category'. For that, we check the frequency of ngrams first by running:

    python3 wiki_cats.py --linksfolder=./links/ --function=1    
    
Where the following message will appear and you can insert the ngram number you would like to check, for example 4:

    Insert 1 for unigrams, 2 for bigrams, 3 for trigrams, and so forth: 4

The output of the ngrams frequency can be found in './wiki_cats/4grams.txt'. You can run the same line of code again and choose a different ngram so that you have a better idea of frequency in which the ngrams appears in the categories name and how you would like to group them.

Finally, after having had a look at the files returned from the previous step, we can decide on a threshold, that is, on a mininum frequency that ngrams appear on the categories names. All ngrams higher than this threshold are grouped into one category. For that, run:

    python3 wiki_cats.py --linksfolder=./links/ --function=2
    
You are required to input the threshold value once the message below appears, in our example, 150:

    Insert minimum frequency of ngrams (threshold): 150
    
You can find the meta-categories and their respective categories in the file './wiki_cats/metacategories_topics.txt'. And the number of pages per meta-categories in './wiki_cats/distrib_metacategories.txt'. 

