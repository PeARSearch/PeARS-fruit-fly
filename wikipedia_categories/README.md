### Download external links from wikipedia
...

### Understand the number of pages downloaded per category
We want to understand the number of documents that have been downloaded by category. For that, we run:

    python3 wiki_cats.py --linksfolder=./links/ --function=0
    
The output of this execution can be found in the file 'distrib_categories.txt' in the newly created 'wiki_cats' folder. The first column represents the name of the category whereas the second shows the number of pages extracted for each category. 

### Create meta-categories
Given that Wikipedia categories can be overwhelmingly specific, we need to merge similar categories into a single meta-category, as we call it here. For that, we check the frequency of ngrams first by running:

    python3 wiki_cats.py --linksfolder=./links/ --function=1    
    
Where the following message will appear and you can insert the ngram number you would like to check, for example 4:

    Insert 1 for unigrams, 2 for bigrams, 3 for trigrams, and so forth: 4

The output of the ngrams frequency can be found in './wiki_cats/4grams.txt'. You can run the same line of code again and choose a different ngram so that you have a better idea of frequency in which the ngrams appears in the categories name and how you would like to group them.

Finally, after having had a look at the files returned from the previous step, we can decide on a threshold, that is, on a mininum frequency that ngrams appear on the categories names. All ngrams higher than this threshold are grouped into one category. For that, run:

    python3 wiki_cats.py --linksfolder=./links/ --function=2
    
You are required to input the threshold value once the message below appears, in our example, 150:

    Insert minumum frequency of ngrams (threshold): 150
    
You can find the meta-categories and their respective categories in the file './wiki_cats/metacategories_topics.txt'. And the number of pages per meta-categories in './wiki_cats/distrib_metacategories.txt'. 

### Create a dataset for the classification

We are going to use the webpages extracted from Wikipedia's external links in a classification task, therefore, we also need to make sure the meta-categories form semantically meaningful groups. That is why we need to go through the 'distrib_metacategories.txt' manually and identify meta-categories that should be discarded. You can add each one of them per line in the file called 'metacats_to_discard.txt' so that the next step will disconsider those categories for the dataset preparation. 

After that, we can decide on the number of meta-categories we want to keep for our classification task and the number of documents we would like to keep per meta-category so that we obtain a more balanced dataset. For example, we choose to keep 180 categories and 2000 documents. Then, we can run:

    python3 prepare_dataset_wiki.py --linksfolder=./links/ --num_docs=2000 --num_metacats=180

The processed datasets can be found in '.wiki_cats'. 

### Classification

We are going to use the fly that evolved to be the best in our evolutionary process. 