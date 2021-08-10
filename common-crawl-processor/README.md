# Common-crawl-processor
It's a pipeline to deal with web documents from Common Crawl. It selects documents from a pre-determined language, pre-processes them and removes samples with unwanted content. 

## Getting raw text out of Common Crawl dumps

We will be using the .wet files from Common Crawl. For more information on the WET format, please consult [https://skeptric.com/text-meta-data-commoncrawl/](https://skeptric.com/text-meta-data-commoncrawl/).

Note that processing Common Crawl files is a very intensive job. At the same time, don't be shy: you can process small amounts of data on your laptop without problems. So give it a go, and find friends to collectively process *more* data!

Before you start, you will have to find the location of some .wet files to process. If you go to the Common Crawl website and look for monthly file listings, for instance [here](https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-50/index.html), you will find files named *wet.paths.gz*. If you uncompress one of those *wet.paths* file, you will get a list of URLs starting with *crawl-data...* Prepend *https://commoncrawl.s3.amazonaws.com/* to each line, and you will get a few tens of thousands of .wet files' URLs.

For the sake of example, there is one such .wet URL in *example-path-file.txt*, in this directory.

## Using the code


To process raw .wet files, do:

    python cc_process_wet.py --file=example-path-file.txt --lang=en
    
You should see the file being processed:

    1 documents processed 0 documents added...
    3 documents processed 0 documents added...
    383 documents processed 100 documents added...
    384 documents processed 100 documents added...
    385 documents processed 100 documents added...
    387 documents processed 100 documents added...
    768 documents processed 200 documents added...
    
We are going to use a topic modelling approach to remove unwanted content. First, we need to transform the .xml files into .txt.

     python3 transform_into_txt.py --folder=processed_wet
     
We take a sample from .wet processed documents in order to train the topic model, in our case a Latent Dirichlet allocation (LDA) model, that will detect unwanted content. LDA is a technique that assigns so-called 'topics' to documents, where each topic is expressed as a collection of characteristic words. For instance, the following might be a topic about fencing and gating:

     iron, wrought, ca, gates, fence, company, fencing, gate, ornamental, contractor

LDA probabilistically assigns topics to documents. So for instance, the Web page of a fence manufacturer might have a 0.6 probability of including the fencing topic, and a 0.4 probability of including a shopping topic.

We first need to preprocess the documents by removing highly and lowly frequent words, punctuation and numbers. We are using the Gensim library both for preprocessing and topic modelling. The vocabulary for the LDA model has already been created and it's available in 'vocabulary.txt'. The command below creates a bag of words for each document. 

     python3 preprocess_gensim.py --folder=processed_wet --ndocs=2000 --lda_path=gensim_lda
     
Then we train our LDA model. To do that, run:

     python3 train_lda.py --lda_path=gensim_lda --model_out=model_lda

This will take a few minutes for 2000 documents (more if you train over a larger set). So go and get a cup of coffee. For best performance, we recommend training over a few tens of thousands of documents.

The 100 topics created by the LDA model are visible at *gensim_lda/topics_lda.txt*, together with an ID number.

Now we can have a look at the top k topics that have been assigned for our web documents, using specific terms to catch topics we might want to remove. For instance, the following returns documents with topics containing the term *var*. Those documents are pieces of JavaScript code which we probably want to discard in our final collection. 

     python3 topk_lda.py --folder=processed_wet --lda_path=gensim_lda --model=model_lda --topk=3 --word=var

The complete output of the script can be found in a .csv file. For our example, *topkprob_var.csv* should show you the documents that were labeled with a topic containing the word *var*. 

Now we are ready to filter your Common Crawl corpus with your topics of choice to be discarded. 

## Filtering documents

The purpose of the next steps are to remove inappropriate content and only save the relevant documents in our corpus. This step requires some manual work because we will need to choose some probability thresholds on the topics you want to exclude. You can replace our example in the *topics_threshold.txt* file and add your own according to your analysis. The *topics_threshold* file looks like this:

     10 0.1
     59 0.2
     ...

The first column shows the indices of the topics to be discarded (as given in *gensim_lda/topics_lda.txt*), and the second column the corresponding probability thresholds. The line *10 0.1* indicates that we want to discard any document which has a probability of at least 0.1 of containing the topic with ID 10. 

Once we have chosen topics and thresholds, we can run the following:

    python3 filter_documents.py --folder=processed_wet --model=model_lda --lda_path=gensim_lda --keep_discarded=True
    
The output of the code is a json file with a dictionary per line, each dictionary contains the keys 'doc', 'title', 'url' and 'lang' of each document kept in the preprocessing. The files are named *kept_n.json* and are located in the newly created folder *./corpus*. Setting *--keep_discarded* to *True*, will ensure that the discarded documents are saved as well in a separate json file named *discarded_n.json*.
    
You can process as many documents as you like (or as many locations as you have) until you reach a corpus size that suits you, just hit Ctrl+C when you want to stop the code. 
    
