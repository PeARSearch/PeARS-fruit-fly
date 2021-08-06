### Common-crawl-processor
It's a pipeline to deal with web documents from Common Crawl. It selects documents from a pre-determined language, pre-processes them and removes samples with unwanted content. 

### Getting raw text out of Common Crawl dumps

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
     
We take a sample from .wet processed documents in order to train the topic model, in our case a Latent Dirichlet allocation (LDA) model, that will detect unwanted content. We first need to preprocess the documents by removing highly and lowly frequent words, punctuation and numbers. We are using the Gensim library both for preprocessing and topic modelling. 

     python3 preprocess_gensim.py --folder=processed_wet --ndocs=70000 --pathdataset=gensim_data
     
Then we train our LDA model. To do that, run:

     python3 train_lda.py --pathdataset=gensim_data --outputfile=model_lda
     
Now we can have a look at the top k topics that have been assigned for our web documents, using specific terms to catch topics we might want to remove. For instance, the following returns documents with topics containing the term *var*. Those documents are pieces of JavaScript code which we probably want to discard in our final collection:

     python3 topk_lda.py --folder=processed_wet --pathdataset=gensim_data --pathmodel=model_lda --topk=3 --word=var

Now you're ready to use the whole pipeline with your own topics to be discarded in the 'pipeline' repository. Have fun!

     

