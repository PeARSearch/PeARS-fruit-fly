### Common-crawl-processor
It's a pipeline to deal with web documents from Common Crawl. It selects documents from a pre-determined language, pre-processes them and removes samples with unwanted content. 

### Getting raw text out of Common Crawl dumps

We will be using the .wet files from Common Crawl. For more information on the WET format, please consult [...].

Note that processing Common Crawl files is a very intensive job. Please refer to the information we have compiled about benchmarking (here in the wiki) before launching your own jobs. At the same time, don't be shy: you can process small amounts of data on your laptop without problems. So give it a go, and find friends to collectively process *more* data!

Before you start, you will have to find the location of some .wet files to process. If you go to the Common Crawl website and look for monthly file listings, for instance [here](https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-50/index.html), you will find files named *wet.paths.gz*. If you uncompress one of those *wet.paths* file, you will get a list of URLs starting with *crawl-data...* Prepend *https://commoncrawl.s3.amazonaws.com/* to each line, and you will get a few tens of thousands of .wet files' URLs.


## Using the code

We recommend using a virtual environment. You can set it up from outside your clone repository, by doing:

     virtualenv common-crawl-processor

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
    
We are going to use the [OCTIS library](https://github.com/MIND-Lab/OCTIS) to remove unwanted content. First, we need to transform the .xml files into .txt.

     python3 transform_into_txt.py --foldertxt=processed_wet
     
We take a sample from .wet processed documents in order to train the topic model, in our case a Latent Dirichlet allocation (LDA) model, that will detect unwanted content. 

     python3 preprocess_octis.py --foldertxt=processed_wet --ndocs=60000 --pathdataset=octis
     
Then, we are ready to run the LDA model. As explained in the documentation of OCTIS, the output is a dictionary with:<br/>
..*topics: the list of the most significative words for each topic (list of lists of strings).<br/>
..*topic-word-matrix: an NxV matrix of weights where N is the number of topics and V is the vocabulary length.<br/>
..*topic-document-matrix: an NxD matrix of weights where N is the number of topics and D is the number of documents in the corpus.<br/>
The code below saves this dictionary into a .npz file.

     python3 lda_octis.py --pathdataset=octis --outputfile=lda_model
     
Now we can have a look at the top k topics that have been assigned for our web documents. We can focus on one word in particular to see how well the model assigns topics to the documents. 

     python3 topk_octis.py --pathdataset=test --foldertxt=processed_wet --topk=3 --lda_model=lda_model --word=computer



     

