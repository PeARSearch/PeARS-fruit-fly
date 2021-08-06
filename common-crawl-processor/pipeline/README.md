## Common-crawl-processor pipeline

If we are processing billions of documents, we need to save some space by avoiding saving documents over and over again. The purpose of this pipeline is to remove innapropriate content and only save the relevant documents in our corpus. 

### Executing the pipeline

In order to run the pipeline, you will have needed to go through the steps presents in the common-crawl-processor folder so that all the necessary models and dictionaries are saved into your repository. 

This step requires some manual work because you will need to choose the thresholds and the topics you want to exclude. You can replace our example in the topics_threshold.txt file and add your own according to your analysis. On the first columns there are the indexes of the topics returned from the model and on the second column the corresponding probability threshold. You can find the indexes of the topics from your model in the topics.txt file. 

Then, you're ready to run:

    python3 filter_documents.py --folder=processed_wet --pathmodel=model_lda --pathdataset=gensim_lda
    
You can process as many documents as you like (or as many locations as you have) until you reach a corpus size that suits you, just hit Ctrl+C when you want to stop the code. 
    
