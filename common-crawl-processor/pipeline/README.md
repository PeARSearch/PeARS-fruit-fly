## Common-crawl-processor pipeline

If we are processing billions of documents, we need to save some space by avoiding saving documents over and over again. The purpose of this pipeline is to remove innapropriate content before saving our relevant corpus. 

### Executing the pipeline

In order to run the pipeline, you will have needed to go through the steps presents in the common-crawl-processor folder so that all the necessary models and dictionaries are saved into your repository. 

This step requires some manual work because you will need to choose the thresholds and the topics you want to exclude. You can replace our example in the tops2remove.txt file and add your own according to your analysis. You just need to add the index of the topic that you find in the topics.txt file. Then, you're ready to run:

    python3 filter_documents.py --folder=processed_wet --pathmodel=model_lda --pathdataset=gensim_lda
    
You can process as many documents as you like until you reach a corpus size that suits you, just hit Ctrl+C when you want to stop the code. 
    
