"""Common Crawl processor - retrieve the top k topics associated with the documents

Usage:
  topk_octis.py --pathdataset=<foldername> --foldertxt=<foldername> --topk=<highestktopics> --lda_model=<filename> --word=<word>
  topk_octis.py (-h | --help)
  topk_octis.py --version


Options:
  -h --help     Show this screen.
  --version     Show version.
  --pathdataset=<foldername>    Name of the folder where the preprocessed data is located  
  --foldertxt=<foldername>     Name of the folder where the zipped .xml files are located
  --topk=<highestktopics>       Number of topics with the highest topics associated with documents
  --lda_model=<filename>        Name of file where the LDA model was saved previously
  --word=<word>                 Word to be focused in your search of topics

"""

from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA
import numpy as np
import csv
import sys
from docopt import docopt
csv.field_size_limit(sys.maxsize)

def print_docs_topics(indexes, web_docs, k, model_output, word):

    for doc, idx in zip(model_output["topic-document-matrix"].T, indexes):
        topk_ind = np.argpartition(doc, -k)[-k:]
        tops = [tp for tp in topk_ind]  #if doc[tp] > 0.60

        list_topics = [model_output['topics'][tp] for tp in tops]
        topics = [item for sublist in list_topics for item in sublist] 
        if word in topics:
            print(tops) # print the topk indices
            print(doc[tops]) #print the corresponding values
            print(list_topics)
            print("index doc in the original file:", idx)
            print(web_docs[idx])


if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Processor')

    pathdataset = args['--pathdataset']
    foldertxt = args['--foldertxt']
    k = int(args['--topk'])
    lda_model= args['--lda_model']
    word = args['--word']

    model_output=np.load('./'+pathdataset+"/"+lda_model+'.npz')

    indexes=[]
    f_index = open("./"+pathdataset+"/indexes.txt", 'r')
    for line in f_index.read().splitlines():
        indexes.append(int(line))

    web_docs = []
    f_txt = open("./"+foldertxt+"/docs_octis.txt", 'r')
    for line in f_txt.read().splitlines():
        web_docs.append(line)

    print_docs_topics(indexes, web_docs, k, model_output, word)





