import sys
import csv
csv.field_size_limit(sys.maxsize)
from gensim.corpora import Dictionary

def preprocess(doc, dictionary, tokenizer):
    doc=doc.lower()  # Convert to lowercase.
    doc=tokenizer.tokenize(doc)  # Split into words.

    doc = [token for token in doc if not token.isnumeric()]
    doc = [token for token in doc if len(token) > 1]

    corpus = dictionary.doc2bow(doc)
    return corpus

def classify_removal(doc, dictionary, lda, tokenizer, topics):
    corpus=preprocess(doc, dictionary, tokenizer)
    tops_text=lda.get_document_topics(corpus, minimum_probability=0)
    label=0
    for tup in tops_text:
        if tup[0] in topics.keys():
            if tup[1]>topics[tup[0]]:
                label=1
                break
    return label





