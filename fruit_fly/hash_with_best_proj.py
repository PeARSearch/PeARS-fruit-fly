"""Hash documents with best performing fly
Usage:
  hash_with_best_proj.py --dataset=<filename> 
  hash_with_best_proj.py (-h | --help)
  hash_with_best_proj.py --version
Options:
  -h --help                       Show this screen.
  --version                       Show version.
  --train_path=<filename>         Name of training file
"""

# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolve_flies import Fly
import pickle
from hyperparam_search import read_n_encode_dataset
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from hash import read_vocab, read_projections, hash_input
from utils import hash_dataset_
from timer import Timer

with open('../wikipedia_categories/wiki_cats/best_val_score', 'rb') as f:  # modified the name of the fruit-fly here
  best = pickle.load(f)
    
projection=best.projection  # get the projection
print(projection.shape)
wta = best.wta
top_word=700

t = Timer()
vocab, reverse_vocab, logprobs = read_vocab()
# projection_functions, pn_to_kc = read_projections(d)
vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')

# Setting up the fly
PN_size = projection.shape[1]
KC_size = projection.shape[0]
proj_size = len(projection_functions[0])
print("SIZES PN LAYER:",PN_size,"KC LAYER:",KC_size)
print("SIZE OF PROJECTIONS:",proj_size)
print("SIZE OF FINAL HASH:",percent_hash,"%")

projection_layer = np.zeros(PN_size)
kenyon_layer = np.zeros(KC_size)

#Reading through documents
n_doc = 0
doc = ""

M_data = []
M_col = []
M_row = []
IDs = []
classes = {}
keywords = {}

in_file_path = args["--file"]
in_file = in_file_path.split('/')[-1]
trial = d.split('.')[0].split('_')[1]
params = '.kc'+str(KC_size) + '.size'+str(proj_size) + '.trial'+str(trial) + ".top"+str(top_tokens)+".wta"+str(percent_hash)

pathlib.Path('./tmp').mkdir(parents=True, exist_ok=True)
hs_file = os.path.join('tmp', in_file.replace('.sp',params+'.hs')).replace('.projs/', '')
ID_file = os.path.join('tmp', in_file.replace('.sp',params+'.ids')).replace('.projs/', '')
class_file = os.path.join('tmp', in_file.replace('.sp',params+'.cls')).replace('.projs/', '')
keyword_file = os.path.join('tmp', in_file.replace('.sp',params+'.kwords')).replace('.projs/', '')


with open(in_file_path,'r') as f:
    for l in f:
        l = l.rstrip('\n')
        if l[:4] == "<doc":
            m = re.search(".*id=([^ ]*) ",l)
            ID=m.group(1)
            m = re.search(".*class=([^ ]*)>",l)
            cl=m.group(1)
            IDs.append(ID+'_'+cl)
            classes[IDs[-1]] = m.group(1)
            print("Processing",IDs[-1])
        elif l[:5] == "</doc":
            #print("Wordpiecing...")
            #t.start()
            ll = sp.encode_as_pieces(doc)
            #t.stop()
            #print("Vectorizing...")
            #t.start()
            X = vectorizer.fit_transform([doc])
            #t.stop()
            X = X.toarray()[0]
            vec = logprobs * X
            vec = wta(vec, top_tokens, percent=False)
            #print("Hashing...")
            #t.start()
            hs = hash_input(vec,reverse_vocab,percent_hash, KC_size, pn_to_kc, projection_functions)
            #t.stop()
            hs = coo_matrix(hs)
            #print(IDs[-1],' '.join([str(i) for i in hs.col]))
            keywords[IDs[-1]] = [reverse_vocab[w] for w in return_keywords(vec)]
            print(keywords[IDs[-1]])
            for i in range(len(hs.data)):
                M_row.append(n_doc)
                M_col.append(hs.col[i])
                M_data.append(hs.data[i])
                #M_data.append(1)
            doc = ""
            n_doc+=1
            #time.sleep(0.002)    #Sleep a little to consume less CPU
        else:
            doc+=l+' '
M = coo_matrix((M_data, (M_row, M_col)), shape=(n_doc, KC_size))

with open(hs_file,"wb") as hsf:
    pickle.dump(M,hsf)
with open(ID_file,"wb") as IDf:
    pickle.dump(IDs,IDf)
with open(keyword_file,"wb") as kf:
    pickle.dump(keywords,kf)
with open(class_file,"wb") as cf:
    pickle.dump(classes,cf)
