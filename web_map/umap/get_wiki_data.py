"""Get Wikipedia data for UMAP processing

Usage:
  get_wiki_data.py --lang=<str>
  get_wiki_data.py (-h | --help)
  get_wiki_data.py --version

Options:
  --lang=<str>         The language of the Wikipedia to process.
  -h --help            Show this screen.
  --version            Show version.

"""

from docopt import docopt
import os
from os.path import join
import pathlib
import re
import bz2
import pickle
import requests
from pathlib import Path
import subprocess
import sentencepiece as spm

def bz2_uncompress(filepath):
    print("--- Uncompressing downloaded bz2:",filepath,"---")
    newfilepath = filepath.replace(".bz2","")
    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)
    return newfilepath

def read_wiki_links(lang):
    with open("./wiki_dump_links/"+lang+"_wiki_dump_links.txt") as f:
        return f.read().splitlines()

def get_wiki_links(lang):
    print("\n--- Getting wiki links for download ---")
    html = requests.get(url = 'https://dumps.wikimedia.org/'+lang+'wiki/latest/').text
    match = re.findall(lang+'wiki-latest-pages-articles[0-9]*\.xml-p[0-9]*p[0-9]*\.bz2', html)
    if len(match) == 0:
        match = re.findall(lang+'wiki-latest-pages-articles.xml.bz2', html) #For wikis with only one dump file.
    match = list(set(match))

    Path("./wiki_dump_links").mkdir(exist_ok=True, parents=True)
    filename = "./wiki_dump_links/"+lang+"_wiki_dump_links.txt"
    outf = open(filename,'w')
    for url in match:
        outf.write("https://dumps.wikimedia.org/"+lang+"wiki/latest/"+url+"\n")
    outf.close()
    return filename


def extract_xml(lang,bz2_file):
    print("\n--- Downloading and extracting XML version of corpus ---")

    out_file = open(bz2_file.replace('bz2','xml'),'w')
    uncompressed = bz2_uncompress(bz2_file)
    os.remove(bz2_file)
    f=open(uncompressed,'r')

    word_count = 0
    content = ""
    for l in f:
        if "</page" in l:
            out_file.write(l)
            content = ""
        else:
            out_file.write(l)
            word_count+=len(l.split()) #Very rough, will include markup. But doesn't matter.

    out_file.write("</mediawiki>")
    print("Word count:",word_count)
    f.close()
    os.remove(uncompressed)
    out_file.close()

def get_categories(bz2_file):
    print("\n--- Get categories from corpus ---")
    xml_file = bz2_file.replace('bz2','xml')
    all_categories = {}

    title = ""
    f=open(xml_file,'r')
    for l in f:
        l.rstrip('\n')
        if "<title" in l:
            m = re.search('<title>([^<]*)<',l)
            title = m.group(1)
            all_categories[title] = []
        if "[[Category:" in l:
            m = re.search('\[\[Category:([^\]]*)\]\]',l)
            if m:
                cat = m.group(1)
                all_categories[title].append(cat)
    pklf = bz2_file.replace('bz2','cats.pkl')
    with open(pklf, 'wb') as f:
        pickle.dump(all_categories,f)

def mk_linear(bz2_file, cat_file):
    print("\n--- Generating linear version of corpus ---")

    xml_file = bz2_file.replace('bz2','xml')
    tmp_linear_file = bz2_file.replace('bz2','raw.tmp')
    command = ['python3','-m','wikiextractor.WikiExtractor','--output',tmp_linear_file,'--no-templates','--html-safe','False',xml_file]
    subprocess.run(command)

    all_categories = pickle.load(open(cat_file,'rb'))
    tmpf = open(tmp_linear_file,'r')
    linear_filename = tmp_linear_file.replace('tmp','txt')
    linear_file = open(linear_filename,'w')
    for l in tmpf:
        if '<doc' in l:
            m = re.search('.*title="([^"]*)">',l)
            title = m.group(1)
            categories = all_categories[title] 
            cs = ' categories="'+'|'.join([c for c in categories])+'"'
            linear_file.write(l.replace('>',cs+'>'))
        else:
            linear_file.write(l.lower())
    linear_file.close()
    tmpf.close()
    os.remove(tmp_linear_file)
    os.remove(xml_file)


def apply_spm(bz2_file):
    print("\n--- Applying sentencepiece to corpus ---")
    start_doc=""
    doc=""
    txt_path = bz2_file.replace('bz2','raw.txt')
    spm_filename = bz2_file.replace('.bz2','.sp') 
    spf = open(spm_filename,'w')
    
    f = open(txt_path,'r')
    for l in f:
        if '<doc' in l:
            start_doc = l
        elif '</doc' in l:
            if len(doc) >= 1000: #Only keep long enough docs
                doc = ' '.join([wp for wp in sp.encode_as_pieces(doc)])+'\n'
                spf.write(start_doc)
                spf.write(doc)
                spf.write(l)
            doc = ""
        else:
            if len(doc) < 1500 and len(l) > 0:
                doc+=l
    f.close()
    spf.close()
    os.remove(txt_path)
    print("\n All done!! Your sentencepieced corpus is at",spm_filename,".")


if __name__ == '__main__':
    args = docopt(__doc__, version='Apply UMAP to Wikipedia, ver 0.1')
    lang = args['--lang']
    sp_model_path = f"../../spm/spm.{lang}wiki.model"
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    processed_dir = join(pathlib.Path(__file__).parent.resolve(),'processed')
    print(processed_dir)

    link_file = get_wiki_links(lang)
    wiki_paths = read_wiki_links(lang)

    for wiki_path in wiki_paths:
        print(wiki_path)
        subprocess.run(["wget",wiki_path, "-P",processed_dir])
        bz2_file = join('./processed',wiki_path.split('/')[-1])

        extract_xml(lang,bz2_file)
        get_categories(bz2_file)
        cat_file = bz2_file.replace('bz2','cats.pkl')
        mk_linear(bz2_file,cat_file)
        apply_spm(bz2_file)
