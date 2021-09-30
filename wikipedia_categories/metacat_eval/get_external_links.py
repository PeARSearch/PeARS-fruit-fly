import os
import re
import bz2
import sys
import gzip
import shutil
import subprocess

#max_url_length = 100

def bz2_uncompress(filepath):
    print("Uncompressing bz2:",filepath)
    newfilepath = filepath.replace(".bz2","")
    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)
    return newfilepath

def read_wiki_paths():
    with open("./wiki_dump_links.txt") as f:
        return f.read().splitlines()

def get_category(line):
    cat = ""
    m = re.search("\[\[.*Category:([^\]]*)\|*.*\]\]",line)
    if m:
        cat = m.group(1)
        if cat[-2:] == '| ':
           cat = cat[:-2]
    return cat

def extract_urls(uncompressed):
    redirect = False
    title = ""
    categories = []
    urls = []

    link_path = uncompressed+".links"
    out_file = open(link_path,'w')
    f=open(uncompressed,'r')
    for l in f:
        l=l.rstrip('\n')
        if "<title" in l:
            m = re.search(r'<title>(.*)</title>',l)
            title=m.group(1).replace(' ','_') 
        if l[:7] == "* [http":
            print(l)
            m = re.search('\* \[(http\S*) ',l)
            if m:
                print("URL",m.group(1),'\n')
                urls.append(m.group(1))
        if l[:12] == "* {{cite web":
            print(l)
            m = re.search("url\s*=\s*(http[^\|]*)\|",l)
            if m :
                print("URL",m.group(1),'\n')
                urls.append(m.group(1))
        if "[[Category:" in l:
            cat = get_category(l)
            if cat != "":
                categories.append(cat)

        if "</page" in l:
            if not redirect:
                categories_s = '|'.join([c for c in categories])
                out_file.write("https://en.wikipedia.org/wiki/"+title+'|'+categories_s+'\n')
                for url in urls:
                    out_file.write(url+'|'+categories_s+'\n')
            title = ""
            urls.clear()
            redirect = False
            categories.clear()
       
        if "<redirect title" in l:
            redirect = True
            
    f.close()
    out_file.close()
    return link_path


wiki_paths = read_wiki_paths()

for bz2_file in wiki_paths:
    subprocess.run(["wget",bz2_file])
    local_file = bz2_file.split('/')[-1]
    uncompressed = bz2_uncompress(local_file)
    os.remove(local_file)

    link_path = extract_urls(uncompressed)
    print(link_path)

    with open(link_path, 'rb') as f_in:
        with gzip.open('./links/'+link_path+'.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)   

    os.unlink(link_path) 
    os.remove(uncompressed)
