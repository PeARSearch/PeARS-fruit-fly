import re
import requests

html = requests.get(url = 'https://dumps.wikimedia.org/enwiki/latest/').text
match = re.findall(r'enwiki-latest-pages-articles[0-9]*\.xml-p[0-9]*p[0-9]*\.bz2', html)
match = list(set(match))

outf = open("./wiki_dump_links.txt",'w')
for url in match:
    outf.write("https://dumps.wikimedia.org/enwiki/latest/"+url+"\n")
outf.close()
