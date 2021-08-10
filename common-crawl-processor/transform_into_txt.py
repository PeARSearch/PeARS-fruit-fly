"""Common Crawl processor - transformation of zipped .xml files into .txt files

Usage:
  transform_into_txt.py --folder=<foldername>
  transform_into_txt.py (-h | --help)
  transform_into_txt.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --folder=<foldername>     Only the name of the folder where the zipped .xml files are located

"""

import glob
import csv
import gzip
import os
from docopt import docopt

def transform_xml_into_txt(f_globs, folder):
	f_txt = './'+folder+'/docs_0.txt'
	os.remove(f_txt)
	n_file=0
	n_doc=0
	for f_gz in f_globs:
		with gzip.open(f_gz,'rt') as f:
			doc = ""
			for line in f.read().splitlines():
				if line.startswith("<doc"):
					doc = ""
					continue
				if line.startswith("</doc>"):
					if doc != "" or doc != " ":
						n_doc+=1
						if os.path.exists(f_txt):
							if os.path.getsize(f_txt) < 209715200:
								pass
							else:
								n_file+=1
								f_txt = f_txt.split("_")[0]+"_"+str(n_file)+".txt"
								print(f_txt)
						with open(f_txt, 'a') as in_file:
							in_file.write(doc+"\n")
						in_file.close()  

					if n_doc%100==0:
						print(f"{n_doc} processed")
					continue
				if line == "":
					continue
				else: 
					doc = doc+" "+line
					continue

if __name__ == '__main__':
    args = docopt(__doc__, version='Common Crawl Processor')

    folder = args['--folder']

    f_globs= glob.glob("./"+folder+"/*.gz")
    transform_xml_into_txt(f_globs, folder)
