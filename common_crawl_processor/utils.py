import json
import os 
from pathlib import Path

def write_as_json(dic, f):
  output_file = open(f, 'w', encoding='utf-8')
  json.dump(dic, output_file) 

def append_as_json(dic, f):
  output_file = open(f, 'a', encoding='utf-8')
  json.dump(dic, output_file) 
  output_file.write("\n")

def append_json_check_len(dic, filename):
  new_filename=filename
  output_file = open(filename, 'a', encoding='utf-8') 
  if os.path.exists(filename):
    if os.path.getsize(filename) < 524288000:
      pass
    else:
      path = Path(filename)
      f = path.parts[-1]
      name, num = f.split("_")[0], int(f.split("_")[1].replace(".json", ""))
      j_name = name+"_"+str(num+1)+".json"
      print(filename)
      output_file.close()
      filename=str(path).replace(f, "")+j_name
      output_file = open(filename, 'a', encoding='utf-8')
  json.dump(dic, output_file)
  output_file.write("\n")
  return filename
