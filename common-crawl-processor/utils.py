import json
import os 
from pathlib import Path
import gzip
import shutil

def write_as_json(dic, f):
  output_file = open(f, 'w', encoding='utf-8')
  json.dump(dic, output_file) 
  output_file.write("\n")

def append_as_json(dic, f):
  output_file = open(f, 'a', encoding='utf-8')
  json.dump(dic, output_file) 
  output_file.write("\n")

def compress_file(file_path):
  with open(file_path, 'rb') as f_in:
    with gzip.open(file_path+'.gz', 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)   
  os.unlink(file_path)
  print(f"Newly compressed {file_path}.gz created and old {file_path} removed")

def append_json_check_len(dic, file_path):
  output_file = open(file_path, 'a', encoding='utf-8') 
  if os.path.exists(file_path):
    if os.path.getsize(file_path) < 524288000:
      pass
    else:
      path = Path(file_path)
      f = path.parts[-1]
      name, num = f.split("_")[0], int(f.split("_")[1].replace(".json", ""))
      j_name = name+"_"+str(num+1)+".json"
      print(file_path)
      output_file.close()
      compress_file(file_path)
      file_path=str(path).replace(f, "")+j_name
      output_file = open(file_path, 'a', encoding='utf-8')
  json.dump(dic, output_file)
  output_file.write("\n")
  return file_path

