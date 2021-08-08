import json
import os 

def write_as_json(dic, f):
  output_file = open(f, 'w', encoding='utf-8')
  json.dump(dic, output_file) 
  output_file.write("\n")

def append_as_json(dic, f):
  output_file = open(f, 'a', encoding='utf-8')
  json.dump(dic, output_file) 
  output_file.write("\n")

def append_json_check_len(dic, filename):
  output_file = open(filename, 'a', encoding='utf-8') 
  if os.path.exists(filename):
    if os.path.getsize(filename) < 524288000:
      pass
    else:
      split_ = filename.split("/")
      n_f = int(split_[-1].split("_")[-1].replace(".json", ""))+1
      filename = split_[-1].split("_")[0]+"_"+str(n_f)+".json"
      print(filename)
      output_file.close()
      output_file = open(filename, 'a', encoding='utf-8')
  json.dump(dic, output_file)
  output_file.write("\n")
  return filename
