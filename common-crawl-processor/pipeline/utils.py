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

def append_json_check_len(dic, filename, folder):
  output_file = open(folder+filename, 'a', encoding='utf-8') 
  if os.path.exists(folder+filename):
    if os.path.getsize(folder+filename) < 524288000:
      pass
    else:
      name, num = filename.split("_")[0], int(filename.split("_")[1])
      filename = name+"_"+num
      print(filename)
      output_file.close()
      output_file = open(folder+filename, 'a', encoding='utf-8')
  json.dump(dic, output_file)
  output_file.write("\n")
  return filename
