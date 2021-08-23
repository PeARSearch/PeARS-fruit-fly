import json

def write_as_json(dic, f):
  output_file = open(f, 'w', encoding='utf-8')
  json.dump(dic, output_file) 

def append_as_json(dic, f):
  output_file = open(f, 'a', encoding='utf-8')
  json.dump(dic, output_file) 
  output_file.write("\n")