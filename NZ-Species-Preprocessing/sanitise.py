import os, shutil

import pandas as pd

def format_row(row):
  if row['taxonRank'] in ['SUBSPECIES', 'VARIETY', 'FORM']:
    return (row['species'], row['taxonRank'], row['infraspecificEpithet'])
  return (row[row['taxonRank'].lower()], row['taxonRank'])

info = pd.read_csv('NZ-Species.csv', delimiter = '\t')
classes = {}
for _, row in info.iterrows():
  class_name = row['verbatimScientificName']
  if class_name in classes:
    if format_row(row) != classes[class_name]:
      raise ValueError(str(format_row(row)))
  else:
    classes[class_name] = format_row(row)
new_classes = {}
for k, v in classes.items():
  if v[1] in ['GENUS', 'ORDER', 'FAMILY']:
    if ' ' not in k or k.endswith('st1') or k.endswith('st2') or k.endswith('virus'):
      new_classes[k] = None
    else:
      new_classes[k] = k
  elif v[1] == 'SPECIES':
    if k.endswith('virus') or ' ' not in k:
      new_classes[k] = None
    elif k.split(' ')[0 : 2] == v[0].split(' ') or v[0] in classes:
      new_classes[k] = v[0]
    elif len(k.split(' ')) == 3 and ' '.join(k.split(' ')[0 : 2]) in classes:
      new_classes[k] = ' '.join(k.split(' ')[0 : 2])
    else:
      new_classes[k] = k
  else:
    new_classes[k] = (' '.join(k.split(' ')[0 : 2]), v[0])
delete_classes = []
rename_classes = {}
for k, v in new_classes.items():
  if type(v) is tuple:
    continue
  else:
    if v == None:
      delete_classes.append(k)
    elif v in rename_classes:
      rename_classes[v].append(k)
    else:
      rename_classes[v] = [k]
for k, v in new_classes.items():
  if type(v) is tuple:
    if k == 'Penion cuvierianus jeakingsi':
      rename_classes['Penion ormesi'] = [k]
    elif v[0] in rename_classes:
      rename_classes[v[0]].append(k)
    elif v[1] in rename_classes:
      rename_classes[v[1]].append(k)
    else:
      rename_classes[v[0]] = [k]

for class_i in delete_classes:
  shutil.rmtree(f'dataset/{class_i}')
  
class_list = os.listdir('dataset')

def move_class(source_name, target_name):
  if source_name != target_name and source_name in class_list:
    if target_name in class_list:
      for file_i in os.listdir(f'dataset/{source_name}'):
        shutil.move(f'dataset/{source_name}/{file_i}', f'dataset/{target_name}/{file_i}')
      os.rmdir(f'dataset/{source_name}')
      class_list.remove(source_name)
    else:
      shutil.move(f'dataset/{source_name}', f'dataset/{target_name}')
      class_list.remove(source_name)
      class_list.append(target_name)

for k, v in rename_classes.items():
  if len(v) == 1:
    if v[0] != k:
      move_class(v[0], k)
  elif all([vi.startswith(k) for vi in v]):
    for vi in v:
      move_class(vi, k)

special_instructions = open('special_instructions.txt', 'r').read().split('\n')
for special_i in special_instructions:
  if ',' in special_i:
    parsed_i = special_i.split(',')
    if parsed_i[0] == 'D':
      shutil.rmtree(f'dataset/{parsed_i[1]}')
      class_list.remove(parsed_i[1])
    elif parsed_i[0] == 'R':
      for vi in parsed_i[2:]:
        move_class(vi, parsed_i[1])
