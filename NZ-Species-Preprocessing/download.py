import os
import tqdm
import socket
import urllib.request

import pandas as pd

socket.setdefaulttimeout(15)

multimedia = pd.read_csv('multimedia.txt', delimiter = '\t')
dataset = pd.read_csv('NZ-Species.csv', delimiter = '\t')
for i, row in tqdm.tqdm(multimedia.iterrows()):
    species_dir ='dataset/' +  dataset.loc[dataset['gbifID'] == row['gbifID'], 'verbatimScientificName'].iat[0]
    if not os.path.exists(species_dir):
        os.makedirs(species_dir)
    try:
        urllib.request.urlretrieve(row['identifier'].replace('https://static.inaturalist.org/', 'https://inaturalist-open-data.s3.amazonaws.com/'), species_dir + '/' + str(i) + '.' + row['format'].split('/')[-1])
    except:
        try:
            urllib.request.urlretrieve(row['identifier'], species_dir + '/' + str(i) + '.' + row['format'].split('/')[-1])
        except:
            print('\nFailed to acquire', str(i), row['gbifID'], row['identifier'])
