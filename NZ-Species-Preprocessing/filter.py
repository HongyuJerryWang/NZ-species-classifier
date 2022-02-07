import os
import tqdm

from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS

classes = os.listdir('dataset')
for class_i in tqdm.tqdm(classes):
  image_files = os.listdir(f'dataset/{class_i}')
  for image_file in image_files:
    if len(image_file.split('.')) > 2 or not image_file.lower().endswith(IMG_EXTENSIONS):
      print(f'Removing dataset/{class_i}/{image_file}', flush=True)
      os.remove(f'dataset/{class_i}/{image_file}')
    else:
      try:
        im = Image.open(f'dataset/{class_i}/{image_file}')
      except:
        print(f'Removing dataset/{class_i}/{image_file}', flush=True)
        os.remove(f'dataset/{class_i}/{image_file}')
