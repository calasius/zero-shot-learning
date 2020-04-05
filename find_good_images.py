import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
from PIL import ImageFile
import os

transform = transforms.Compose([transforms.Resize((450,450)), transforms.CenterCrop(448), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

df = pd.read_csv('tags_corrected.csv', index_col=0)

files = []

root = '/data/hotel_images/'

for i in range(len(df)):
    file = df.iloc[i]['MEDIA_KEY']
    
    try:
        img = Image.open(root + file)
        transform(img)
        files.append(file)
    except Exception as e:
        print(e)
    
    if i % 100 == 0:
        print(i)
print(files)
pd.DataFrame(files).to_pickle('good_files.pkl')
