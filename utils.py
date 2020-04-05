import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
from PIL import ImageFile
import os
from sklearn.utils import shuffle


image_transformations_with_normalization = transforms.Compose([transforms.Resize((450,450)), transforms.CenterCrop(448), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

image_transformations = transforms.Compose([transforms.Resize((450,450)), transforms.CenterCrop(448), transforms.ToTensor()])


class MultitagDataset(Dataset):
    
    def __init__(self, image_root_path, label, tags, transform = image_transformations_with_normalization):
        self.tags = tags
        self.image_root_path = image_root_path
        self.transform = transform
        self.label = label
        
    def __len__(self):
        return self.tags.shape[0]
        
    
    def __getitem__(self, idx):
        try:
            label = np.array(self.tags.iloc[idx][self.label])
            file = self.image_root_path + self.tags.iloc[idx]['file']
            img = Image.open(file)
            images = self.transform(img)
            labels = torch.from_numpy(label)
            return images, labels
        except Exception as e:
            print(e)
            
            
def get_data(label, train, seed = 100):
    np.random.seed(seed)

    tags = pd.read_pickle('tags_onehot_corrected.pkl')

    files = pd.read_pickle('good_files.pkl')[0].values

    tags = tags[tags.file.isin(files)]

    train_test_mask = np.random.rand(len(tags)) < 0.8

    if train:
        tags = tags[train_test_mask]
        labels = tags[label].values

        mask = labels == 1

        positive = tags[mask].sample(n=15000, replace=True)

        negative = tags[~mask].sample(n=20000, replace=True)
        
        return shuffle(positive.append(negative, ignore_index = True))
    else:
        return tags[~train_test_mask]
    
    
def get_data_multilabel(label):
    data = get_data(label, False)
    indices = []
    for i in range(len(data)):
        labels = data.iloc[i,:31].values
        if np.sum(labels) > 1:
            indices.append(i)
    return data.iloc[indices]

            
            

