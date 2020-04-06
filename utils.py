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
from torchnlp.word_to_vector import FastText


RANDOM_SEED = 100

EMBEDDING_SIZE = 300

np.random.seed(RANDOM_SEED)

category_to_word = {'ballroom':'ballroom', 'bar_lounge':'bar', 'bathroom':'bathroom', 'beach':'beach', 'breakfast':'breakfast', 'city_view': 'city', 
                   'golf_course':'golfcourse', 'guest_room':'bedroom', 'health_club':'gym', 'hotel_front':'facade', 'kitchen':'kitchen', 'living_room':'livingroom',
                   'lobby_view':'lobby', 'meeting_room':'meetingroom', 'natural_view':'mountain', 'pool_view':'pool', 'recreational_facility':'recreational',
                   'restaurant':'restaurant', 'spa':'spa'}


test_categories = ['kitchen', 'restaurant', 'hotel_front', 'natural_view']

word_embedding = FastText(cache='/data/word_embeddings/')


category_embeddings = {}
for k in category_to_word.keys():
    category_embeddings[k] = word_embedding[category_to_word[k]]
    
def get_category_embedding_matrix():
    categories = list(category_embeddings.keys())
    categories.sort()
    
    matrix = torch.empty(len(categories), EMBEDDING_SIZE)
    
    i = 0
    for k in categories:
        matrix[i] = category_embeddings[k]
        i = i + 1
        
    return matrix.T
    
    

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
            
class ZeroshotDataset(Dataset):
    
    def __init__(self, image_root_path, data, transform = image_transformations_with_normalization):
        self.data = data
        self.image_root_path = image_root_path
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
        
    
    def __getitem__(self, idx):
        try:
            label = self.data.iloc[idx, :-1].values.astype(np.float)
            file = self.image_root_path + self.data.iloc[idx]['file']
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


def get_data(train, seed=100):
    np.random.seed(seed)

    tags = pd.read_pickle('tags_onehot_corrected.pkl')

    files = pd.read_pickle('good_files.pkl')[0].values

    tags = tags[tags.file.isin(files)]

    train_test_mask = np.random.rand(len(tags)) < 0.8
    
    if train:
        return tags[train_test_mask]
    else:
        return tags[~train_test_mask]
    
    
def get_data_multilabel():
    data = get_data(False)
    indices = []
    for i in range(len(data)):
        labels = data.iloc[i,:31].values
        if np.sum(labels) > 1:
            indices.append(i)
    return data.iloc[indices]


def get_zeroshot_dataset(train, samples_per_category = 1500):
    tags = pd.read_pickle('tags_onehot_corrected.pkl')
    

    categories = list(category_to_word.keys())
    categories.sort()
        
    data = tags[categories]
    data['file'] = tags['file']
        
    if train:
        
        indices = []
        multitag_indices = []
        for i in range(data.shape[0]):
            if np.sum(data.iloc[i, :-1].values) == 1:
                indices.append(i)
            else:
                multitag_indices.append(i)
        
        data = data.iloc[indices]
        
        dataset = []
        
        for cat in categories:
            samples = data[data[cat] == 1]
            samples = samples.sample(n=samples_per_category, replace = True)
            dataset.append(samples)
            
        dataset = pd.concat(dataset)
        
        return shuffle(dataset)
    else:
            
        indices = []
        for i in range(data.shape[0]):
            if np.sum(data.iloc[i, :-1].values) > 1:
                indices.append(i)
                
        return shuffle(data.iloc[indices])
