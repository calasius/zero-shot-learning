import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
from PIL import ImageFile
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import urllib
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from torch.distributions import MultivariateNormal
from torch.nn import *
from torch import nn
from utils import *
from torch import optim
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import balanced_accuracy_score
from torch.autograd import Variable
from attention_model import *

    
def calculate_pretrain_dataset():
        #max_positions = torch.load('max_positions.pt')
        
        max_positions = ((torch.load('max_positions.pt') + 0.5) / 14.) * 448.

        attention_maps_unnormalized = torch.load('attention_maps_unnormalized.pt')

        max_positions[0].shape

        boxes = torch.empty(max_positions.shape[0], max_positions.shape[1], max_positions.shape[2]+1)

        for i in range(boxes.shape[0]):
            edge_size = torch.from_numpy(np.full((max_positions.shape[1],1), 200)).float()
            boxes[i] = torch.cat((max_positions[i], edge_size), dim=1)

        s = attention_maps_unnormalized.shape

        x = attention_maps_unnormalized.view(s[0], s[1], -1)
                    
        y = boxes

        return x,y
    
    
def pretrain_crop_network(model, device):
        
        x,y = calculate_pretrain_dataset()

        for i in range(4):
            pretrain_dataset = torch.utils.data.TensorDataset(x[i], y[i])
            pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=512, shuffle=True)
            pretrain_optimizer = optim.Adam(model.get_crop_network_parameters(i), lr=0.001)

            print('network = %s' % i)

            for epoch in range(100):
                total_loss = 0
                for xb, yb in pretrain_loader:
                    pretrain_optimizer.zero_grad()
                    pred = model.crop_network_forward(xb.to(device),i)
                    
                    #print('pred = %s' % pred[0])
                    #print('true = %s' % yb[0])
                    loss = F.mse_loss(pred, yb.to(device))

                    loss.backward()

                    pretrain_optimizer.step()

                    total_loss += loss.item()
                print('loss = %s' % (total_loss / len(pretrain_dataset)))

                
def pretrain_feature_attention_network(model, device):
        
        y = torch.load('feature_maps_weights.pt')
        x = F.avg_pool2d(torch.load('feature_maps.pt'), FEATURE_MAP_SIZE)
        s = x.shape
        x = x.view(s[0],-1)
        

        for i in range(4):
            pretrain_dataset = torch.utils.data.TensorDataset(x, y[i])
            pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=512, shuffle=True)
            pretrain_optimizer = optim.Adam(model.get_feature_map_attention_network_parameters(i), lr=0.001)

            print('network = %s' % i)

            for epoch in range(60):
                total_loss = 0
                for xb, yb in pretrain_loader:
                    
                    pretrain_optimizer.zero_grad()
                    pred = model.feature_map_attention_network_forward(xb.to(device),i)

                    loss = F.binary_cross_entropy(pred, yb.to(device))

                    loss.backward()

                    pretrain_optimizer.step()

                    total_loss += loss.item()
                print('loss = %s' % (total_loss))

                


                
            
def train_model(device, label):

    #model = AttentionModel(torch.device(device), load_pretrained_networks = False)

    batch_size = 16
    
    train_data = get_data(label, True)
    
    multitag_data = get_data_multilabel(label)
    

    train_dataset = MultitagDataset('/data/hotel_images/', label, train_data)

    test_dataset = MultitagDataset('/data/hotel_images/', label, multitag_data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    #pretrain_crop_network(model, device)

    #pretrain_feature_attention_network(model, device)

    #model.save_crop_network()

    #model.save_feature_map_attention_network()

    model = AttentionModel(torch.device(device), load_pretrained_networks = True)

    optimizer = optim.Adam(model.get_model_parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0, amsgrad=False)



    for i in range(5):
        for images, labels in train_dataloader:

            optimizer.zero_grad()

            probs, multi_attention_loss = model(images.to(device))

            loss = F.binary_cross_entropy(probs, labels.to(device).float()) + multi_attention_loss
            print('model = %s, epoch = %s, loss = %s' %(label, i, loss.item()))

            loss.backward()
            optimizer.step()
            
            
            model.eval_mode()

            with torch.no_grad():
                
                total = .0
                y_true = []
                y_pred = []
                for images, labels in test_dataloader:
                    probs, _ = model.forward(images.to(device), eval_mode=True)
                    pred = probs > 0.5
                    pred = pred.float().T.detach().cpu().numpy().ravel()
                    labels = labels.float().detach().cpu().numpy().ravel()
                    
                    y_true.extend(labels.tolist())
                    y_pred.extend(pred.tolist())

                    label_indices = np.argwhere(pred == 1.0).ravel()
                    pred_indices = np.argwhere(labels == 1.0).ravel()


                    pred_ones = {*pred_indices} if len(pred_indices) > 0 else set()
                    label_ones = {*label_indices} if len(label_indices) > 0 else set()

                    good_ones = pred_ones.intersection(label_ones)

                    total += len(good_ones)
                    
                        
                acc = balanced_accuracy_score(y_true, y_pred)
                
                if acc > 0.75:
                    print('saving model')
                    model.save_model('models/%s-%.2f' % (label, acc))

                print('multitag accuracy = %s' % str(total))
                print('balanced accuracy = %s' % str(acc))
            

            model.train_mode()
    
    
device = 'cuda:0'

label = 'natural_view'

train_model(device, label)
