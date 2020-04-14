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


to_image = transforms.Compose([transforms.ToPILImage(), transforms.Resize(448)])

transform = transforms.Compose([transforms.Resize((450,450)), transforms.CenterCrop(448), transforms.ToTensor()])

def get_max_activation_positions(feature_maps):
    feature_maps_plain = feature_maps.view(feature_maps.shape[0], feature_maps.shape[1], -1)
    max_feature_maps_plain = torch.argmax(feature_maps_plain, dim=2)
    y = max_feature_maps_plain / 14
    x = max_feature_maps_plain % 14
    
    return torch.stack([x,y], dim = 2)

def get_channel_clusters(max_activation_positions, num_clusters=4):
    channel_features = max_activation_positions.permute((1,0,2)).reshape(512,-1)
    kmeans = KMeans(n_clusters= num_clusters, random_state=0).fit(channel_features.detach().numpy())
    return torch.from_numpy(kmeans.predict(channel_features.detach().numpy()))


def build_feature_map_mask(channel_groups, group):
    return channel_groups == group


def get_attention_map(group, channel_groups, image_feature_map):
    mask = build_feature_map_mask(channel_groups,group).type(torch.float)
    return torch.sum(image_feature_map * mask.view(512,1,1), dim=0).view(1,14,14)


def get_attention_map_image(group, channel_groups, image_feature_map):
    att = get_attention_map(group, channel_groups, image_feature_map)
    min_value = torch.min(att)
    max_value = torch.max(att)
    att = att / torch.max(att)
    print(att.shape)
    gray_scale_att_image = Image.fromarray(np.uint8(att.detach().numpy()[0] * 255) , 'L').resize((448,448), Image.BILINEAR)
    heatmapimg = np.array(np.array(gray_scale_att_image), dtype = np.uint8)
    
    return Image.fromarray(cv2.applyColorMap(-heatmapimg, cv2.COLORMAP_JET))


def plot_heatmap_on_image(image_file_name, image_feature_map, channel_groups , group = 1):
    img = Image.open(image_file_name)
    img_tensor = transform(img)
    img = TF.to_pil_image(img_tensor)
    att_image = get_attention_map_image(group, channel_groups, image_feature_map)
    return Image.blend(img, att_image, 0.3)


def get_max_center(group, channel_groups, image_feature_map):
    att = get_attention_map(group, channel_groups, image_feature_map)
    att_max_index = torch.argmax(att).float()
    x_att_max = (att_max_index % 14) + 0.5
    y_att_max = (att_max_index // 14) + 0.5
    x_max = (x_att_max / 14) * 448
    y_max = (y_att_max / 14) * 448
    return x_max, y_max


def plot_box_on_image(plot_name, image_file_name, channel_groups, image_feature_map, w = 100, h = 100):
    
    plt.xticks([])
    plt.yticks([])
    
    f, axarr = plt.subplots(2,2)
    
    for group in range(4):
        i = group % 2
        j = group // 2
        x,y = get_max_center(group, channel_groups, image_feature_map)
        x = x - w/2
        y = y - h/2
        image = plot_heatmap_on_image(image_file_name, image_feature_map, channel_groups, group=group)
        axarr[i,j].imshow(image)
        rect = patches.Rectangle((x,y),w,h, edgecolor='b', facecolor="none")
        axarr[i,j].add_patch(rect)
        axarr[i,j].set_yticklabels([])
        axarr[i,j].set_xticklabels([])
            
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.savefig(plot_name)
    
    
    
def plot_attention_samples(category):

    tags = df.columns.values

    bathroom_df = df[df[category] == 1]

    bathroom_indices = bathroom_df.index.values

    bathroom_df.reset_index(inplace = True)

    feature_maps = features[bathroom_indices]

    max_points = get_max_activation_positions(feature_maps)

    channel_groups = get_channel_clusters(max_points, num_clusters=5)

    for j in range(4):
        image_feature_map = feature_maps[j]
        image_file_name = '/data/hotel_images/' + df.iloc[bathroom_indices[j]].file
        plot_box_on_image('plots/'+category + '_attention_' + str(j), image_file_name, channel_groups, image_feature_map)
        

bird_features = torch.load('untracked/feature_maps_albatross.pt')

bird_max_activation = get_max_activation_positions(bird_features)

bird_channel_groups = get_channel_clusters(bird_max_activation, num_clusters=4)

path_bird_images = 'images/albatross/'

bird_image_file_names = os.listdir(path_bird_images)

def plot_bird_attention_samples():
    for j in range(4):
        image_feature_map = bird_features[4+j]
        image_file_name = path_bird_images + bird_image_file_names[4+j]
        image_feature_map = bird_features[4+j]
        plot_box_on_image('plots/bird_attention_' + str(j), image_file_name, bird_channel_groups, image_feature_map)
        
 

features = torch.load('untracked/feature_maps.pt')
df = pd.read_pickle('tags_onehot_corrected.pkl')
        
#plot_attention_samples('bathroom')
#plot_attention_samples('guest_room')
#plot_attention_samples('kitchen')
#plot_attention_samples('living_room')
#plot_attention_samples('health_club')

plot_bird_attention_samples()

