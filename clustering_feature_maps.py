import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def get_max_activation_positions(feature_maps):
    feature_maps_plain = feature_maps.view(feature_maps.shape[0], feature_maps.shape[1], -1)
    max_feature_maps_plain = torch.argmax(feature_maps_plain, dim=2)
    y = max_feature_maps_plain / 14
    x = max_feature_maps_plain % 14
    
    return torch.stack([x,y], dim = 2)


def get_channel_clusters(max_activation_positions, num_clusters=4):
    channel_features = max_activation_positions.permute((1,0,2)).reshape(512,-1)
    kmeans = KMeans(n_clusters= num_clusters, random_state=0).fit(channel_features.numpy())
    return torch.from_numpy(kmeans.predict(channel_features.detach().numpy()))


def build_feature_map_mask(channel_groups, group):
    return channel_groups == group


df = pd.read_pickle('multitag_dataset.pkl')

features = torch.load('features.pt').detach().cpu()

embeddings = F.avg_pool2d(features, 14)

embeddings = embeddings.view(embeddings.shape[0], -1)

kmeans = KMeans(n_clusters= 30, random_state=0, n_jobs=8).fit(embeddings)

groups = kmeans.predict(embeddings)

all_group_positions = []
all_group_attention_maps = []
all_group_attention_maps_normalized = []
all_feature_map_weights = []

indices_track = []

for group in range(30):
    print(group)
    indices = np.argwhere(groups == group).ravel()
    indices_track.extend(indices.tolist())
    feature_maps = features[indices]
    print(feature_maps.shape)
    max_activation_positions = get_max_activation_positions(feature_maps)
    channel_groups = get_channel_clusters(max_activation_positions.detach().cpu(), num_clusters=4)
    
    group_positions = torch.empty(4,indices.shape[0],2)
    group_attention_maps = torch.empty(4, indices.shape[0],14,14)
    group_attention_maps_normalized = torch.empty(4, indices.shape[0],14,14)
    feature_map_weights = torch.empty(4,indices.shape[0], 512)
    
    for group in range(4):
        mask = build_feature_map_mask(channel_groups,group).type(torch.float)
        feature_map_weights[group] = torch.stack([mask] * indices.shape[0])
        attention_maps = torch.sum(feature_maps * mask.view(512,1,1), dim=1)
        max_values, _ = torch.max(attention_maps.view(attention_maps.shape[0],-1), dim=1)
        attention_maps_normalized = attention_maps / max_values.view(attention_maps.shape[0],1,1)
        
        max_pos = torch.argmax(attention_maps.view(attention_maps.shape[0],-1), dim=1)
        x = max_pos % 14
        y = max_pos / 14
        
        positions = torch.stack([x,y], dim=-1)
        
        group_positions[group] = positions
        group_attention_maps_normalized[group] = attention_maps_normalized
        group_attention_maps[group] = attention_maps
        
    all_group_positions.append(group_positions)
    all_group_attention_maps.append(group_attention_maps)
    all_group_attention_maps_normalized.append(group_attention_maps_normalized)
    all_feature_map_weights.append(feature_map_weights)

permutation = np.argsort(indices_track)

M = df.shape[0]

max_positions = torch.empty(4, M, 2)
attention_maps_normalized = torch.empty(4, M,14,14)
attention_maps_unnormalized = torch.empty(4, M,14,14)
feature_maps_weights = torch.empty(4, M, 512)

for i in range(4):
    max_positions[i] = torch.cat([pos[i] for pos in all_group_positions])[permutation]
    attention_maps_normalized[i] = torch.cat([att_maps[i] for att_maps in all_group_attention_maps_normalized])[permutation]
    attention_maps_unnormalized[i] = torch.cat([att_maps[i] for att_maps in all_group_attention_maps])[permutation]
    feature_maps_weights[i] = torch.cat([w[i] for w in all_feature_map_weights])[permutation]

torch.save(max_positions, 'max_positions.pt_p')

torch.save(attention_maps_normalized, 'attention_maps_normalized.pt_p')

torch.save(attention_maps_unnormalized, 'attention_maps_unnormalized.pt_p')

torch.save(feature_maps_weights, 'feature_maps_weights.pt_p')

