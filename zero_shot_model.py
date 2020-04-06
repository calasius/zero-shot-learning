from torch.utils.data import Dataset, DataLoader
import math
from PIL import Image
from PIL import ImageFile
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import urllib
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from torch.distributions import MultivariateNormal
from torch.nn import *
from torch import nn
from utils import MultitagDataset
from torch import optim
from sklearn.metrics import balanced_accuracy_score
from torch.autograd import Variable
import torch
from torchvision import datasets, transforms, models

FEATURE_MAP_SIZE = 14
FEATURE_MAPS = 512
BATCH_SIZE = 1
K_SIGMOIDE_CROP_MASK = 10
IMAGE_SIZE = 448
CROP_ENLARGED_SIZE = 224
IMAGE_EMBEDDING_SIZE = 512
SEMANTIC_EMBEDDING_SIZE = 300
SIGMA = 3
SIGMA_MATRIX = (torch.eye(2) * SIGMA)
REDUCTION_SIZE = 64
DEFAULT_BOX_SIZE = 200
import os


class ZeroshotModel(nn.Module):
    
    @staticmethod
    def load_model(device, name):
        model = AttentionModel(device)
        
        for k in model.keys:
            model.feature_map_attention_network[k].load_state_dict(torch.load(name + '/' + 'feature_map_attention_network_' + k))
            model.crop_networks[k].load_state_dict(torch.load(name + '/' + 'crop_networks_' + k))
            model.projection_network[k].load_state_dict(torch.load(name + '/' + 'dimension_reduction_networks_' +k))
        
        return model
    
    def __init__(self, device, load_pretrained_networks = False):
        
        super().__init__()
        
        self.device = device
        
        self.backbone_model = models.vgg19(pretrained=True).features.to(device).eval()
        
        self.freeze_backbone_models_parameters(self.backbone_model)
        
        self.feature_map_attention_network = self.build_feature_map_attention_network(FEATURE_MAPS)
        
        self.crop_networks = self.build_crop_network()
        
        self.projection_network = self.build_projection_network()
        
        self.keys  = ['att1', 'att2', 'att3', 'att4']
        
        if load_pretrained_networks:
            for k in self.keys:
                crop_network = torch.load('crop_network_%s' % k)
                feature_map_attention_network = torch.load('feature_map_attention_network_%s' % k)
                self.crop_networks[k].load_state_dict(crop_network)
                self.feature_map_attention_network[k].load_state_dict(feature_map_attention_network)
                
    def save_model(self, name):
        if not os.path.exists(name):
            os.mkdir(name)
        
        for k in self.feature_map_attention_network:
            torch.save(self.feature_map_attention_network[k].state_dict(), name + '/' + 'feature_map_attention_network_' + k)
            torch.save(self.crop_networks[k].state_dict(), name + '/' + 'crop_networks_' + k)
            torch.save(self.projection_network[k].state_dict(), name + '/' + 'projection_network_' +k)
        
    def eval_mode(self):
        
        for k in self.feature_map_attention_network:
            self.feature_map_attention_network[k].eval()
            self.crop_networks[k].eval()
            self.dimension_reduction_networks[k].eval()
        
    def train_mode(self):
        
        for k in self.feature_map_attention_network:
            self.feature_map_attention_network[k].train()
            self.crop_networks[k].train()
            self.dimension_reduction_networks[k].train()
        
    def build_projection_network(self):
        
        W1 = torch.nn.Linear(IMAGE_EMBEDDING_SIZE, SEMANTIC_EMBEDDING_SIZE, bias = False).to(self.device)
        W2 = torch.nn.Linear(IMAGE_EMBEDDING_SIZE, SEMANTIC_EMBEDDING_SIZE, bias = False).to(self.device)
        W3 = torch.nn.Linear(IMAGE_EMBEDDING_SIZE, SEMANTIC_EMBEDDING_SIZE, bias = False).to(self.device)
        W4 = torch.nn.Linear(IMAGE_EMBEDDING_SIZE, SEMANTIC_EMBEDDING_SIZE, bias = False).to(self.device)
        
        projection_network = {}
        projection_network['att1'] = W1
        projection_network['att2'] = W2
        projection_network['att3'] = W3
        projection_network['att4'] = W4
        
        return projection_network
        
    
    def freeze_backbone_models_parameters(self,model):
        for param in model.parameters():
            param.required_grad = False
        
        
    def build_crop_network(self):
        CROP_SUBNET_INPUT_SIZE = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE
        
        crop_subnet_1 = nn.Sequential(
            nn.Linear(CROP_SUBNET_INPUT_SIZE, CROP_SUBNET_INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(CROP_SUBNET_INPUT_SIZE, CROP_SUBNET_INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(CROP_SUBNET_INPUT_SIZE, 3)
        ).to(self.device)
        
        crop_subnet_2 = nn.Sequential(
            nn.Linear(CROP_SUBNET_INPUT_SIZE, CROP_SUBNET_INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(CROP_SUBNET_INPUT_SIZE, CROP_SUBNET_INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(CROP_SUBNET_INPUT_SIZE, 3)
        ).to(self.device)
        
        crop_subnet_3 = nn.Sequential(
            nn.Linear(CROP_SUBNET_INPUT_SIZE, CROP_SUBNET_INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(CROP_SUBNET_INPUT_SIZE, CROP_SUBNET_INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(CROP_SUBNET_INPUT_SIZE, 3)
        ).to(self.device)
        
        crop_subnet_4 = nn.Sequential(
            nn.Linear(CROP_SUBNET_INPUT_SIZE, CROP_SUBNET_INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(CROP_SUBNET_INPUT_SIZE, CROP_SUBNET_INPUT_SIZE),
            nn.ReLU(),
            nn.Linear(CROP_SUBNET_INPUT_SIZE, 3)
        ).to(self.device)
        
        crop_subnets = {}
        crop_subnets['att1'] = crop_subnet_1
        crop_subnets['att2'] = crop_subnet_2
        crop_subnets['att3'] = crop_subnet_3
        crop_subnets['att4'] = crop_subnet_4
        
        return crop_subnets
        
    def build_feature_map_attention_network(self, n_features):
        
        fc1 = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.Sigmoid()
        ).to(self.device)
        
        fc2 = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.Sigmoid()
        ).to(self.device)
        
        fc3 = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.Sigmoid()
        ).to(self.device)
        
        fc4 = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, n_features),
            nn.Sigmoid()
        ).to(self.device)
        
        feature_map_attention_network ={}
        feature_map_attention_network['att1'] = fc1
        feature_map_attention_network['att2'] = fc2
        feature_map_attention_network['att3'] = fc3
        feature_map_attention_network['att4'] = fc4
        
        return feature_map_attention_network
    
    def calculate_ideal_attention_maps(self, max_centers, sigma_matrix):
        with torch.no_grad():
            ideal_attention_maps = {}
            for k in max_centers:
                centers = max_centers[k].cpu()
                ideal_attention_map = torch.empty(centers.shape[0], FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)
                
                for i in range(centers.shape[0]):
                    center = centers[i]
                    x = torch.arange(FEATURE_MAP_SIZE)
                    y = torch.arange(FEATURE_MAP_SIZE)
                    yy, xx = torch.meshgrid([x,y])
                    points = torch.as_tensor(torch.stack((xx, yy), dim=-1), dtype=torch.float32, device='cpu')
                    gaussian_multi = MultivariateNormal(center, sigma_matrix)
                    ideal_attention_map[i] = torch.exp(gaussian_multi.log_prob(points))
                    
                ideal_attention_maps[k] = ideal_attention_map.to(self.device)
            return ideal_attention_maps
    
    def calculate_max_centers(self, attention_maps):
        
        with torch.no_grad():
            max_centers = {}
            for k in attention_maps:
                attention_maps_plain = attention_maps[k].view(attention_maps[k].shape[0], -1)
                max_indices = torch.argmax(attention_maps_plain, dim=1)
                y = max_indices / FEATURE_MAP_SIZE
                x = max_indices % FEATURE_MAP_SIZE
                max_centers[k] = torch.stack([x,y], dim = 1)
            return max_centers
        
    def compactness_loss(self, attention_maps, ideal_attention_maps):
        res = torch.zeros(BATCH_SIZE).to(self.device)
        for k in attention_maps:
            res += ((attention_maps[k] - ideal_attention_maps[k])**2).mean()
        return res
    
    def diversity_loss(self, attention_maps, mrg):
        res = torch.zeros(BATCH_SIZE).to(self.device)
        for k in attention_maps:
            res += self.diversity_loss_by_key(k, attention_maps, mrg)
        return res.mean()
    
    def diversity_loss_by_key(self, current_key, attention_maps, margin = 2):
        keys = list(attention_maps.keys())
        keys.remove(current_key)
        
        reminder = torch.empty((attention_maps['att1'].shape[0], len(keys), FEATURE_MAP_SIZE, FEATURE_MAP_SIZE)).to(self.device)

        for i in range(len(keys)):
            reminder[:,i] = attention_maps[keys[i]]

        current_attantion_map = attention_maps[current_key]
        
        res = current_attantion_map * torch.max(torch.tensor(0.0).to(self.device), torch.max(reminder,dim=1)[0] - margin)

        return torch.sum(res, (1,2)).mean()
    
    def multi_attention_loss(self, attention_maps, ideal_attention_maps, margin, balance_factor):
        return self.compactness_loss(attention_maps, ideal_attention_maps) + balance_factor * self.diversity_loss(attention_maps, margin)
        #return self.compactness_loss(attention_maps, ideal_attention_maps)
        #return self.diversity_loss(attention_maps, margin)
    
    def take_crops(self,boxes, images):
        batch_size = boxes.shape[0]
        grid_tensor = torch.empty((batch_size, CROP_ENLARGED_SIZE, CROP_ENLARGED_SIZE, 2), requires_grad = True).to(self.device)
        M = boxes.shape[0]
        for i in range(M):
            box = boxes[i]
            ox = box[0] -224
            oy = -box[1] + 224
            l = box[2]
            
            p = l / CROP_ENLARGED_SIZE
            x = ((torch.linspace(-1,1, CROP_ENLARGED_SIZE).to(self.device) * (l/2)) - oy) / (IMAGE_SIZE/2)
            y = ((torch.linspace(-1,1, CROP_ENLARGED_SIZE).to(self.device) * (l/2)) + ox) / (IMAGE_SIZE/2)
            yy, xx = torch.meshgrid([x,y])
            yy.to(self.device)
            xx.to(self.device)

            points = torch.stack((xx, yy), dim=-1).to(self.device)
            
            grid_tensor[i] = points
            
            
        return torch.nn.functional.grid_sample(images, grid_tensor)
    
    def crop_network_forward(self, x, idx):
        attention_maps = x
        
        crop_network = self.crop_networks[self.keys[idx]]
        
        return crop_network(attention_maps)
    
    def feature_map_attention_network_forward(self, x, idx):
        
        net = self.feature_map_attention_network[self.keys[idx]]
        
        return net(x)
    
    def get_feature_map_attention_network_parameters(self, idx):
        return self.feature_map_attention_network[self.keys[idx]].parameters()
        
    
    def get_crop_network_parameters(self, idx):
        return self.crop_networks[self.keys[idx]].parameters()
    
    def save_crop_network(self):
        for k in self.keys:
            torch.save(self.crop_networks[k].state_dict(), 'crop_network_%s' % k)
            
    def save_feature_map_attention_network(self):
        for k in self.keys:
            torch.save(self.feature_map_attention_network[k].state_dict(), 'feature_map_attention_network_%s' % k)
            
    def get_model_parameters(self):
        parameters = list()
        for k in self.feature_map_attention_network:
            parameters += list(self.feature_map_attention_network[k].parameters())
            parameters += list(self.crop_networks[k].parameters())
            parameters += list(self.projection_network[k].parameters())
            
        return parameters
    
    def get_crops_forward(self, x):
        
        #Calculo los feature maps con la red base
        with torch.no_grad():
            feature_maps = self.backbone_model(x)
        
            # X tiene dimension (BATCH_SIZE, 512, 14, 14), hay que hacer un average pooling
            feature_map_avg_pool = F.avg_pool2d(feature_maps, FEATURE_MAP_SIZE)


            #Calculo la atencion de cada filtro para cada parte
            feature_map_attention_network_outputs = {}
            for k in self.feature_map_attention_network:
                out = self.feature_map_attention_network[k](feature_map_avg_pool.view(feature_map_avg_pool.shape[0],-1))
                feature_map_attention_network_outputs[k] = out


            #Calculo los attention maps
            attention_maps = {}
            for k in feature_map_attention_network_outputs:

                attention_weights = feature_map_attention_network_outputs[k]

                m = torch.sum(feature_maps * attention_weights.view(attention_weights.shape[0],FEATURE_MAPS,1,1), dim=1)
                attention_maps[k] = m


            max_centers = self.calculate_max_centers(attention_maps)

            ideal_attention_maps = self.calculate_ideal_attention_maps(max_centers, SIGMA_MATRIX)


            attention_crops = {}
            att_boxes = {}
            for k in self.crop_networks:
                crop_network = self.crop_networks[k]
                attention_maps_part = attention_maps[k]
                boxes = crop_network(attention_maps_part.view(attention_maps_part.shape[0],-1))
                crops = self.take_crops(boxes, x)
                attention_crops[k] = crops
                att_boxes[k] = boxes
                
            return attention_maps, attention_crops, ideal_attention_maps, max_centers, att_boxes
            
            
    
    def forward(self, x, eval_mode=False):
        
        #Calculo los feature maps con la red base
        with torch.no_grad():
            feature_maps = self.backbone_model(x)
        
        # X tiene dimension (BATCH_SIZE, 512, 14, 14), hay que hacer un average pooling
        feature_map_avg_pool = F.avg_pool2d(feature_maps, FEATURE_MAP_SIZE)
        
        features = feature_map_avg_pool.view(feature_map_avg_pool.shape[0],-1)
        
        #Calculo la atencion de cada filtro para cada parte
        feature_map_attention_network_outputs = {}
        for k in self.feature_map_attention_network:
            out = self.feature_map_attention_network[k](features)
            feature_map_attention_network_outputs[k] = out
            
        
        #Calculo los attention maps
        attention_maps = {}
        for k in feature_map_attention_network_outputs:
            
            attention_weights = feature_map_attention_network_outputs[k]
            
            m = torch.sum(feature_maps * attention_weights.view(attention_weights.shape[0],FEATURE_MAPS,1,1), dim=1)
            attention_maps[k] = m
            
            
        max_centers = self.calculate_max_centers(attention_maps)
        
        ideal_attention_maps = self.calculate_ideal_attention_maps(max_centers, SIGMA_MATRIX)
        
        multi_attention_loss = self.multi_attention_loss(attention_maps, ideal_attention_maps, 0.2, 1)
        
        projections = torch.empty((4, x.shape[0],SEMANTIC_EMBEDDING_SIZE),  requires_grad=True ).to(self.device)
        i = 0
        for k in self.crop_networks:
            crop_network = self.crop_networks[k]
            attention_maps_part = attention_maps[k]
            boxes = crop_network(attention_maps_part.view(attention_maps_part.shape[0],-1))
            crops = self.take_crops(boxes, x)
            projection_matrix = self.projection_network[k]
            
            if eval_mode:
                with torch.no_grad():
                    crop_feature_maps = F.avg_pool2d(self.backbone_model(crops),7)
                    
            else:
                crop_feature_maps = F.avg_pool2d(self.backbone_model(crops),7)
                
            crop_embeddings = crop_feature_maps.view(crop_feature_maps.shape[0],-1)
            
            projection = projection_matrix(crop_embeddings)
            
            projections[i] = projection
            
            i = i + 1
            
        return projections
