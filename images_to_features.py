import pandas as pd
from utils import *
from zero_shot_model import *
from PIL import Image
from PIL import ImageFile




images_root_path = '/data/hotel_images/'
batch_size = 64
device = 'cuda:1'

tags = pd.read_pickle('multitag_dataset.pkl')

dataset = MultitagDataset(images_root_path, tags)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = models.vgg19(pretrained=True).features.eval().to(device)

feature_batches = []
for images, labels in dataloader:
    with torch.no_grad():
        features = model(images.to(device))
        feature_batches.append(features)

features = torch.cat(feature_batches)

torch.save(features, 'features.pt')