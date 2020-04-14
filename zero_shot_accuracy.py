import pandas as pd
from utils import *
from zero_shot_model import *
from PIL import Image
from PIL import ImageFile



device = 'cuda:0'

model = ZeroshotModel.load_model(device, 'models/zeroshot_model_0.85/')

embedding_matrix = get_category_embedding_matrix().to(device)

semantic_embeddings = pd.read_pickle('semantic_embeddings.pkl')
categories = semantic_embeddings.Class.values
mask = np.isin(categories, test_categories)
mask = np.argwhere(mask)

df = pd.read_pickle('tags_onehot_corrected.pkl')
    
files = pd.read_pickle('good_files.pkl')[0].values

df = df[df.file.isin(files)]
    
indices = []
for i in range(df.shape[0]):
    row = df.iloc[i,:-1].values
    if np.sum(row) == 1:
        indices.append(i)
            
unitag = df.iloc[indices]

def accuracy(category):
    
    samples = unitag[unitag[category] == 1].file.values
    
    results = [0]*len(categories)
    
    
    for i in range(samples.shape[0]):
        
        img = Image.open('/data/hotel_images/'+samples[i])
        
        img_tensor = image_transformations_with_normalization(img).repeat(1,1,1,1)
    
        p , _ = model(img_tensor.to(device))
        
        sum_logits = torch.zeros(p.shape[1], embedding_matrix.shape[1]).to(device)
        for i in range(4):
            sum_logits += torch.mm(p[i], embedding_matrix.to(device))

        sum_logits = torch.nn.functional.softmax(sum_logits, dim = 1)

        #sum_logits = sum_logits.T[mask.ravel()].T

        indices_pred = torch.topk(sum_logits, k = 1, dim = 1)[1].detach().cpu().numpy().squeeze()
        
        results[indices_pred] += 1
        
        #cat = categories[indices_pred][0]
        
        #if cat  in results:
        #    results[cat] = results[cat] + 1
        #else:
        #    results[cat] = 1
            
        
    return results


