from attention_model import *
from utils import *
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

multilabel_data = get_data_multilabel()

multilabel_data.shape

test_data = get_data(False)

device = 'cuda:1'

guestroom_model = AttentionModel.load_model(device, 'models/guest_room/')

bathroom_model = AttentionModel.load_model(device, 'models/bathroom/')

kitchen_model = AttentionModel.load_model(device, 'models/kitchen/')

livingroom_model = AttentionModel.load_model(device, 'models/living_room/')

beach_model = AttentionModel.load_model(device, 'models/beach/')

pool_model = AttentionModel.load_model(device, 'models/pool_view/')

hotel_front_model = AttentionModel.load_model(device, 'models/hotel_front/')

natural_view_model = AttentionModel.load_model(device, 'models/natural_view/')

models = {}
models['guest_room'] = guestroom_model
models['bathroom'] = bathroom_model
models['kitchen'] = kitchen_model
models['living_room'] = livingroom_model
models['beach'] = beach_model
models['pool_view'] = pool_model
models['hotel_front'] = hotel_front_model
models['natural_view'] = natural_view_model

label = 'bathroom'
batch_size = 32

test_dataset = MultitagDataset('/data/hotel_images/', test_data, label)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_dataset_without_normalization = MultitagDataset('/data/hotel_images/', multilabel_data, label, transform=image_transformations)
test_dataloader_without_normalization = DataLoader(test_dataset_without_normalization, batch_size = batch_size, shuffle = False)
multilabel_dataset = MultitagDataset('/data/hotel_images/', multilabel_data, label)
multilabel_dataloader = DataLoader(multilabel_dataset, batch_size = batch_size, shuffle = False)


with torch.no_grad():
        
    y_pred = {}
    for k in models.keys():
        y_pred[k] = []
        
    i = 0
    for images, labels in test_dataloader:
            
        print('model = %s, batch = %s'%(k, str(i)))
            
        for k in models.keys():
            model = models[k]
            model.eval_mode()
            
            probs, _ = model.forward(images.to(device), eval_mode=True)
            pred = probs > 0.5
            pred = pred.float().T.detach().cpu().numpy().ravel()
            y_pred[k].extend(pred.tolist())
                
        i += 1
            
            
for k in y_pred.keys():
    test_data['pred_' + k] = y_pred[k]
        
        
test_data.to_pickle('test_evaluation.pkl')