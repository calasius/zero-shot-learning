from utils import *
from zero_shot_model import *

device = 'cuda:1'

batch_size = 20

model = ZeroshotModel(device, load_pretrained_networks=True)

embedding_matrix = get_category_embedding_matrix().to(device)

train_data = get_zeroshot_dataset(True)

test_data = get_zeroshot_dataset(False)

train_dataset = ZeroshotDataset('/data/hotel_images/', train_data)

train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)

test_dataset = ZeroshotDataset('/data/hotel_images/', test_data)

test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

criterion = torch.nn.CrossEntropyLoss()

def embedding_sofmax_loss(projections, labels, embedding_matrix):
    
    sum_logits = torch.zeros(projections.shape[1], embedding_matrix.shape[1]).to(device)
    
        
    for i in range(4):    
        sum_logits += torch.mm(projections[i], embedding_matrix)
        
    target = torch.topk(labels.long(), k=1, dim=1)[1]

    
    return criterion(sum_logits, target.T[0])



def class_center_triplet_loss(centers, labels, projections, margin=0.8):
    
    p = torch.sum(projections, dim=0)
    
    p = p / p.norm(p=2, dim=1).view(-1,1)
    
    c = centers / (centers.norm(p=2, dim=1)).view(-1,1)
    
    one_positions = torch.argmax(labels, dim=1)
    
    center_distances = torch.empty(p.shape[0], centers.shape[0]).to(device)
    for i in range(centers.shape[0]):
        center_distances[:, i] = (p - c[i]).norm(p=2, dim=1)**2
        
    total_loss = torch.FloatTensor([0]).to(device)
    for i in range(center_distances.shape[0]):
        sample_distances = center_distances[i]
        idx = one_positions[i]
        center_class_distance = sample_distances[idx]
        
        sample_loss = torch.FloatTensor([0]).to(device)
        for j in range(sample_distances.shape[0]):
            if j != idx:
                sample_loss += torch.max(torch.FloatTensor([0]).to(device), margin + center_class_distance - sample_distances[j])
    
        total_loss += sample_loss
        
    return total_loss / projections.shape[0]


optimizer = torch.optim.Adam(model.get_model_parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0, amsgrad=False)

epochs = 5

for epoch in range(epochs):
    
    for images, labels in train_dataloader:
        
        optimizer.zero_grad()
        
        projections, multi_attention_loss, class_centers = model(images.to(device))
        
        loss = embedding_sofmax_loss(projections, labels.to(device), embedding_matrix) + class_center_triplet_loss(class_centers, labels.to(device), projections) + multi_attention_loss
        
        loss.backward()
        
        print('norma grad class_centers = %s' % model.class_centers.grad.norm(p=2))
        
        print('loss = %s' % loss.item())
    
        optimizer.step()
        
        model.eval_mode()
        
        with torch.no_grad():
        
            acc = 0
            for images, labels in test_dataloader:
                
                projections, _, _ = model(images.to(device))
                
                sum_logits = torch.zeros(projections.shape[1], embedding_matrix.shape[1]).to(device)


                for i in range(4):
                    sum_logits += torch.mm(projections[i], embedding_matrix)

                    
                sum_logits = torch.nn.functional.softmax(sum_logits, dim = 1)

                indices_pred = torch.topk(sum_logits, k = 2, dim = 1)[1].detach().cpu().numpy()
                
                indices_true = torch.topk(labels.to(device), k = 2)[1].detach().cpu().numpy()
                
                
                for i in range(indices_true.shape[0]):
                    intersection = set(indices_pred[i].tolist()).intersection(indices_true[i].tolist())
                    acc += 1 if len(intersection) > 0 else 0
                    
                
            acc /= len(test_dataset)
            
            print('acc = %s' % acc)
            
        model.train_mode()
        
        if acc > .4:
            print('saving model')
            model.save_model('models/zeroshot_model_centers_08_sun_attributes_%.2f' % acc)




