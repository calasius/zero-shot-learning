from utils import *
from zero_shot_model import *

device = 'cuda:0'

batch_size = 16

model = ZeroshotModel(device, load_pretrained_networks=True)

embedding_matrix = get_category_embedding_matrix().to(device)

train_data = get_zeroshot_dataset(True)

test_data = get_zeroshot_dataset(False)

train_dataset = ZeroshotDataset('/data/hotel_images/', train_data)

train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=False)

test_dataset = ZeroshotDataset('/data/hotel_images/', test_data)

test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

def embedding_sofmax_loss(projections, labels, embedding_matrix):
    
    sum_logits = torch.zeros(projections.shape[1], embedding_matrix.shape[1]).to(device)
    
    print(sum_logits.shape)
    
    for i in range(4):
            
        sum_logits += torch.mm(projections[i], embedding_matrix)
        
        print(labels.shape)
    torch.nn.functional.cross_entropy(sum_logits, labels.long())


optimizer = torch.optim.Adam(model.get_model_parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0, amsgrad=False)

epochs = 5

for epoch in range(epochs):
    
    for images, labels in train_dataloader:
        
        optimizer.zero_grad()
        
        projections = model(images.to(device))
        
        print(projections.shape)
        
        loss = embedding_sofmax_loss(projections, labels.to(device), embedding_matrix)
        
        loss.backward()
        
        print(loss.item())
        
        optimizer.step()
        
        model.eval_mode()
        
        with torch.no_grad():
        
            for images, labels in test_dataloader:
                
                projections = model(images.to(device))
                
                logits = torch.mm(projections, embedding_matrix)
                
                pred, indices_pred = torch.topk(1, logits)
                
                indices_true = torch.topk(1, labels.to(device))
                
                print(indices_true.shape)
                print(indices_pred.shape)




