from model import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from dataloader import *
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

if torch.cuda.is_available():       
    device = torch.device("cuda:3")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(3))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
model=BERT().to(device)
loss_fn = nn.CrossEntropyLoss()

#Initialize Optimizer
optimizer= optim.Adam(model.parameters(),lr= 0.0001)

for param in model.bert_model.parameters():
    param.requires_grad = False

def finetune(epochs,dataloader,valid_loader, model,loss_fn,optimizer, ):
    counter = 0
    print_every = 5000
    model.train()
    for  epoch in range(epochs):
        print(epoch)
        
        loop=tqdm(enumerate(dataloader),leave=True,total=len(dataloader))
        for batch, dl in loop:
            counter += 1
            ids=dl['ids']
            token_type_ids=dl['token_type_ids']
            mask= dl['mask']
            label=dl['target']
            label = label.type(dtype=torch.LongTensor).to(device)#.unsqueeze(1)
            
            optimizer.zero_grad()
            
            output=model(
                ids=ids.to(device),
                mask=mask.to(device),
                token_type_ids=token_type_ids.to(device))
            label = label#.type_as(output)
            loss=loss_fn(output,label)
            loss.backward()
            optimizer.step()
            
           
            if counter % print_every == 0:
                # Get validation loss
                val_losses = []
                model.eval()
                val_loop=tqdm(enumerate(valid_loader),leave=False,total=len(valid_loader))
                
                for batch, dl in val_loop:
                    ids=dl['ids']
                    token_type_ids=dl['token_type_ids']
                    mask= dl['mask']
                    label=dl['target']
                    label = label.type(dtype=torch.LongTensor).to(device)#.unsqueeze(1)
                    
                    optimizer.zero_grad()
                    
                    output=model(
                        ids=ids.to(device),
                        mask=mask.to(device),
                        token_type_ids=token_type_ids.to(device))
                    label = label#.type_as(output)
                    val_loss=loss_fn(output,label)
                    val_losses.append(val_loss.item())
                    # pred = np.where(output >= 0.5, 1, 0)
                    # num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
                    # num_samples = pred.shape[0]
                    # accuracy = num_correct/num_samples
                print("The validation loss is: ",np.mean(val_losses))
        PATH = "../../../../../../data/spand43/unt/BERT_Quora.pth"
        torch.save(model.state_dict(), PATH)    
            # Show progress while training
            # loop.set_description(f'Epoch={epoch}/{epochs}')
            # loop.set_postfix(loss=loss.item(),acc=accuracy)
    return model
PATH = "../../../../../../data/spand43/unt/BERT_Quora.pth"
# model=finetune(10, train_loader,valid_loader, model, loss_fn, optimizer)
# torch.save(model.state_dict(), PATH)
# model = CNN_NLP().to(device)
model.load_state_dict(torch.load(PATH))

# pred = np.where(output >= 0.5, 1, 0)
test_losses = [] # track loss
num_correct = 0
model.eval()
total_num = 0
y_true,y_pred = [], []
# iterate over test data
test_loop=tqdm(enumerate(test_loader),leave=True,total=len(test_loader))
print(test_loop)
prob_1 = []
for batch, dl in test_loop:
    
    ids=dl['ids']
    token_type_ids=dl['token_type_ids']
    mask= dl['mask']
    label=dl['target']
    label = label.type(dtype=torch.LongTensor).to(device)#.unsqueeze(1)
    output = model(
                ids=ids.to(device),
                mask=mask.to(device),
                token_type_ids=token_type_ids.to(device))
    label = label#.type_as(output)
    test_loss=loss_fn(output,label)
    test_losses.append(test_loss)
    pred = F.softmax(output,dim=-1)
    prob_1+= pred.data.tolist()
    pred = torch.argmax(pred,dim=-1)
    y_pred += pred.data.tolist()
    y_true += label.data.tolist()
    # num_correct = sum(1 for a, b in zip(pred, label) if a[0] == b[0])
    # num_samples = pred.shape[0]

    
accuracy = accuracy_score(y_true, y_pred) * 100.
f1 = f1_score(y_true, y_pred, average='macro') * 100.
#################################################################
ns_probs = [0 for _ in range(len(y_true))]
ns_auc = metrics.roc_auc_score(y_true, ns_probs)
ns_fpr, ns_tpr, _ = metrics.roc_curve(y_true, ns_probs)
ps_auc = metrics.roc_auc_score(y_true, np.array(prob_1)[:,1])
print(ps_auc)
fpr, tpr, _ = metrics.roc_curve(y_true,  np.array(prob_1)[:,1])

#create ROC curve
plt.plot(fpr,tpr)
plt.plot(ns_fpr,ns_tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig('bert_roc.png')
##################################################################
results_dict = {
    'accuracy': accuracy_score(y_true, y_pred) * 100.,
    'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
    'Confusion Matrix': confusion_matrix(y_true, y_pred)
}
for k, v in results_dict.items():
    print(f'{k} = {v}')