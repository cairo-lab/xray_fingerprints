import glob
import os
import torch
import copy
from torch.autograd import Variable


import numpy as np
import pandas as pd
from PIL import Image

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms, models, datasets
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

METADATA_SUBSET_PATH= "/dartfs-hpc/rc/home/t/f006cht/code/torch_test/dicom_metadata_df.csv"
#METADATA_SUBSET_PATH = "/Users/franceskoback/Documents/research/pytorch_1/metadata_100subset_df.csv"


def get_manufacturer_labels(encoder, target_variable = "(0008, 0070) Manufacturer"):
    df = pd.read_csv(METADATA_SUBSET_PATH)
    df["id"] = df["id"].astype("str").str.zfill(8)
    df["code"] = encoder.fit_transform(df[target_variable])
    dictionary= {row["id"]: row["code"] for i, row in df.iterrows()}
    keys = dictionary.keys()
    values = dictionary.values()
    return dictionary

class CustomImageDataset(Dataset):
    def __init__(self):
        #self.img_dir = "/Users/franceskoback/Documents/research/pytorch_1/xray_subsets"
        self.img_dir = "/dartfs-hpc/rc/home/t/f006cht/scratch/OAI/processed_images/xrays/knee/BilatPAFixedFlex/224x224/no_dicom_proc/self_scaled/group_norm"
        self.images = glob.glob(os.path.join(self.img_dir, "*.npy"))
        self.le = preprocessing.LabelEncoder()
        self.label_map = get_manufacturer_labels(self.le)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.fromarray(np.load(img_path)).convert("RGB")
        image = transforms.ToTensor()(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        image=normalize(image)
        xray_id = os.path.basename(img_path).replace(".npy", "")
        label= self.label_map[xray_id]
        label= torch.as_tensor(label).long()       
        return (image, label)

def train_val_test_dataset(dataset, val_split=0.20):
    train_idx, rem_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

    Test_size=0.5 # split equally between validataion and test sets 
    val_idx, test_idx  = train_test_split(list(rem_idx), test_size=Test_size)

    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    return datasets

    #####
    import numpy as np

#img_array = np.load('/Users/franceskoback/Documents/research/pytorch_1/xray_subsets/00098404.npy')
img_array = np.load('/dartfs-hpc/rc/home/t/f006cht/scratch/OAI/processed_images/xrays/knee/BilatPAFixedFlex/224x224/no_dicom_proc/self_scaled/group_norm/00728803.npy')
from matplotlib import pyplot as plt
print(img_array)
image_ = Image.fromarray(img_array).convert("RGB")
image_ = transforms.ToTensor()(image_)
print(image_.shape)

plt.imshow(img_array, cmap='gray')
#plt.show()
#print(img_array.shape)

####
dataset = CustomImageDataset()
datasets = train_val_test_dataset(dataset)
print(len(datasets['train'].dataset)) #6
train_dataset= datasets['train']
val_dataset= datasets['val']
test_dataset= datasets['test']


train_loader = DataLoader(
    train_dataset, batch_size=10, shuffle=True
)
valid_loader = DataLoader(
    val_dataset, batch_size=10, shuffle=True
)
test_loader = DataLoader(
    test_dataset, batch_size=10, shuffle=True
)

print(len(train_loader.dataset)) #6 
len_train=len(datasets['train'])
len_val= len(datasets['val'])
len_test= len(datasets['test'])

print("Training length", len(train_loader))
print("Validation length", len(valid_loader))
print("Testing length", len(test_loader))
####

def Net(num_classes):
    model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    for params in model.parameters():
        params.requires_grad = True

    return model

params = {
    "model": "resnet18",
    #"device": "cuda",
    "lr": 0.001,
    "batch_size": 10, #64
    "num_workers": 1, #20
    "n_epochs": 50, #100
    "image_size": 224, 
    "in_channels": 2, #3
    "num_classes": 11, #12
    "device": "cpu"
}

model = Net(params['num_classes'])
model.eval().to(params["device"])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = params['lr'])
###
def train_one_epoch(epoch, model, loss_fn, optimizer, loader, device = "cpu"):
    #put model in training state
    model.train()
    train_loss=0.0
    epoch_loss=[]

    for batch_idx, (img,label) in enumerate(loader):  
        inputs = Variable(img.to(device).float(),requires_grad=True)
        labels = Variable(label.to(device).float())
        optimizer.zero_grad() # sets all grads to None 

        # forward + backward + optimize
        outputs = model(inputs)
        pred = torch.argmax(outputs, dim=1)
        pred= Variable(pred.to(device).float(),requires_grad=True)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()

        train_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
        print("train loss is")
        print(train_loss)
        if batch_idx % 10 == 0:
            loss, current = loss.item(), batch_idx * len(inputs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(loader):>5d}]")
    print('Epoch {} avg Training loss: {:.3f}'.format(epoch+1, train_loss))
    return model, train_loss


def test_one_epoch(epoch, model, loss_fn, loader, len_val, device = "cpu"):
    model.eval()
    running_loss = 0
    actual_labels = []
    pred_labels = []
    loss_values=[]
    
    for batch_idx, (img,label) in enumerate(loader):    
        inputs =img
        labels = label
        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))
        
        log_preds = model(inputs)
        loss = loss_fn(log_preds, labels)

        preds = torch.exp(log_preds)
        running_loss+=((1 / (batch_idx + 1)) * (loss.data.item() - running_loss))
        
        #calculate accuracy
        top_prob, top_class = preds.topk(1, dim=1)
        pred_labels+= list((top_class.view(-1)).cpu().numpy())
        actual_labels+= list(labels.cpu().numpy())

    accuracy = ((np.array(pred_labels)==np.array(actual_labels)).sum())/len(pred_labels) #size of test set
    correct = ((np.array(pred_labels)==np.array(actual_labels)).sum())
    total = len(pred_labels)
    
    
    return running_loss, accuracy, correct, total

###
train_losses = []
valid_losses = []

for epoch in range(params['n_epochs']):
    model, train_loss = train_one_epoch(epoch, model, loss_fn, optimizer, train_loader)
    train_losses+= [train_loss]
    valid_loss, accuracy, correct, total = test_one_epoch(epoch, model, loss_fn, valid_loader, len_val)
    valid_losses+= [valid_loss]
    print('Epoch {} avg Valid loss: {:.3f}'.format(epoch+1, valid_loss))
    print('Epoch {} Valid accuracy: {:.1%} ({} of {} right)\n'.format(epoch+1, accuracy, correct, total))
    if len(valid_losses)>1 and (valid_loss<min(valid_losses[:-1])):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,
            }, 'checkpoint.tar')
##


print(train_losses)
plt.plot(np.linspace(1, params['n_epochs'], params['n_epochs']).astype(int), train_losses)
plt.savefig('train_losses.png')

###
# load the model that got the best validation accuracy
checkpoint = torch.load('checkpoint.tar')
loaded_model = Net(params['num_classes'])
loaded_model.to(params["device"])
loaded_model.load_state_dict(checkpoint['model_state_dict'])

loaded_criterion = checkpoint['loss']


last_epoch = checkpoint['epoch']+1

test_loss, accuracy, correct, total = test_one_epoch(None, loaded_model, loaded_criterion, test_loader, len_val)

print('Test loss: {:.3f}'.format(test_loss))
print('Test accuracy: {:.1%} ({} of {} right)\n'.format(accuracy, correct, total))
##
