import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from VAE import VAE, calc_VAE_dims, vae_hf_loss_func

from VAE_test import Autoencoder


DATA_PATH = r'C:\Users\FS-Ma\OneDrive\Documents\projects\customer-segmentation-analysis\datasets\processed\yelp\test_struc_unstruc_with_embedding.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

df_base = pd.read_csv(DATA_PATH, sep=',')

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def load_and_standardize_data(path):
    # read in from csv
    df = pd.read_csv(path, sep=',')
    # replace nan with -99
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    # randomly split
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    # standardize values
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, X_test, scaler

class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(DATA_PATH)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len
    
traindata_set=DataBuilder(DATA_PATH, train=True)
testdata_set=DataBuilder(DATA_PATH, train=False)

trainloader=DataLoader(dataset=traindata_set,batch_size=1024)
testloader=DataLoader(dataset=testdata_set,batch_size=1024)

epochs = 1500
log_interval = 50
val_losses = []
train_losses = []
test_losses = []

class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
    
    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar 
    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return loss_MSE + loss_KLD
loss_mse = customLoss()

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(data)
        loss = loss_mse(x_recon, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        break
    if epoch % 200 == 0:        
        print('====> Epoch: {} Average training loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))
        train_losses.append(train_loss / len(trainloader.dataset))


from VAE import VAE, calc_VAE_dims
dims = calc_VAE_dims(df_base.shape[1])
print(df_base.shape[1])
print(dims)
dims = [793,1000,1000,300]
model = VAE(
    dims = dims, 
    dropout_prob = 0, #Don’t use Dropout layers in a VAE!
    act = "lrelu",
    return_layer_outs = False,
    latent_dim = 300,
    bn_enc = False,
    bn_dec = False
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#MUDAR BERTTOCLASSIFICATION TO BERT SELF SUPERVISED LEARNING


'''
D_in = df_base.shape[1]
H = 50
H2 = 12
model = Autoencoder(D_in, H, H2).to(device)
model.apply(weights_init_uniform_rule)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
'''

print(model)
for epoch in range(1, epochs + 1):
    train(epoch)

