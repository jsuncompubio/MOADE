import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")



class simdatset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(device)
        y = torch.from_numpy(self.Y[index]).float().to(device)
        return x, y


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, input1_dim, input2_dim):
        super().__init__()
        self.name = 'ae'
        self.state = 'train' # or 'test'
        self.inputdim = input_dim
        self.outputdim = output_dim
        self.input1dim = input1_dim
        self.input2dim = input2_dim
        self.encoder = nn.Sequential(nn.Dropout(),
                                     nn.Linear(self.inputdim, 512),
                                     nn.CELU(),
                                     
                                     
                                     nn.Dropout(),
                                     nn.Linear(512, 256),
                                     nn.CELU(),
                                     
                                     
                                     nn.Dropout(),
                                     nn.Linear(256, 128),
                                     nn.CELU(),
                                     
                                     
                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),
                                     
                                     nn.Linear(64, output_dim),
                                     )


        self.s1 = nn.Linear(self.outputdim, self.outputdim, bias=False)
        self.s2 = nn.Linear(self.outputdim, self.outputdim, bias=False)
        

        self.decoder1 = nn.Sequential(nn.Linear(self.outputdim, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, self.input1dim, bias=False))


        self.decoder2 = nn.Sequential(nn.Linear(self.outputdim, 64, bias=False),
                                     nn.Linear(64, 128, bias=False),
                                     nn.Linear(128, 256, bias=False),
                                     nn.Linear(256, 512, bias=False),
                                     nn.Linear(512, self.input2dim, bias=False))

    def encode(self, x):
        return self.encoder(x)

    #def decode(self, z):
    #    return self.decoder(z)

    def refraction(self,x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        return x/x_sum
    
    def sigmatrix1(self):
        w0 = (self.decoder1[0].weight.T)
        w1 = (self.decoder1[1].weight.T)
        w2 = (self.decoder1[2].weight.T)
        w3 = (self.decoder1[3].weight.T)
        w4 = (self.decoder1[4].weight.T)
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return F.relu(w04)

    def sigmatrix2(self):
        w0 = (self.decoder2[0].weight.T)
        w1 = (self.decoder2[1].weight.T)
        w2 = (self.decoder2[2].weight.T)
        w3 = (self.decoder2[3].weight.T)
        w4 = (self.decoder2[4].weight.T)
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return F.relu(w04)

    def forward(self, x):
        
        sigmatrix1 = self.sigmatrix1()
        sigmatrix2 = self.sigmatrix2()
        p = self.encode(x)
        z1 = self.s1(p)
        z2 = self.s2(p)
        if self.state == 'train':
            pass
        elif self.state == 'test':
            z1 = F.relu(z1)
            z1 = self.refraction(z1)
            z2 = F.relu(z2)
            z2 = self.refraction(z2)
            
        x_recon1 = torch.mm(z1, sigmatrix1)
        x_recon2 = torch.mm(z2, sigmatrix2)
        return x_recon1, x_recon2, z1, z2, sigmatrix1, sigmatrix2


def reproducibility(seed=9):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


