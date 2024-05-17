import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import simdatset, AutoEncoder, device
from utils import showloss

def reproducibility(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def training_stage(model, train_loader, optimizer, epochs=128):
    
    model.train()
    model.state = 'train'
    loss = []
    recon_loss = []
    
    #for i in tqdm(range(epochs)):
    for i in range(epochs):
        for k, (data1, label1, data2, label2) in enumerate(train_loader):
            # reproducibility(seed=0)
            optimizer.zero_grad()
            data = torch.cat([data1, data2], dim=-1)
            x_recon1, x_recon2, cell_prop1, cell_prop2, sigm1, sigm2 = model(data)
            batch_loss = F.l1_loss(cell_prop1, label1) + F.l1_loss(x_recon1, data1) + F.l1_loss(cell_prop2, label2) + F.l1_loss(x_recon2, data2)
            batch_loss.backward()
            optimizer.step()
            loss.append(F.l1_loss(cell_prop1, label1).cpu().detach().numpy() + F.l1_loss(cell_prop2, label2).cpu().detach().numpy())
            recon_loss.append(F.l1_loss(x_recon1, data1).cpu().detach().numpy() + F.l1_loss(x_recon2, data2).cpu().detach().numpy())

    return model, loss, recon_loss

def adaptive_stage(model, data, data1, data2, optimizerD1, optimizerD2, optimizerE, optimizerS1, optimizerS2, step=10, max_iter=5):      # step=10, max_iter=5: default
    #data = torch.from_numpy(data).float().to(device)
    loss = []
    model.eval()
    model.state = 'test'
    _, _, ori_pred1, ori_pred2, ori_sigm1, ori_sigm2 = model(data)
    ori_sigm1 = ori_sigm1.detach()
    ori_pred1 = ori_pred1.detach()
    ori_sigm2 = ori_sigm2.detach()
    ori_pred2 = ori_pred2.detach()
    model.state = 'train'
    
    for k in range(max_iter):
        model.train()
        for i in range(step):
            reproducibility(seed=0)
            optimizerD1.zero_grad()
            optimizerD2.zero_grad()
            x_recon1, x_recon2, _, _, sigm1, sigm2 = model(data)
            batch_loss = F.l1_loss(x_recon1, data1) + F.l1_loss(sigm1, ori_sigm1) + F.l1_loss(x_recon2, data2) + F.l1_loss(sigm2, ori_sigm2)
            batch_loss.backward()
            optimizerD1.step()
            optimizerD2.step()
            loss.append(F.l1_loss(x_recon1, data1).cpu().detach().numpy() + F.l1_loss(x_recon2, data2).cpu().detach().numpy())

        for i in range(step):
            reproducibility(seed=0)
            optimizerE.zero_grad()
            optimizerS1.zero_grad()
            optimizerS2.zero_grad()
            x_recon1, x_recon2, pred1, pred2, _, _ = model(data)
            batch_loss = F.l1_loss(ori_pred1, pred1) + F.l1_loss(x_recon1, data1) + F.l1_loss(ori_pred2, pred2) + F.l1_loss(x_recon2, data2)
            batch_loss.backward()
            optimizerE.step()
            optimizerS1.step()
            optimizerS2.step()
            loss.append(F.l1_loss(x_recon1, data1).cpu().detach().numpy() + F.l1_loss(x_recon2, data2).cpu().detach().numpy())

    model.eval()
    model.state = 'test'
    _, _, pred1, pred2, sigm1, sigm2 = model(data)
    return sigm1.cpu().detach().numpy(), sigm2.cpu().detach().numpy(), loss, pred1.detach().cpu().numpy(), pred2.detach().cpu().numpy()

def train_model(train_x1, train_y1, train_x2, train_y2,
                model_name=None,
                batch_size=128, epochs=128):
    
    train_x1 = torch.from_numpy(train_x1).float().to(device)
    train_y1 = torch.from_numpy(train_y1).float().to(device)
    train_x2 = torch.from_numpy(train_x2).float().to(device)
    train_y2 = torch.from_numpy(train_y2).float().to(device)
    dataset = TensorDataset(train_x1, train_y1, train_x2, train_y2)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = AutoEncoder(train_x1.shape[1] + train_x2.shape[1], train_y1.shape[1], train_x1.shape[1], train_x2.shape[1]).to(device)
    # reproducibility(seed=0)
    optimizer = Adam(model.parameters(), lr=1e-4)
    print('Start training')
    model, loss, reconloss = training_stage(model, train_loader, optimizer, epochs=epochs)
    print('Training is done')
    print('prediction loss is:')
    showloss(loss, 'prediction_loss')
    print('reconstruction loss is:')
    showloss(reconloss, 'reconstruction_loss')
    if model_name is not None:
        print('Model is saved')
        torch.save(model, model_name+".pth")
    return model

def predict(test_x1, test_x2, genename1, genename2, celltypes, samplename,
            model_name=None, model=None,
            adaptive=True, mode='overall'):
    
    test_x1 = torch.from_numpy(test_x1).float().to(device)
    test_x2 = torch.from_numpy(test_x2).float().to(device)
    test_x = torch.cat([test_x1, test_x2], dim=-1)
    
    if model is not None and model_name is None:
        print('Model is saved without defined name')
        torch.save(model, 'model.pth')
    if adaptive is True:
        if mode == 'high-resolution':
            TestSigmList1 = np.zeros((test_x1.shape[0], len(celltypes), len(genename1)))
            TestPred1 = np.zeros((test_x1.shape[0], len(celltypes)))
            TestSigmList2 = np.zeros((test_x2.shape[0], len(celltypes), len(genename2)))
            TestPred2 = np.zeros((test_x2.shape[0], len(celltypes)))
            print('Start adaptive training at high-resolution')
            
            #for i in tqdm(range(len(test_x))):
            for i in range(len(test_x)):
                x = test_x[i,:].reshape(1,-1)
                x1 = test_x1[i,:].reshape(1,-1)
                x2 = test_x2[i,:].reshape(1,-1)
                if model_name is not None and model is None:
                    model = torch.load(model_name + ".pth")
                elif model is not None and model_name is None:
                    model = torch.load("model.pth")
                decoder1_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder1' in n]}]
                decoder2_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder2' in n]}]
                encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
                s1_parameters = [{'params': [p for n, p in model.named_parameters() if 's1' in n]}]
                s2_parameters = [{'params': [p for n, p in model.named_parameters() if 's2' in n]}]
                optimizerD1 = torch.optim.Adam(decoder1_parameters, lr=1e-4)
                optimizerD2 = torch.optim.Adam(decoder2_parameters, lr=1e-4)
                optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
                optimizerS1 = torch.optim.Adam(s1_parameters, lr=1e-3)
                optimizerS2 = torch.optim.Adam(s2_parameters, lr=1e-3)
                test_sigm1, test_sigm2, loss, test_pred1, test_pred2 = adaptive_stage(model, x, x1, x2, optimizerD1, optimizerD2, optimizerE, optimizerS1, optimizerS2, step=100, max_iter=3)
                TestSigmList1[i, :, :] = test_sigm1
                TestPred1[i,:] = test_pred1
                TestSigmList2[i, :, :] = test_sigm2
                TestPred2[i,:] = test_pred2
            TestPred1 = pd.DataFrame(TestPred1, columns=celltypes, index=samplename)
            TestPred2 = pd.DataFrame(TestPred2, columns=celltypes, index=samplename)
            CellTypeSigm1 = {}
            for i in range(len(celltypes)):
                cellname = celltypes[i]
                sigm = TestSigmList1[:,i,:]
                sigm = pd.DataFrame(sigm,columns=genename1,index=samplename)
                CellTypeSigm1[cellname] = sigm
            CellTypeSigm2 = {}
            for i in range(len(celltypes)):
                cellname = celltypes[i]
                sigm = TestSigmList2[:,i,:]
                sigm = pd.DataFrame(sigm,columns=genename2,index=samplename)
                CellTypeSigm2[cellname] = sigm
            print('Adaptive stage is done')

            return CellTypeSigm1, TestPred1, CellTypeSigm2, TestPred2

        elif mode == 'overall':
            if model_name is not None and model is None:
                model = torch.load(model_name + ".pth")
            elif model is not None and model_name is None:
                model = torch.load("model.pth")
            decoder1_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder1' in n]}]
            decoder2_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder2' in n]}]
            encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
            optimizerD1 = torch.optim.Adam(decoder1_parameters, lr=1e-4)
            optimizerD2 = torch.optim.Adam(decoder2_parameters, lr=1e-4)
            optimizerE = torch.optim.Adam(encoder_parameters, lr=1e-4)
            print('Start adaptive training for all the samples')
            test_sigm1, test_sigm2, loss, test_pred1, test_pred2 = adaptive_stage(model, test_x, test_x1, test_x2, optimizerD1, optimizerD2, optimizerE, step=300, max_iter=3)
            print('Adaptive stage is done')
            test_sigm1 = pd.DataFrame(test_sigm1,columns=genename1,index=celltypes)
            test_pred1 = pd.DataFrame(test_pred1,columns=celltypes,index=samplename)
            test_sigm2 = pd.DataFrame(test_sigm2,columns=genename2,index=celltypes)
            test_pred2 = pd.DataFrame(test_pred2,columns=celltypes,index=samplename)

            return test_sigm1, test_pred1, test_sigm2, test_pred2

    else:
        if model_name is not None and model is None:
            model = torch.load(model_name+".pth")
        elif model is not None and model_name is None:
            model = model
        print('Predict cell fractions without adaptive training')
        model.eval()
        model.state = 'test'
        #data = torch.from_numpy(test_x).float().to(device)
        _, _, pred1, pred2, _, _ = model(test_x)
        pred1 = pred1.cpu().detach().numpy()
        pred2 = pred2.cpu().detach().numpy()
        pred1 = pd.DataFrame(pred1, columns=celltypes, index=samplename)
        pred2 = pd.DataFrame(pred2, columns=celltypes, index=samplename)
        print('Prediction is done')
        return pred1, pred2




