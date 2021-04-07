
# imports
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from os.path import join
import zipfile
from utils import extract, standardize
from datasets import SeismicDataset
from torch.utils.data import DataLoader
from model import Model

    
def preprocess(no_wells_marmousi, no_wells_seam):
    """Function initializes data, performs standardization, and train test split
    
    Parameters:
    ----------
    no_wells_marmousi : int,
        number of evenly spaced wells and seismic samples to be evenly sampled 
        from marmousi section.
        
    no_wells_seam : int
        number of evenly spaced wells and seismic samples to be evenly sampled from SEAM
        
    Returns
    -------
    seismic_marmousi : array_like, shape(num_traces, depth samples)
        2-D array containing seismic section for marmousi
        
    seismic_seam : array_like, shape(num_traces, depth samples)
        2-D array containing seismic section for SEAM
        
    model_marmousi : array_like, shape(num_wells, depth samples)
        2-D array containing model section from marmousi 2
        
    model_seam : array_like, shape(num_wells, depth samples)
        2-D array containing model section from SEAM
    
    """
    
    # get project root directory
    project_root = os.getcwd()
    
    if ~os.path.isdir('data'): # if data directory does not exists then extract
        extract('data.zip', project_root)
        
    
    # Load data
    seismic_marmousi = np.load(join('data','marmousi_synthetic_seismic.npy')).squeeze()
    seismic_seam = np.load(join('data','poststack_seam_seismic.npy')).squeeze()[:, 50:]
    seismic_seam = seismic_seam[::2, :]
    
    # Load targets and standardize data
    model_marmousi = np.load(join('data', 'marmousi_Ip_model.npy')).squeeze()[::5, ::4]
    model_seam = np.load(join('data','seam_elastic_model.npy'))[::3,:,::2][:, :, 50:]
    model_seam = model_seam[:,0,:] * model_seam[:,2,:]
    
    # standardize
    seismic_marmousi, model_marmousi = standardize(seismic_marmousi, model_marmousi, no_wells_marmousi)
    seismic_seam, model_seam = standardize(seismic_seam, model_seam, no_wells_seam)
    
    return seismic_marmousi, seismic_seam, model_marmousi, model_seam


def train(**kwargs):
    """Function trains 2-D TCN as specified in the paper"""
    
    # obtain data
    seismic_marmousi, seismic_seam, model_marmousi, model_seam = preprocess(50, 12)
    
    # specify width of seismic image samples around each pseudolog
    width = 7
    offset = int(width/2)
    
    # specify pseudolog positions for training and validation
    traces_marmousi_train = np.linspace(451, 2199, kwargs['no_wells_marmousi'], dtype=int)
    traces_seam_train = np.linspace(offset, len(model_seam)-offset-1, kwargs['no_wells_seam'], dtype=int)
    traces_seam_validation = np.linspace(offset, len(model_seam)-offset-1, 3, dtype=int)
    
    # set up dataloaders
    marmousi_dataset = SeismicDataset(seismic_marmousi, model_marmousi, traces_marmousi_train, width)
    marmousi_loader = DataLoader(marmousi_dataset, batch_size = 16)
    
    seam_train_dataset = SeismicDataset(seismic_seam, model_seam, traces_seam_train, width)
    seam_train_loader = DataLoader(seam_train_dataset, batch_size = len(seam_train_dataset))
    
    seam_val_dataset = SeismicDataset(seismic_seam, model_seam, traces_seam_validation, width)
    seam_val_loader = DataLoader(seam_val_dataset, batch_size = len(seam_val_dataset))
    
    
    # define device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up models
    model_marmousi = Model(1,1,[10, 30, 60, 90, 120], 9, 0.4).to(device)
    model_seam = Model(1,1,[10, 30, 60, 90, 120], 9, 0.4).to(device)
    
    # define weight sharing value
    gamma = torch.tensor([1e-4], requires_grad=True, dtype=torch.float, device=device)  # learnable weight for weight mismatch loss
    
    # Set up loss
    criterion = torch.nn.MSELoss()
    
    # Define Optimizer
    optimizer_marmousi = torch.optim.Adam(model_marmousi.parameters(),
                                     weight_decay=0.0001,
                                     lr=0.001)
    
    optimizer_seam = torch.optim.Adam(model_seam.parameters(),
                                     weight_decay=0.0001,
                                     lr=0.001)
    
    # start training 
    for epoch in range(900):
    
      loss1 = torch.tensor([0.0], requires_grad=True).float().cuda()
      model_marmousi.train()
      model_seam.train()
      optimizer_marmousi.zero_grad()
      optimizer_seam.zero_grad()
      
      for i, (x,y) in enumerate(marmousi_loader):  
    
        y_pred, x_hat = model_marmousi(x)
        loss1 += criterion(y_pred, y) + criterion(x_hat, x)
    
      loss1 = loss1/i  
      
      for x,y in seam_train_loader:
        y_pred, x_hat = model_seam(x)
        loss2 = criterion(y_pred, y) + criterion(x_hat, x)
    
      for x, y in seam_val_loader:
        model_seam.eval()
        y_pred, _ = model_seam(x)
        val_loss = criterion(y_pred, y)
        
      weight_mismatch_loss = 0
      for param1, param2 in zip(model_marmousi.parameters(), model_seam.parameters()):
        weight_mismatch_loss += torch.sum((param1-param2)**2)
    
      loss = loss1 + loss2 + gamma*weight_mismatch_loss  # original gamma val = 0.0001
      loss.backward()
      optimizer_marmousi.step()
      optimizer_seam.step()
      
      print('Epoch: {} | Marmousi Loss: {:0.4f} | Seam Loss: {:0.4f} | Val Loss: {:0.4f} | Mismatch Loss: {:0.4f} | Gamma: {:0.4f}'.format(epoch, loss1.item(), loss2.item(), val_loss.item(), weight_mismatch_loss.item(), gamma.item()))  
        


    
train(no_wells_marmousi=50, no_wells_seam=12)
    