# import all the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import wave

import torch
from torch import autograd
from torchsummary import summary
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

#from sklearn.metrics import classification_report

from torchmetrics.functional.classification import binary_accuracy
from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.functional.classification import binary_f1_score

#define the required hyperparameter
num_classes = 8
learning_rate=0.00005
num_epochs = 1000

#define the spectrograms
sample_rate = 16000

#Define the feature to be extracted

def gccphat(sig, refsig, fs=sample_rate, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0] + 10
    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n= n)
    REFSIG = np.fft.rfft(refsig, n= len(refsig))
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc


#Define the models
def target(BF):
  y = None
  phi = None
  if '@0' in BF:
    phi = 0
    y = torch.FloatTensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  elif '@45' in BF:
    phi = 45
    y = torch.FloatTensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  elif '@90' in BF:
    phi = 90
    y = torch.FloatTensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  elif '@135' in BF:
    phi = 135
    y = torch.FloatTensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
  elif '@180' in BF:
    phi = 180
    y = torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  elif '@225' in BF:
    phi = 225
    y = torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
  elif '@270' in BF:
    phi = 270
    y = torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
  elif '@315' in BF:
    phi = 315
    y = torch.FloatTensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
  return y, phi

def inside(x1_spec, x2_spec, x3_spec, x4_spec):
  x12, ccx12 = gccphat(x1_spec, x2_spec)
  x13, ccx13 = gccphat(x1_spec, x3_spec)
  x14, ccx14 = gccphat(x1_spec, x4_spec)
  x23, ccx23 = gccphat(x2_spec, x3_spec)
  x24, ccx24 = gccphat(x2_spec, x4_spec)
  x34, ccx34 = gccphat(x3_spec, x4_spec)
  #x_cat = torch.FloatTensor([xxp, xxn, xyp, xyn, x21, x41, x32, x12, x43, x23, x14, x34])
  x_cat = torch.FloatTensor([x12*1000, x13*1000, x14*1000, x23*1000, x24*1000, x34*1000])

  #x_cat = torch.nn.functional.normalize((x_cat), dim = 0)
  return x_cat

#model1 = nn.Sequential(
#    nn.Linear(12, 50),
#    nn.Tanh(),
#    nn.Linear(50, 25),
#    nn.Tanh(),
#    nn.Linear(25, 8),
#    nn.Sigmoid()
#)
 
def direction_phi(y_cap):
  phi_cap =torch.argmax(y_cap)*45
  return phi_cap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = 'F:\\Drive D\\Splitted_RS_Mic\\train'
val_dir = 'F:\\Drive D\\Splitted_RS_Mic\\validate'
test_dir = 'F:\\Drive D\\Splitted_RS_Mic\\test'
 
# Building Our Mode
class Network(nn.Module):
    # Declaring the Architecture
    def __init__(self):
        super(Network,self).__init__()
        self.fc1 = nn.Linear(6, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 8)
 
    # Forward Pass
    def forward(self, x):
        #x = x.view(x.shape[0],-1)    # Flatten the images
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
model = Network()
model = model.to(device)
 
# Declaring Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
 

train = os.listdir(train_dir)
val = os.listdir(val_dir)


# Training with Validation
epochs = 10
min_valid_loss = np.inf

for e in range(epochs):
    train_loss = 0.0
    for file in train:
        # Transfer Data to GPU if available
        if file.endswith("BF.wav"):
            x1,fs1 = torchaudio.load(f"{train_dir}\\{file[0:-6]}1.wav")
            x2,fs2 = torchaudio.load(f"{train_dir}\\{file[0:-6]}2.wav")
            x3,fs3 = torchaudio.load(f"{train_dir}\\{file[0:-6]}3.wav")
            x4,fs4 = torchaudio.load(f"{train_dir}\\{file[0:-6]}4.wav")
            data = inside(x1, x2, x3, x4)
            data = data.to(device)
            
            labels, phi = target(file)
            #print(labels)
            labels  = labels.to(device)

            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            out = model(data)
            print(out)
            # Find the Loss
            loss = criterion(out,labels)
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()
     
    valid_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in validloader:
        # Transfer Data to GPU if available
        for file in val:
            if file.endswith("BF.wav"):
                x1,fs1 = torchaudio.load(f"{train_dir}\\{file[0:-6]}1.wav")
                x2,fs2 = torchaudio.load(f"{train_dir}\\{file[0:-6]}2.wav")
                x3,fs3 = torchaudio.load(f"{train_dir}\\{file[0:-6]}3.wav")
                x4,fs4 = torchaudio.load(f"{train_dir}\\{file[0:-6]}4.wav")
                data = inside(x1, x2, x3, x4)
                data = data.to(device)
            
                labels, phi = target(file)
                #print(labels)
                labels  = labels.to(device)

                # Clear the gradients
                optimizer.zero_grad()
                # Forward Pass
                out = model(data)
                #print(out)
                # Find the Loss
                loss = criterion(out,labels)
                # Calculate Loss
                valid_loss += loss.item()
 
    print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
     
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f\
        }--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
         
        # Saving State Dict
        torch.save(model.state_dict(), 'F:\\Drive D\\Splitted_RS_Mic\\azimuth_tdoa1.pth')

