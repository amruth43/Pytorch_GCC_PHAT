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
import torchaudio.functional as F
import torchaudio.transforms as T

#from sklearn.metrics import classification_report

from torchmetrics.functional.classification import binary_accuracy
from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.functional.classification import binary_f1_score

#define the required hyperparameter
num_classes = 8
learning_rate=0.001
num_epochs = 50

#define the spectrograms
sample_rate = 16000

#Define the feature to be extracted

def gccphat(sig, refsig, fs=sample_rate, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0] + 1024

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
    y = torch.FloatTensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
  elif '@45' in BF:
    phi = 45
    y = torch.FloatTensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
  elif '@90' in BF:
    phi = 90
    y = torch.FloatTensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
  elif '@135' in BF:
    phi = 135
    y = torch.FloatTensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
  elif '@180' in BF:
    phi = 180
    y = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
  elif '@225' in BF:
    phi = 225
    y = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
  elif '@270' in BF:
    phi = 270
    y = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
  elif '@315' in BF:
    phi = 315
    y = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
  return y, phi

def inside(x1_spec, x2_spec, x3_spec, x4_spec):
  xxp, ccxxp = gccphat(x1_spec, x3_spec)
  xxn, ccxxn = gccphat(x3_spec, x1_spec)
  xyp, ccxyp = gccphat(x2_spec, x4_spec)
  xyn, ccxyn = gccphat(x4_spec, x2_spec)
  x21, ccx21 = gccphat(x2_spec, x1_spec)
  x41, ccx41 = gccphat(x4_spec, x1_spec)
  x32, ccx32 = gccphat(x3_spec, x2_spec)
  x12, ccx12 = gccphat(x1_spec, x2_spec)
  x43, ccx43 = gccphat(x4_spec, x3_spec)
  x23, ccx23 = gccphat(x2_spec, x3_spec)
  x14, ccx14 = gccphat(x1_spec, x4_spec)
  x34, ccx34 = gccphat(x3_spec, x4_spec)
  x_cat = torch.FloatTensor([xxp, xxn, xyp, xyn, x21, x41, x32, x12, x43, x23, x14, x34])
  #x_cat = torch.nn.functional.normalize((x_cat), dim = 0)
  return x_cat

#class NeuralNet(nn.Module):
#  def __init__(self):
#    super(NeuralNet, self).__init__()
# 
#    self.fc1 = nn.Linear(12, 50)
#    self.fc2 = nn.Linear(50, 25)
#    #self.fc3 = nn.Linear(25, 25)
#    self.fc4 = nn.Linear(25, 8)
#
#  def forward(self, x):
#    #print(x)
#    x = torch.flatten(x)
#    x = torch.nn.functional.tanh(self.fc1(x))
#    x = torch.nn.functional.tanh(self.fc2(x))
#    #x = torch.nn.functional.tanh(self.fc3(x))
#    x = torch.nn.functional.softmax(self.fc4(x), dim = 0)
#    return x

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
#model1 = NeuralNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr = learning_rate)

train_dir = 'F:\\Drive D\\Splitted_RS_Mic\\train'
val_dir = 'F:\\Drive D\\Splitted_RS_Mic\\validate'
test_dir = 'F:\\Drive D\\Splitted_RS_Mic\\test'

#Read all the inputs from the dataset to train the components
for epochs in range(num_epochs):
  entries = os.listdir(train_dir)

  train_loss = 0
  train_acc = 0
  train_f1_score = 0

  for file in entries:
    if file.endswith("BF.wav"):
      try:
        BF = file
        y_cap = None
        y_cap, phi = target(BF)
        #print(y_cap)
        if y_cap is not None:
          y_cap = y_cap.squeeze(0)
          y_cap = y_cap.to(device)
        #if (f"{BF[0:-6]}RS_1.wav") in entries:  # and (f"{BF[0:-6]}RS_2.wav") (f"{BF[0:-6]}RS_3.wav") and (f"{BF[0:-6]}RS_4.wav")) in entries:


        x1,fs1 = torchaudio.load(f"{train_dir}\\{BF[0:-6]}1.wav")
        x2,fs2 = torchaudio.load(f"{train_dir}\\{BF[0:-6]}2.wav")
        x3,fs3 = torchaudio.load(f"{train_dir}\\{BF[0:-6]}3.wav")
        x4,fs4 = torchaudio.load(f"{train_dir}\\{BF[0:-6]}4.wav")

        x1 = torch.FloatTensor(x1).to(device)
        x2 = torch.FloatTensor(x2).to(device)
        x3 = torch.FloatTensor(x3).to(device)
        x4 = torch.FloatTensor(x4).to(device)

        x_in = inside(x1, x2, x3, x4)
        x_in = x_in.to(device)
        #x_in = torch.nn.functional.relu(x_in).to(device) #Remove negative values

        y_predicted = model1(x_in)
        y_predicted = y_predicted.squeeze(0)
        y_predicted = y_predicted.to(device)

        phi_cap = direction_phi(y_predicted)
        if y_cap is not None:
          loss = criterion(y_predicted, y_cap)
          acc = binary_accuracy(y_predicted, y_cap)
          f1_score = binary_f1_score(y_predicted, y_cap)

        #empty the gradient
        optimizer.zero_grad()

        #backward pass
        loss.backward()
        
        #update
        optimizer.step()

        train_loss += loss.item()
        #train_loss = train_loss/len(entries)*5

        #train_acc += acc.item()
        #train_acc = train_acc/len(entries)*5

        if (phi == phi_cap):
          train_acc += 1
        #train_f1_score += f1_score.item()
        #train_f1_score = train_f1_score/len(entries)*5

        #print(f'Training Expected_Output = {y_cap}, Training_Predicted_Output = {y_predicted} :: Training_Loss = {loss.item()}')

      except RuntimeError:
        print(f"Did not find the complete file: {train_dir}/{BF[0:-6]}")

  if (epochs+1) %1 == 0:
    print(f'epoch: {epochs+1}:: Train_loss = {train_loss/len(entries)*5} :: Train_Acc = {train_acc/len(entries)*5} :: Train_F1 = {train_f1_score/len(entries)*5}')

  torch.save(model1.state_dict(), 'F:\\Drive D\\Splitted_RS_Mic\\azimuth_tdoa.pth')

  #let us try for each evaluation model

  model1.load_state_dict(torch.load('F:\\Drive D\\Splitted_RS_Mic\\azimuth_tdoa.pth'))
  model1.eval()

  val_loss = 0
  val_acc = 0
  val_f1_score = 0

  entries1 = os.listdir(val_dir)
  for file in entries1:
    if file.endswith("BF.wav"):
      try:

        BF = file
        y_cap1 = None
        y_cap1, phi1 = target(BF)
        if y_cap1 is not None:
          y_cap1 = y_cap1.squeeze(0).to(device)
          y_cap1 = y_cap1.to(device)

        #if (f"{BF[0:-6]}RS_1.wav") in entries:  # and (f"{BF[0:-6]}RS_2.wav") (f"{BF[0:-6]}RS_3.wav") and (f"{BF[0:-6]}RS_4.wav")) in entries:
        x1,fs1 = torchaudio.load(f"{val_dir}\\{BF[0:-6]}1.wav")
        x2,fs2 = torchaudio.load(f"{val_dir}\\{BF[0:-6]}2.wav")
        x3,fs3 = torchaudio.load(f"{val_dir}\\{BF[0:-6]}3.wav")
        x4,fs4 = torchaudio.load(f"{val_dir}\\{BF[0:-6]}4.wav")

        x1 = torch.FloatTensor(x1).to(device)
        x2 = torch.FloatTensor(x2).to(device)
        x3 = torch.FloatTensor(x3).to(device)
        x4 = torch.FloatTensor(x4).to(device)

        x_in1 = inside(x1, x2, x3, x4)
        x_in1 = x_in1.to(device)

        #x_in = torch.nn.functional.relu(x_in).to(device) #Remove negative values
        y_predicted1 = model1(x_in1)
        y_predicted1 = y_predicted1.squeeze(0)
        y_predicted1 = y_predicted1.to(device)

        phi_cap1 = direction_phi(y_predicted1)
        
        if y_cap is not None:
          loss1 = criterion(y_predicted1, y_cap1)
          acc1 = binary_accuracy(y_predicted1, y_cap1)
          f1_score1 = binary_f1_score(y_predicted1, y_cap1)

        val_loss += loss1.item()
        #val_loss = val_loss/len(entries)*5


        if (phi1 == phi_cap1):
          val_acc += 1

        val_f1_score += f1_score1.item()

        #print(f'Val_Expected_Output = {y_cap1}, Val_Predicted_Output = {y_predicted1} :: Val_Loss = {loss1.item()}')

      except RuntimeError:
        print(f"Did not find the complete file: {val_dir}/{BF[0:-6]}")


  if (epochs+1) %1 == 0:
    print(f'epoch: {epochs+1}:: Val_Loss = {val_loss/len(entries)*5} :: Val_Acc = {val_acc/len(entries)*5} :: Val_F1 = {val_f1_score/len(entries)*5}')

  #let us plot the output materials


