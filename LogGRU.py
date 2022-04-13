# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 22:17:33 2022

@author: H
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import Spell
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report 

log_structured = pd.read_csv(
    r'Spell_result\openstack_normal2.log_structured.csv')
log_templates = pd.read_csv(
    r'Spell_result\openstack_normal2.log_templates.csv')

data_index = []

for data in log_structured["EventId"]:
       data_index.append(log_templates[log_templates["EventId"] == data].index)
L = 32
dataset = [data_index[i:i+L] for i in range(len(data_index)-L+1)]
dataset = torch.LongTensor(dataset).reshape(len(dataset), L)


batch_size = 256   

tr_dataset = torch.utils.data.TensorDataset(dataset)
tr_loader = torch.utils.data.DataLoader(
    tr_dataset, batch_size=batch_size)


class LogGRU(torch.nn.Module):
  def __init__(self, vocab_size=10, embedding_dim=20, hidden_size=8, num_layers=1, dropout=0.1):
    super(LogGRU, self).__init__()

    self.embedder = torch.nn.Embedding(vocab_size, embedding_dim)

    self.rnn = torch.nn.GRU(embedding_dim, hidden_size,
                            num_layers, batch_first=True)
    self.linear = torch.nn.Linear(hidden_size, vocab_size)
    self.hidden_size = hidden_size
    self.num_layers = num_layers

  def forward(self, X):
    embeddings = self.embedder(X)
    Z, _ = self.rnn(embeddings)
    out = self.linear(Z[:, -1, :])
    return out


# Hyperparameters
num_epoch = 100
lr = 0.001
hidden_size = L-1
num_layers =2
embedding_dim = 75
dropout = 0.55
device = 'cuda:0'  # use 'cpu' if a GPU is not available.

rnn = LogGRU(vocab_size=len(log_templates["EventId"]), embedding_dim=embedding_dim,
          hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
rnn.to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)


def training(tr_loader, model, loss, optimizer, num_epoch=500, batch_size=100, device='cpu',verbose=1,print_every=2):
  loss_list = []
  accuracy_list = []
  for epoch in range(num_epoch):
     epoch_accuracy_list = []

     for i, X in enumerate(tr_loader):

              data = X[0]

              X = data[:, :L-1]
              y = data[:, L-1]

              X = X.to(device)
              y = y.to(device)

              out = rnn(X)
              prediction = torch.max(out, 1)[1]
              epoch_accuracy_list.append(accuracy_score(y.cpu(), prediction.cpu()))
              l = loss(out, y.reshape(-1))
              optimizer.zero_grad()
              l.backward()
              optimizer.step()

     loss_list.append(l.item())
     accuracy = np.array(epoch_accuracy_list).mean()
     accuracy_list.append(accuracy)

     if (verbose):
        if ((epoch + 1) % print_every == 0) :
                print('epoch {}/{}: loss {:.3f} accuracy {:.3f}'.format(epoch + 1, num_epoch, l.item(), accuracy))
  return loss_list, accuracy_list


loss, accuracy=training(tr_loader , rnn, loss, optimizer, num_epoch, batch_size, device)

torch.save(rnn.state_dict(), "project.pt")
plt.figure()
plt.plot(loss)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("training loss")

plt.figure()
plt.plot(accuracy)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("training accuracy")

plt.show()



torch.manual_seed(0)
total_anomalies = 100
sample_rows = torch.randint(0, len(dataset), (5000,))
test_data = dataset[sample_rows]
affected_rows = torch.randint(0, len(test_data), (total_anomalies,))
test_data[affected_rows,-1] = (test_data[affected_rows,-1] + 5) % len(log_templates["EventId"])

def FalsePositiveRate(y_pred, y_true):
  FP = TN = 0
  for i in np.arange(len(y_pred)):
    if (y_pred[i] != y_true[i]) and not(i in affected_rows):
      FP += 1
    if (y_pred[i] == y_true[i]) and not(i in affected_rows):
      TN += 1
  return FP / (FP + TN)

def FalseNegativeRate(y_pred, y_true):
  FN = TP = 0
  for i in np.arange(len(y_pred)):
    if (y_pred[i] == y_true[i]) and (i in affected_rows):
      FN += 1
    if (y_pred[i] != y_true[i]) and (i in affected_rows):
      TP += 1
  return FN / (FN + TP)
def Precision(y_pred, y_true):
  TP = FP = 0
  for i in np.arange(len(y_pred)):
    if (y_pred[i] != y_true[i]) and (i in affected_rows):
      TP += 1
    if (y_pred[i] != y_true[i]) and not(i in affected_rows):
      FP += 1
  return TP / (TP + FP)

def Recall(y_pred, y_true):
  TP = FN = 0
  for i in np.arange(len(y_pred)):
    if (y_pred[i] != y_true[i]) and (i in affected_rows):
      TP += 1
    if (y_pred[i] == y_true[i]) and (i in affected_rows):
      FN += 1
  return TP / (TP + FN)
def Accuracy(y_pred, y_true):
  correct = 0
  for i in np.arange(len(y_pred)):
    if ((y_pred[i] != y_true[i]) and (i in affected_rows)) or ((y_pred[i] == y_true[i]) and not(i in affected_rows)):
      correct += 1
  return correct / len(y_pred)


rnn.eval()
with torch.no_grad():

      X =test_data[:, :L-1]
      y = test_data[:, L-1]

      X = X.to(device)
      y = y.cpu()
        
      out = rnn(X).cpu()
      y_pred=torch.max(out, 1)[1]
      print("FalsePositiveRate: {: .2f}".format(FalsePositiveRate(y_pred, y) *100))
      print("FalseNegativeRate: {: .2f}".format(FalseNegativeRate(y_pred, y)*100))
      print("Precision: {: .2f}".format(Precision(y_pred, y)*100))
      print("Recall: {: .2f}".format(Recall(y_pred, y)*100))
      print("Accuracy: {: .2f}".format(Accuracy(y_pred, y)*100))