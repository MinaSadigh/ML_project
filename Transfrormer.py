
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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
log_structured = pd.read_csv(
    r'Spell_result\openstack_normal2.log_structured.csv')
log_templates = pd.read_csv(
    r'Spell_result\openstack_normal2.log_templates.csv')

data_index = []

for data in log_structured["EventId"]:
       data_index.append(log_templates[log_templates["EventId"] == data].index)
L = 7
dataset = [data_index[i:i+L] for i in range(len(data_index)-L+1)]
dataset = torch.LongTensor(dataset).reshape(len(dataset), L)


batch_size = 128
tr_dataset = torch.utils.data.TensorDataset(dataset)
tr_loader = torch.utils.data.DataLoader(
    tr_dataset, batch_size=batch_size)

class TransformerModel(torch.nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = torch.nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = torch.nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.2
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:,-1,:])
        return output

    
class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)    
# Hyperparameters
ntokens = len(log_templates["EventId"])
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.4  # dropout probability
num_epoch = 100
lr = 0.001
device = 'cuda:0'  # use 'cpu' if a GPU is not available.

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def training(tr_loader, model, loss, optimizer, num_epoch=500, batch_size=100, device='cpu',verbose=1,print_every=5):
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

              out = model(X)
          
              prediction = torch.max(out, 1)[1]
              epoch_accuracy_list.append(accuracy_score(y.cpu(), prediction.cpu()))
              l = loss(out, y)
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


loss, accuracy=training(tr_loader , model, loss, optimizer, num_epoch, batch_size, device)

torch.save(model.state_dict(), "project.pt")
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


model=TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
model.load_state_dict(torch.load("project.pt"),strict=False)

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


model.eval()
with torch.no_grad():

      X =test_data[:, :L-1]
      y = test_data[:, L-1]

      X = X.to(device)
      y = y.cpu()
        
      out = model(X).cpu()
      y_pred=torch.max(out, 1)[1]
      print("FalsePositiveRate: {: .2f}".format(FalsePositiveRate(y_pred, y) *100))
      print("FalseNegativeRate: {: .2f}".format(FalseNegativeRate(y_pred, y)*100))
      print("Precision: {: .2f}".format(Precision(y_pred, y)*100))
      print("Recall: {: .2f}".format(Recall(y_pred, y)*100))
      print("Accuracy: {: .2f}".format(Accuracy(y_pred, y)*100))