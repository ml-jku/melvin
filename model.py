import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence

import numpy as np

class SNN(nn.Module):
  @staticmethod
  def init(m):
    if type(m) == nn.Linear:
      nn.init.normal_(m.weight, std=np.sqrt(1./m.in_features))
      nn.init.constant_(m.bias, 0.)
  
  def __init__(self, layer_sizes):
    super(SNN, self).__init__()
    seq = []
    
    for i, d in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
      if i > 0:
        seq.append(nn.SELU())
      seq.append(nn.Linear(*d))
    
    self.model = nn.Sequential(*seq)
    self.model.apply(SNN.init)
  
  def forward(self, input):
    return self.model(input)


class MelvinNet(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, vocab_size, ff1_size, snn_hidden_dim, output_dim):
    super(MelvinNet, self).__init__()
    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim)
    self.batchnorm = nn.BatchNorm1d(hidden_dim+ff1_size, affine=False)
    self.snn = SNN([hidden_dim+ff1_size] + snn_hidden_dim + [output_dim])
    
    # deactivate forget gate by initialization
    self.lstm.bias_hh_l0.data[hidden_dim:2*hidden_dim].fill_(float(np.finfo(np.float32).max))

  def init_hidden(self, batch_size, device):
    # The axes semantics are (num_layers, batch_size, hidden_dim)
    return (torch.zeros(1, batch_size, self.hidden_dim).to(device),
            torch.zeros(1, batch_size, self.hidden_dim).to(device))

  def forward(self, com, ff1, hidden):
    embeds = PackedSequence(self.embedding(com.data), com.batch_sizes)
    lstm_out, hidden = self.lstm(embeds, hidden)
    cat = torch.cat([hidden[1][0], ff1], -1)
    bn = self.batchnorm(cat)
    return self.snn(bn)
  
  def forward_lstm(self, com, hidden):
    embeds = PackedSequence(self.embedding(com.data), com.batch_sizes)
    lstm_out, hidden = self.lstm(embeds, hidden)
    return hidden[1][0]
  
  def forward_snn(self, ff1, cell):
    cat = torch.cat([cell, ff1], -1)
    bn = self.batchnorm(cat)
    return self.snn(bn)


class MelvinLstm(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    super(MelvinLstm, self).__init__()
    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim)
    self.linear = nn.Linear(hidden_dim, output_dim)
    
    # init weights
    for i in range(4):
      torch.nn.init.orthogonal_(self.lstm.weight_hh_l0[hidden_dim * i : hidden_dim * (i+1)])
      torch.nn.init.xavier_uniform_(self.lstm.weight_ih_l0[hidden_dim * i : hidden_dim * (i+1)])
    
    # init biases to zero, deactivate forget gate
    self.lstm.bias_hh_l0.data.fill_(0.)
    self.lstm.bias_ih_l0.data.fill_(0.)
    self.lstm.bias_hh_l0.data[hidden_dim:2*hidden_dim].fill_(float(np.finfo(np.float32).max))
    self.lstm.bias_ih_l0.data[hidden_dim:2*hidden_dim].fill_(float(np.finfo(np.float32).max))
    
    self.linear.bias.data.fill_(0.)
    torch.nn.init.xavier_uniform_(self.linear.weight)
  
  def init_hidden(self, batch_size, device):
    # The axes semantics are (num_layers, batch_size, hidden_dim)
    return (torch.zeros(1, batch_size, self.hidden_dim).to(device),
            torch.zeros(1, batch_size, self.hidden_dim).to(device))
  
  def forward(self, com, hidden):
    embeds = PackedSequence(self.embedding(com.data), com.batch_sizes)
    lstm_out, hidden = self.lstm(embeds, hidden)
    return self.linear(hidden[0][0])



class Merlin(nn.Module):
  def __init__(self, io_size, hidden_size):
    super(Merlin, self).__init__()
    self.io_size = io_size
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(io_size, hidden_size)
    self.linear = nn.Linear(hidden_size, io_size)
    
    for i in range(4):
      torch.nn.init.orthogonal_(self.lstm.weight_hh_l0[hidden_size * i : hidden_size * (i+1)])
      torch.nn.init.xavier_uniform_(self.lstm.weight_ih_l0[hidden_size * i : hidden_size * (i+1)])
    
    self.lstm.bias_hh_l0.data.fill_(0.)
    self.lstm.bias_ih_l0.data.fill_(0.)
    self.lstm.bias_hh_l0.data[hidden_size:2*hidden_size].fill_(1.)
    self.lstm.bias_ih_l0.data[hidden_size:2*hidden_size].fill_(1.)
    torch.nn.init.xavier_uniform_(self.linear.weight)
    self.linear.bias.data.fill_(0.)
  
  
  def forward(self, com, hc):
    out, hc = self.lstm(com, hc)
    
    if isinstance(out, PackedSequence):
      out = PackedSequence(self.linear(out.data), out.batch_sizes)
    else:
      out = self.linear(out)
    
    return out, hc
  
  
  
  
  
  
  
  
  
  

