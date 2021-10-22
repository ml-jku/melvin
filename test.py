import argparse
import pandas as pd
import numpy as np
import datetime
import signal
import time
import yaml
import sys
import io
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import MelvinDataset, TxtDataLoader, collate, commap, ff1map, lsetmap, decode, \
  get_melvin_data_loader
from loss import srv_soft_prediction, srv_prediction
from model import MelvinNet
from utils import load_setup, Object


def print_result():
  print('mean inference time: %f ms' % (1000 * sum(exec_times) / len(exec_times)), file=sys.stderr)
  print('%d effective samples' % num_samples)
  print('maxent confusion matrix: tp=%f tn=%f fp=%f fn=%f' % (tp/num_samples, tn/num_samples, 
    fp/num_samples, fn/num_samples))
  print('SRV accuracy: %f' % (acc / num_samples))


def signal_handler(sig, frame):
  print_result()
  sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser(description='Evaluate model on a subset of the test set')
parser.add_argument('maxent', help='Path to maxent model without file extension. There must ' + 
                    'exist a [maxent].pt and a [maxent].txt file containing weights and ' + 
                    'hyperparameters respectively.')
parser.add_argument('srv', help='Path to SRV model without file extension. There must ' + 
                    'exist a [srv].pt and a [srv].txt file containing weights and ' + 
                    'hyperparameters respectively.')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for inference (default: 256)')
parser.add_argument('--n', type=int, default=0, 
                    help='minimal n-value for SRV subset (default: 0)')
parser.add_argument('--m', type=int, default=0, 
                    help='minimal m-value for SRV subset (default: 0)')
parser.add_argument('--k', type=int, default=0, 
                    help='minimal k-value for SRV subset (default: 0)')
parser.add_argument('--threshold', type=float, default=0.5, 
                    help='maxent threshold for subset (default: 0.5)')
parser.add_argument('--large', action='store_true')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()
models = []

for model_name in [args.maxent, args.srv]:
  setup = load_setup(model_name + '.txt')
  ctx = Object()
  ctx.args = setup
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(ctx.args.seed)

  #FIXME: change this to loading the entire model (or maybe not? we need map_location)
  #https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
  model = MelvinNet(embedding_dim=setup.embedding_dim, 
                    hidden_dim=setup.lstm_hidden_dim, 
                    vocab_size=len(commap), 
                    ff1_size=len(ff1map),
                    snn_hidden_dim=setup.snn_hidden_dim, 
                    output_dim=(1 if setup.task == 'maxent' else 3))
  state_dict = torch.load(model_name + '.pt', map_location=device)
  model.load_state_dict(state_dict)
  model = model.to(device)
  model.eval()
  models.append(model)

# ctx is set to SRV model context
model_maxent, model_srv = models
loader = get_melvin_data_loader(ctx.args.batch_size, 'test')
exec_times = []
lsetmap = torch.tensor(lsetmap, device=device, dtype=torch.uint8)
num_samples, acc = 0, 0
tp, tn, fp, fn = 0, 0, 0, 0

with torch.no_grad():
  for com, ff1, maxent, srv, file in loader:
    # load data to GPU and do inference
    com, ff1, srv, maxent = com.to(device), ff1.to(device), srv.to(device), maxent.to(device)
    start = time.perf_counter()
    logit_maxent = model_maxent(com, ff1, model.init_hidden(args.batch_size, device))
    logit_srv = model_srv(com, ff1, model.init_hidden(args.batch_size, device))
    exec_times.append(time.perf_counter() - start)
    
    # compute predictions
    pred_maxent = (logit_maxent >= 0.)
    pred_srv = srv_prediction(ctx, logit_srv).type(torch.int64)+1
    n, m, k = pred_srv.t()
    
    # create subset mask
    mask_valid = (srv.sum(dim=1) > 0)
    mask_thresh = (logit_maxent.reshape(args.batch_size).sigmoid() >= args.threshold)
    mask_hidim = (n >= args.n) & (m >= args.m) & (k >= args.k)
    mask = mask_valid & mask_thresh & mask_hidim
    
    if args.large:
      mask &= lsetmap[file]
    
    # calculate confusion matrix for maxent
    num_samples += mask.sum().item()
    maxent = maxent.type(torch.uint8)
    tp += (pred_maxent[mask] & maxent[mask]).sum().item()
    tn += (~pred_maxent[mask] & ~maxent[mask]).sum().item()
    fp += (pred_maxent[mask] & ~maxent[mask]).sum().item()
    fn += (~pred_maxent[mask] & maxent[mask]).sum().item()
    
    # calculate accuracy for SRV
    acc += (srv[mask] == pred_srv[mask]).all(dim=1).long().sum().item()
    
    # in verbose mode, print samples and their predictions
    if args.verbose and mask.sum().item() > 0:
      # unpack component sequences
      com, com_len = nn.utils.rnn.pad_packed_sequence(com, batch_first=True)
      
      # download to CPU and convert to numpy
      srv = srv[mask].data.cpu().numpy()
      com = com[mask].data.cpu().numpy()
      com_len = com_len[mask].data.cpu().numpy()
      ff1 = ff1[mask].data.cpu().numpy()
      maxent = maxent[mask].data.cpu().numpy()
      srv_ = srv_soft_prediction(ctx, logit_srv[mask]).data.cpu().numpy()+1
      maxent_ = logit_maxent.reshape(args.batch_size)[mask].sigmoid().data.cpu().numpy()
      
      for s, c, l, f, m, s_, m_ in zip(srv, com, com_len, ff1, maxent, srv_, maxent_):
        print(decode(s, c[:l], f) + ' | {%.6f}' % m)
        print('srv estimate: %f %f %f, maxent estimate: %f\n' % (*s_, m_))

print_result()


