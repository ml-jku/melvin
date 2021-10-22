import argparse
import pandas as pd
import numpy as np
import datetime
import time
import yaml
import sys
import io
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import MelvinDataset, TxtDataLoader, collate, commap, ff1map, decode
from loss import srv_soft_prediction
from model import MelvinNet
from utils import load_setup, Object


parser = argparse.ArgumentParser(description='Filter setups for specified properties predicted by \
a model. If the model predicts maximal entangledness, use --threshold to specify which setups \
should be considered positive. If the model predicts an SRV use --n, --m, and --k to specify \
which setups should be considered positive. The SRV estimated by the model is a float vector. \
Therefore, if you are interested in SRVs equal to or higher than (7, 7, 7), you should specify a \
threshold somewhat below, e.g. --n 6.5 --m 6.5 --k 6.5. If --radius is used, the meaning of --n, \
--m, --k changes. Instead of a threshold, they specify the center of a Euclidean ball and only \
setups whose SRV predictions lie within the ball are considered positive.')
parser.add_argument('model', help='*.pt-file containing a MelvinNet state_dict')
parser.add_argument('setup', help='*.txt-file containing hyperparameters')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for inference (default: 256)')
parser.add_argument('--threshold', type=float, default=0.5, 
                    help='maxent threshold ')
parser.add_argument('--in-file', type=str, default=sys.stdin,
                    help='file to read setups from (default: stdin)')
parser.add_argument('--pos-file', type=str, default=sys.stdout,
                    help='file to write positive setups to (default: stdout)')
parser.add_argument('--neg-file', type=str, default=sys.stderr,
                    help='file to write negative setups to (default: stderr)')
parser.add_argument('--n', type=float, default=6, 
                    help='minimal/center n-value for SRV search (default: 6)')
parser.add_argument('--m', type=float, default=6, 
                    help='minimal/center m-value for SRV search (default: 6)')
parser.add_argument('--k', type=float, default=6, 
                    help='minimal/center k-value for SRV search (default: 6)')
parser.add_argument('--radius', type=float, default=None, 
                    help='if not used, n, m, k will serve as thresholds, otherwise n, m, k \
                    specify the center of a Euclidean search ball with given radius')
args = parser.parse_args()

setup = load_setup(args.setup)
ctx = Object()
ctx.args = setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(ctx.args.seed)
refsrv = torch.tensor([[args.n, args.m, args.k]], dtype=torch.float).to(device)

#FIXME: change this to loading the entire model (or maybe not? we need map_location)
#https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model
model = MelvinNet(embedding_dim=setup.embedding_dim, 
                  hidden_dim=setup.lstm_hidden_dim, 
                  vocab_size=len(commap), 
                  ff1_size=len(ff1map),
                  snn_hidden_dim=setup.snn_hidden_dim, 
                  output_dim=(1 if setup.task == 'maxent' else 3))
state_dict = torch.load(args.model, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
loader = TxtDataLoader(args.in_file, batch_size=args.batch_size, drop_last=True)

"""
if args.data.endswith('.txt'):
  loader = TxtDataLoader(args.data, batch_size=args.batch_size, drop_last=True)
elif args.data.endswith('.db'):
  data = MelvinDataset(args.data)
  loader = DataLoader(data, batch_size=args.batch_size, collate_fn=collate, 
                      shuffle=False, drop_last=True)
else:
  print('%s: unknown file format' % args.data)
  exit()
"""

num_positive = 0
exec_times = []
lines = []
out = []
o_is_io = []

for f in [args.pos_file, args.neg_file]:
  o_is_io.append(type(f) is io.TextIOWrapper)
  out.append(f if o_is_io[-1] else open(f, 'a'))

with torch.no_grad():
  for com, ff1, maxent, srv, file in loader:
    com, ff1, srv = com.to(device), ff1.to(device), srv.to(device)
    start = time.perf_counter()
    output = model(com, ff1, model.init_hidden(args.batch_size, device))
    exec_times.append(time.perf_counter() - start)
    
    if setup.task == 'maxent':
      maxent = output.sigmoid()
      mask = (maxent >= args.threshold).resize_(maxent.shape[0])
    elif setup.task == 'srv':
      srv = srv_soft_prediction(ctx, output)+1
      
      if args.radius is None:
        n, m, k = srv.t()
        mask = (n >= args.n) & (m >= args.m) & (k >= args.k)
      else:
        mask = ((refsrv-srv).norm(dim=1) <= args.radius)
      
      srv = srv.round()
    
    com, com_len = nn.utils.rnn.pad_packed_sequence(com, batch_first=True)
    
    for o, b in zip(out, [mask, 1-mask]):
      num_positive += b.sum().item()
      np_srv = srv[b].data.cpu().numpy()
      np_com = com[b].data.cpu().numpy()
      np_com_len = com_len[b].data.cpu().numpy()
      np_ff1 = ff1[b].data.cpu().numpy()
      np_maxent = maxent[b].data.cpu().numpy()
      
      for s, c, l, f, m in zip(np_srv, np_com, np_com_len, np_ff1, np_maxent):
        lines.append(decode(s, c[:l], f) + ' | {%.6f}' % m)
      
      if len(lines) > 0:
        o.write('\n'.join(lines) + '\n')
        o.flush()
        lines = []

for o, is_io in zip(out, o_is_io):
  if not is_io:
    o.close()

#print('encountered %d positive samples' % num_positive, file=sys.stderr)
#print('mean inference time: %f ms' % (1000 * sum(exec_times) / len(exec_times)), file=sys.stderr)


