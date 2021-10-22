import os
import sys
import argparse
from shutil import rmtree
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split

from data import commap, MelvinDataset, CrossValidationDataset, collate_noff1, TxtDataLoader
from model import MelvinLstm
from utils import load_setup, log_scalars, Object
from loss import maxent_criterion, srv_criterion, srv_prediction, srv_soft_prediction
from loss import num_correct_srv



def validate(ctx, radius, thresh):
  """ run both models on validation fold
      we are trying to identify the maximally entangled setups in the validation fold
      if the predicted SRV belongs to the validation fold and the ME prediction is positive
        ==> the sample is classified as positive
      if above condition is not satisfied, the sample is classified as negative
      compute the confusion matrix
  """
  # TODO: implement


def main(args):
  # setup context and allocate model
  ctx = Object()
  ctx.srv_args = load_setup(args.srv_model + '.txt')
  ctx.me_args = load_setup(args.me_model + '.txt')
  
  args.gpu = (ctx.me_args.gpu if args.gpu is None else args.gpu)
  ctx.args = args
  ctx.args.srv_loss = ctx.srv_args.srv_loss
  ctx.device = torch.device(('cuda:%d' % args.gpu) if args.gpu >= 0 else 'cpu')
  torch.manual_seed(ctx.me_args.seed)
  ctx.metric_min = float('Inf')
  
  if ctx.args.radius is None:
    ctx.args.radius = ctx.srv_args.radius
  
  # init data loader
  folds, label = torch.load(ctx.args.cross_validation)
  
  if ctx.args.data[-3:] == '.db':
    ds = CrossValidationDataset(MelvinDataset(ctx.args.data), 
                                folds, label, ctx.args.fold, training=False)
    ctx.loader = DataLoader(ds, collate_fn=collate_noff1, batch_size=args.batch_size, 
                            sampler=ds.getSampler(), drop_last=False)
  elif ctx.args.data[-4:] == '.txt':
    ctx.loader = TxtDataLoader(ctx.args.data, maxent=None, batch_size=ctx.args.batch_size, 
                               collate_fn=collate_noff1)
  
  ctx.srv_model_file = args.srv_model + '.pt.best'
  ctx.srv_model = MelvinLstm(vocab_size=len(commap), 
                             embedding_dim=ctx.srv_args.embedding_dim, 
                             hidden_dim=ctx.srv_args.lstm_hidden_dim, 
                             output_dim=3)
  ctx.me_model_file = args.me_model + '.pt.best'
  ctx.me_model = MelvinLstm(vocab_size=len(commap), 
                            embedding_dim=ctx.me_args.embedding_dim, 
                            hidden_dim=ctx.me_args.lstm_hidden_dim, 
                            output_dim=1)
  
  state_dict = torch.load(ctx.srv_model_file, map_location=ctx.device)
  ctx.srv_model.load_state_dict(state_dict)
  ctx.srv_model = ctx.srv_model.to(ctx.device)
  ctx.srv_model.eval()
  
  state_dict = torch.load(ctx.me_model_file, map_location=ctx.device)
  ctx.me_model.load_state_dict(state_dict)
  ctx.me_model = ctx.me_model.to(ctx.device)
  ctx.me_model.eval()
  
  with open(ctx.args.out_file, 'ab') as f:
    for i, (com, target, srv, file) in enumerate(ctx.loader):
      com, target, srv = [x.to(ctx.device) for x in [com, target, srv]]
      srv = srv.float()
      
      me_output = ctx.me_model(com, ctx.me_model.init_hidden(target.shape[0], ctx.device))
      me_prediction = torch.sigmoid(me_output)
      
      srv_output = ctx.srv_model(com, ctx.srv_model.init_hidden(target.shape[0], ctx.device))
      srv_prediction = srv_soft_prediction(ctx, srv_output)
      
      result = torch.cat([target, me_prediction, srv, srv_prediction], dim=1)
      np.save(f, result.detach().cpu().numpy())



def create_argparser():
  parser = argparse.ArgumentParser(description='melvin args')
  parser.add_argument('--srv-model', type=str, default='model', help='name for SRV model file')
  parser.add_argument('--me-model', type=str, default='model', help='name for ME model file')
  parser.add_argument('--radius', type=float, default=2.0, help='radius around SRV prediction \
                       defining correctness for accurracy; 0 means equal after rounding')
  parser.add_argument('--thresh', type=float, default=0.5, help='classification threshold')
  parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                      help='input batch size for training (default: 256)')
  parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 means CPU (default: 0)')
  parser.add_argument('--fold', type=int, default=8, help='which fold to use for validation')
  parser.add_argument('--cross-validation', type=str, default='/publicwork/adler/folds_n9.pt', 
                      help='location of file defining cross validation folds')
  parser.add_argument('--data', type=str, default='melvinNoFF1Train.db', help='data file location')
  parser.add_argument('--out-file', type=str, default=None, 
                      help='where to write the results in npy format')
  return parser


if __name__ == '__main__':
  main(create_argparser().parse_args())

