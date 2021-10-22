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
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader, random_split

from data import commap, MelvinDataset, CrossValidationDataset, collate_noff1
from model import MelvinLstm
from utils import load_setup, log_scalars, Object
from loss import maxent_criterion, srv_criterion, srv_prediction, srv_soft_prediction
from loss import num_correct_srv


def validate(ctx, step):
  # TODO: maybe outsource to different file. this parallelizes training and validation but 
  # complicates usage, either we have to start separate programs or we use multiprocessing. 
  # we have to check model files for updates and validate them. but we do get full batch 
  # validation instead of stochastic validation results as we have now. 
  ctx.model.eval()
  valid_loss = 0
  cm = np.zeros((2, 2), dtype=np.int)
  srv_dist = 0
  srv_acc = 0
  num_samples = 0
  
  print('validating step %d...' % step)
  
  with torch.no_grad():
    for i, (com, target, srv, file) in enumerate(ctx.valldr):
      print(i)
      com, target, srv = [x.to(ctx.device) for x in [com, target, srv]]
      srv = srv.float()
      output = ctx.model(com, ctx.model.init_hidden(target.shape[0], ctx.device))
      
      if ctx.args.task == 'maxent':
        loss, name = maxent_criterion(output, target, reduction='sum')
        valid_loss += loss.item()
        pred = (output >= 0).type(torch.float)
        cm += confusion_matrix(target.cpu().data.numpy().ravel(), pred.cpu().data.numpy().ravel())
        num_samples += ctx.args.batch_size
      elif ctx.args.task == 'srv':
        loss, name, mask = srv_criterion(output, srv, loss=ctx.args.srv_loss, reduction='sum')
        valid_loss += loss.item()
        srv_acc += num_correct_srv(output, srv, mask, ctx, ctx.args.radius)
        pred = srv_soft_prediction(ctx, output)
        srv_dist += (srv[mask] - 1 - pred[mask]).norm(dim=1).sum().item()
        num_samples += mask.long().sum().item()
      
      if ctx.args.valid_batches_max > 0 and i == ctx.args.valid_batches_max-1:
        break
  
  valid_loss /= num_samples
  srv_acc /= num_samples
  log_scalars(ctx.valwrt, [name, 'acc'], [valid_loss, srv_acc], step)
  
  if False and valid_loss < ctx.metric_min:
    print('new record, saving model')
    ctx.metric_min = valid_loss
    torch.save(ctx.model.state_dict(), ctx.model_file + '.best')
  
  if ctx.args.task == 'maxent':
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    spc = tn / (tn + fp)
    bacc = (tpr + spc) / 2
    speedup = 1/(1-spc) * tpr
    names = ['true_positive_rate', 'specificity', 'bacc', 'speedup']
    log_scalars(ctx.valwrt, names, [tpr, spc, bacc, speedup], step)
    print('validation loss: {:.4f}, BACC: {:.4f}, TPR: {:.4f}, SPC: {:.4f}, SPEEDUP: {:.4f}'.format(
      valid_loss, bacc, tpr, spc, speedup))
    print('confusion matrix: TN: {}, TP: {}, FN: {}, FP: {}'.format(tn, tp, fn, fp))
    
    if 1-bacc < ctx.metric_min:
      print('new record, saving model')
      ctx.metric_min = 1-bacc
      torch.save(ctx.model.state_dict(), ctx.model_file + '.best')
    
  elif ctx.args.task == 'srv':
    srv_dist /= num_samples
    log_scalars(ctx.valwrt, ['srv_dist'], [srv_dist], step)
    print('validation loss: {:.4f}, ACC: {:.4f}, DIST: {:.4f}'.format(
      valid_loss, srv_acc, srv_dist))
    
    if srv_dist < ctx.metric_min:
      print('new record, saving model')
      ctx.metric_min = srv_dist
      torch.save(ctx.model.state_dict(), ctx.model_file + '.best')
  
  print('')


def main(args):
  if args.stdout is not None:
    sys.stdout = open(args.stdout, 'w', buffering=1)
  
  # setup context and allocate model
  ctx = Object()
  ctx.args = load_setup(args.model_name + '.txt')
  args.gpu = (ctx.args.gpu if args.gpu is None else args.gpu)
  ctx.valid_args = args
  ctx.device = torch.device(('cuda:%d' % args.gpu) if args.gpu >= 0 else 'cpu')
  torch.manual_seed(ctx.args.seed)
  ctx.metric_min = float('Inf')
  ctx.logdir = os.path.join(ctx.args.logdir, ctx.args.name)
  
  if ctx.valid_args.radius is not None:
    ctx.args.radius = ctx.valid_args.radius
  
  # init data loader
  folds, label = torch.load(ctx.args.cross_validation)
  ds = MelvinDataset(args.data)
  ctx.valldr = DataLoader(ds, collate_fn=collate_noff1, batch_size=args.batch_size, 
                          sampler=SequentialSampler(ds))
  
  ctx.valwrt = SummaryWriter(log_dir=os.path.join(ctx.logdir, 'test'))
  ctx.model_file = args.model_name + '.pt'
  ctx.model = MelvinLstm(vocab_size=len(commap), 
                         embedding_dim=ctx.args.embedding_dim, 
                         hidden_dim=ctx.args.lstm_hidden_dim, 
                         output_dim=(1 if ctx.args.task == 'maxent' else 3))
  
  state_dict = torch.load(ctx.model_file, map_location=ctx.device)
  ctx.model.load_state_dict(state_dict['model'])
  ctx.model = ctx.model.to(ctx.device)
  ctx.model.eval()
  step = state_dict['step']
  validate(ctx, step)


def create_argparser():
  parser = argparse.ArgumentParser(description='melvin args')
  parser.add_argument('--model-name', type=str, default='model', help='name for model file')
  parser.add_argument('--radius', type=float, default=None, help='radius around SRV prediction \
                       defining correctness for accurracy; 0 means equal after rounding')
  parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                      help='input batch size for training (default: 256)')
  parser.add_argument('--gpu', type=int, default=None, help='GPU to use, -1 means CPU (default: 0)')
  parser.add_argument('--data', type=str, default='melvinNoFF1Test.db', help='data file location')
  parser.add_argument('--stdout', type=str, default=None, help='redirect stdout to file')
  return parser


if __name__ == '__main__':
  main(create_argparser().parse_args())


