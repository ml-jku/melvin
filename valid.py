import os
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

from data import get_melvin_data_loader, commap, ff1map
from model import MelvinNet
from utils import load_setup, log_scalars, Object
from loss import maxent_criterion, srv_criterion, srv_prediction


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
    for i, (com, ff1, target, srv, file) in enumerate(ctx.loader):
      com, ff1, target, srv = [x.to(ctx.device) for x in [com, ff1, target, srv]]
      srv = srv.float()
      output = ctx.model(com, ff1, ctx.model.init_hidden(ctx.args.batch_size, ctx.device))
      
      if ctx.args.task == 'maxent':
        loss, name = maxent_criterion(output, target, reduction='sum')
        valid_loss += loss.item()
        pred = (output >= 0).type(torch.float)
        cm += confusion_matrix(target.cpu().data.numpy().ravel(), pred.cpu().data.numpy().ravel())
        num_samples += ctx.args.batch_size
      elif ctx.args.task == 'srv':
        loss, name, mask = srv_criterion(output, srv, loss=ctx.args.srv_loss, reduction='sum')
        valid_loss += loss.item()
        pred = srv_prediction(ctx, output)
        srv_dist += (srv[mask] - 1 - pred[mask]).norm(dim=1).sum().item()
        srv_acc += (srv[mask] - 1 == pred[mask]).all(dim=1).long().sum().item()
        num_samples += mask.long().sum().item()
      
      if ctx.args.valid_batches_max > 0 and i == ctx.args.valid_batches_max-1:
        break
  
  valid_loss /= num_samples
  log_scalars(ctx.writer, [name], [valid_loss], step)
  
  if valid_loss < ctx.valid_loss_min:
    print('new record, saving model')
    ctx.valid_loss_min = valid_loss
    torch.save(ctx.model.state_dict(), ctx.model_file + '.best')
  
  if ctx.args.task == 'maxent':
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    spc = tn / (tn + fp)
    bacc = (tpr + spc) / 2
    names = ['true_positive_rate', 'specificity', 'bacc']
    log_scalars(ctx.writer, names, [tpr, spc, bacc], step)
    print('validation loss: {:.4f}, BACC: {:.4f}, TPR: {:.4f}, SPC: {:.4f}'.format(
      valid_loss, bacc, tpr, spc))
    print('confusion matrix: TN: {}, TP: {}, FN: {}, FP: {}'.format(tn, tp, fn, fp))
  elif ctx.args.task == 'srv':
    srv_dist /= num_samples
    srv_acc /= num_samples
    log_scalars(ctx.writer, ['srv_acc', 'srv_dist'], [srv_acc, srv_dist], step)
    print('validation loss: {:.4f}, ACC: {:.4f}, DIST: {:.4f}'.format(
      valid_loss, srv_acc, srv_dist))
  
  print('')


def main(args):
  # setup context and allocate model
  ctx = Object()
  ctx.args = load_setup(args.model_name + '.txt')
  args.gpu = (ctx.args.gpu if args.gpu is None else args.gpu)
  ctx.device = torch.device(('cuda:%d' % args.gpu) if args.gpu >= 0 else 'cpu')
  torch.manual_seed(ctx.args.seed)
  ctx.valid_loss_min = float('Inf')
  ctx.logdir = os.path.join(ctx.args.logdir, ctx.args.name)
  ctx.loader = get_melvin_data_loader(ctx.args.batch_size, 'valid')
  ctx.writer = SummaryWriter(log_dir=os.path.join(ctx.logdir, 'valid'))
  ctx.model_file = args.model_name + '.pt'
  ctx.model = MelvinNet(embedding_dim=ctx.args.embedding_dim, 
                        hidden_dim=ctx.args.lstm_hidden_dim, 
                        vocab_size=len(commap), 
                        ff1_size=len(ff1map),
                        snn_hidden_dim=ctx.args.snn_hidden_dim, 
                        output_dim=(1 if ctx.args.task == 'maxent' else 3))
  
  # busily wait for checkpoints
  last_mtime = 0.
  
  while True:
    this_mtime = os.stat(ctx.model_file).st_mtime
    
    if this_mtime > last_mtime:
      last_mtime = this_mtime
      state_dict = torch.load(ctx.model_file, map_location=ctx.device)
      ctx.model.load_state_dict(state_dict['model'])
      ctx.model = ctx.model.to(ctx.device)
      ctx.model.eval()
      step = state_dict['step']
      validate(ctx, step)
    else:
      sleep(10)


def create_argparser():
  parser = argparse.ArgumentParser(description='melvin args')
  parser.add_argument('--model-name', type=str, default='model', help='name for model file')
  parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                      help='input batch size for training (default: 256)')
  parser.add_argument('--gpu', type=int, default=None, help='GPU to use, -1 means CPU (default: 0)')
  return parser


if __name__ == '__main__':
  main(create_argparser().parse_args())


