import os
import sys
import argparse
from shutil import rmtree
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split

from data import commap, MelvinDataset, CrossValidationDataset, collate_noff1
from model import MelvinLstm
from utils import dump_setup, log_scalars, Object
from loss import maxent_criterion, srv_criterion, srv_prediction, num_correct_srv
from valid_noff1 import validate


class NopScheduler(object):
  def step(*arg1):
    pass


def train(ctx):
  for batch_idx, (com, target, srv, file) in enumerate(ctx.loader):
    step = ctx.epoch * len(ctx.loader) + batch_idx
    
    if step % ctx.args.valid_interval == 0 or step == ctx.args.step_max:
      if ctx.args.sync_valid:
        validate(ctx, step)
        ctx.model.train()
      else:
        state_dict = {'model': ctx.model.state_dict(), 'step': step}
        torch.save(state_dict, ctx.model_file)
    
    if step == ctx.args.step_max:
      break
    
    com, target, srv = [x.to(ctx.device) for x in [com, target, srv]]
    srv = srv.float()
    ctx.optimizer.zero_grad()
    ctx.model.zero_grad()
    output = ctx.model(com, ctx.model.init_hidden(target.shape[0], ctx.device))
    
    if ctx.args.task == 'maxent':
      loss, name = maxent_criterion(output, target, reduction='sum')
      this_batch = torch.tensor(ctx.args.batch_size, dtype=torch.float)
      ctx.num_correct += ((output >= 0).type(torch.float) == target).sum().item()
    elif ctx.args.task == 'srv':
      loss, name, mask = srv_criterion(output, srv, loss=ctx.args.srv_loss, reduction='sum')
      this_batch = mask.long().sum()
      ctx.num_correct += num_correct_srv(output, srv, mask, ctx, ctx.args.radius)
    
    assert(torch.isfinite(loss).item())
    ctx.sum_loss += loss.item()
    ctx.num_samples += this_batch.item()
    (loss/this_batch).backward()
    ctx.optimizer.step()
    
    if step > 0 and step % ctx.args.log_interval == 0:
      avg_loss = ctx.sum_loss / ctx.num_samples
      avg_acc = ctx.num_correct / ctx.num_samples
      log_scalars(ctx.writer, [name, 'acc'], [avg_loss, avg_acc], step)
      ctx.scheduler.step(avg_loss)
      
      ctx.sum_loss = 0
      ctx.num_samples = 0
      ctx.num_correct = 0
      
      print('epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}\tacc: {:.6f}'.format(
        ctx.epoch+1, batch_idx * ctx.args.batch_size, len(ctx.loader) * ctx.args.batch_size,
        100. * batch_idx / len(ctx.loader), avg_loss, avg_acc))
  
  if step == ctx.args.step_max:
    raise StopIteration


def main(args):
  # collect objects for use in different context (i.e. function)
  ctx = Object()

  # parse command line args
  ctx.args = args
  ctx.device = torch.device(('cuda:%d' % ctx.args.gpu) if ctx.args.gpu >= 0 else 'cpu')
  torch.manual_seed(ctx.args.seed)

  # init data loader
  folds, label = torch.load(ctx.args.cross_validation)
  ds = CrossValidationDataset(MelvinDataset(ctx.args.data), folds, label, 
                              ctx.args.fold, task=ctx.args.task)
  ctx.loader = DataLoader(ds, collate_fn=collate_noff1, batch_size=ctx.args.batch_size, 
                          sampler=ds.getSampler())
  
  if ctx.args.sync_valid:
    ds = CrossValidationDataset(MelvinDataset(ctx.args.data), folds, label, 
                                ctx.args.fold, task=ctx.args.task, training=False)
    ctx.valldr = DataLoader(ds, collate_fn=collate_noff1, batch_size=ctx.args.batch_size, 
                            sampler=ds.getSampler())
    ctx.metric_min = float('Inf')

  # init model
  ctx.model = MelvinLstm(vocab_size=len(commap), 
                         embedding_dim=ctx.args.embedding_dim, 
                         hidden_dim=ctx.args.lstm_hidden_dim, 
                         output_dim=(1 if ctx.args.task == 'maxent' else 3))
  ctx.model = ctx.model.to(ctx.device)
  ctx.model.train()
  ctx.sum_loss = 0
  ctx.num_samples = 0
  ctx.num_correct = 0

  # init optimizer
  ctx.optimizer = torch.optim.SGD(ctx.model.parameters(), lr=ctx.args.lr, 
                                  momentum=ctx.args.momentum)
  ctx.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, verbose=True, 
                                                             patience=100)
  #ctx.scheduler = NopScheduler()

  # init summary writer
  ctx.logdir = os.path.join(ctx.args.logdir, ctx.args.name)
  
  if os.path.exists(ctx.logdir):
    if __name__ == '__main__':
      # ask the user whether it's ok to overwrite the logs
      a = input('Directory %s exists. Delete? [y/n] ' % ctx.logdir)
      
      if a.lower().startswith('y'):
        rmtree(ctx.logdir)
      else:
        print('exiting')
        exit()
    else:
      # don't use input if we're not main (e.g. called from hyper), overwrite!
      print('Warning: overwriting %s' % ctx.logdir)
      rmtree(ctx.logdir)
  
  ctx.writer = SummaryWriter(log_dir=os.path.join(ctx.logdir, 'train'))
  dump_setup(ctx)
  ctx.model_file = os.path.join(ctx.logdir, ctx.args.model_name + '.pt')
  
  if ctx.args.sync_valid:
    ctx.valwrt = SummaryWriter(log_dir=os.path.join(ctx.logdir, 'valid'))
  
  if args.stdout is not None:
    sys.stdout = open(args.stdout, 'w', buffering=1)

  for ctx.epoch in range(ctx.args.epochs):
    train(ctx)


def create_argparser():
  parser = argparse.ArgumentParser(description='melvin args')
  parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                      help='input batch size for training (default: 256)')
  parser.add_argument('--epochs', type=int, default=6, metavar='N',
                      help='number of epochs to train (default: 6)')
  parser.add_argument('--step-max', type=int, default=0, metavar='N',
                      help='maximal number of weight updates (default: no limit)')
  parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                      help='learning rate (default: 0.1)')
  parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 means CPU (default: 0)')
  parser.add_argument('--seed', type=int, default=1, metavar='S',
                      help='random seed (default: 1)')
  parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--valid-interval', type=int, default=1000, metavar='N',
                      help='how many batches to train before validating')
  parser.add_argument('--valid-batches-max', type=int, default=0, 
                      help='maximum number of batches to validate at once (default: 0, no limit)')
  parser.add_argument('--lambda-srv', type=float, default=0.,
                      help='lambda for auxiliary SRV prediction')
  parser.add_argument('--task', type=str, default='maxent', choices=['maxent', 'srv'],
                      help='which task to learn, maxent or srv')
  parser.add_argument('--logdir', type=str, default='runs', help='path to log directory')
  parser.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), 
                      help='name for this run (default: current timestamp)')
  parser.add_argument('--model-name', type=str, default='model', help='name for model file')
  parser.add_argument('--embedding-dim', type=int, default=64, help='size of embedding layer')
  parser.add_argument('--lstm-hidden-dim', type=int, default=256, help='number of LSTM units')
  parser.add_argument('--srv-loss', type=str, default='poisson_binomial', choices=['mse', 
                      'poisson', 'poisson_binomial'], help='the loss function for SRV regression')
  parser.add_argument('--radius', type=float, default=0, help='radius around SRV prediction \
                       defining correctness for accurracy; 0 means equal after rounding')
  parser.add_argument('--fold', type=int, default=8, help='which fold to hold out while training')
  parser.add_argument('--cross-validation', type=str, default='/publicwork/adler/folds.pt', 
                      help='location of file defining cross validation folds')
  parser.add_argument('--data', type=str, default='melvinNoFF1Train.db', help='data file location')
  parser.add_argument('--stdout', type=str, default=None, help='redirect stdout to file')
  #parser.add_argument('--comment', type=str, default='', help='postfix for the working directory')
  parser.add_argument('--sync-valid', action='store_true')
  return parser


if __name__ == '__main__':
  main(create_argparser().parse_args())


