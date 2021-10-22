import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import PackedSequence, pack_sequence

import os
import numpy as np
import argparse
from datetime import datetime

from tensorboardX import SummaryWriter

from data import MelvinDataset, commap, collate_noff1_unpacked
from utils import dump_setup, load_setup, log_scalars, Object
from model import Merlin


###############################################################################
# train and validate an LSTM for next element prediction 
# on the positive samples of the melvin data. 
# 
# TODO:
#   * check if class weighting is correct (inconsistency with topk metric)
#   * add a begin-of-sequence (BOS) token to alphabet 
#     so we can sample also the first component from the model
#   * implement sampling
#   * regularize! implement LSTM dropout
#     see https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
###############################################################################


def make_data_split(db_file, train_frac=0.7, valid_frac=0.1):
  data = MelvinDataset(db_file)
  len_train = int(len(data) * train_frac)
  len_valid = int(len(data) * valid_frac)
  split = [len_train, len_valid, len(data) - len_train - len_valid]
  return random_split(data, split)


def create_argparser():
  descr = 'train an LSTM for next element prediction or use a trained model to generate sequences. \n\nexample for training \npython3 next_element_prediction.py --data /local00/bioinf/adler/melvinNoFF1Maxent.db --lr 0.001 --batch-size 128 --lstm-size 1024 --name nep_lr1e-3_lstm1024 --epochs 10 --gpu 0 \n\nexample for generating\npython3 next_element_prediction.py --generate 256 --batch-size 4096 --logdir /publicwork/adler/melvin/runs/ --name nep_lr1e-3_lstm1024\n'
  parser = argparse.ArgumentParser(description='melvin args')
  parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                      help='input batch size for training (default: 256)')
  parser.add_argument('--epochs', type=int, default=6, metavar='N',
                      help='number of epochs to train (default: 6)')
  parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                      help='learning rate (default: 0.1)')
  parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                      help='SGD momentum (default: 0.5)')
  parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 means CPU (default: 0)')
  parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                      help='how many batches to wait before logging training status')
  parser.add_argument('--logdir', type=str, default='runs', help='path to log directory')
  parser.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%dT%H-%M-%S'), 
                      help='name for this run (default: current timestamp)')
  parser.add_argument('--model-name', type=str, default='model', help='name for model file')
  parser.add_argument('--lstm-size', type=int, default=64, help='number of LSTM units')
  parser.add_argument('--data', type=str, default='melvinNoFF1Maxent.db', help='data file location')
  parser.add_argument('--generate', type=int, metavar='N', default=None, help='do not train but load the model specified by logdir/name/model-name and generate N batches of samples from it')
  parser.add_argument('--maxlen', type=int, metavar='N', default=20, help='maximal length of generated sequences')
  parser.add_argument('--k', type=int, metavar='K', default=3, help='beam search width')
  return parser


def preprocess(ctx, com):
  with torch.no_grad():
    # prepare inputs as one-hot seqs and outputs as integers shifted by one and ending with EOS
    cin = pack_sequence(com).to(ctx.device)
    cin = PackedSequence(torch.eye(len(commap)).to(ctx.device)[cin.data], cin.batch_sizes)
    cout = (torch.cat((c[1:], torch.tensor((len(commap)-1,)))) for c in com)
    cout = pack_sequence(tuple(cout)).to(ctx.device)
  
  return cin, cout


def topk(pred, true, k):
  return (pred.topk(k, dim=1)[1] == true.reshape(len(true), 1)).float().sum() / len(true)


def train(ctx):
  ctx.model.train()
  
  for i, (com, maxent, srv, file) in enumerate(ctx.train_ldr):
    ctx.step = ctx.epoch * len(ctx.train_ldr) + i
    cin, cout = preprocess(ctx, com)
    h, c = torch.zeros(2, 1, cin.batch_sizes[0], ctx.args.lstm_size).to(ctx.device)
    h, _ = ctx.model(cin, (h, c))
    loss = F.cross_entropy(h.data, cout.data, weight=ctx.weight)
    ctx.optimizer.zero_grad()
    loss.backward()
    ctx.optimizer.step()
    ctx.writer.add_scalar('loss', loss, ctx.step)
    
    if ctx.step % ctx.args.log_interval == 0:
      top1 = 100 * topk(h.data, cout.data, 1)
      ctx.writer.add_scalar('top1', top1, ctx.step)
      top5 = 100 * topk(h.data, cout.data, 5)
      ctx.writer.add_scalar('top5', top5, ctx.step)
      print('step', ctx.step, 'loss', loss.item(), 'top1', top1.item(), 'top5', top5.item())


def validate(ctx):
  print('validating...')
  ctx.model.eval()
  loss, top1, top5 = [], [], []
  
  with torch.no_grad():
    for i, (com, maxent, srv, file) in enumerate(ctx.valid_ldr):
      cin, cout = preprocess(ctx, com)
      h, c = torch.zeros(2, 1, cin.batch_sizes[0], ctx.args.lstm_size).to(ctx.device)
      h, _ = ctx.model(cin, (h, c))
      loss.append(F.cross_entropy(h.data, cout.data, weight=ctx.weight))
      top1.append(100 * topk(h.data, cout.data, 1))
      top5.append(100 * topk(h.data, cout.data, 5))
    
    last_batch_size = len(ctx.valid_data) % ctx.args.batch_size
    
    # FIXME: this correction is wrong because of different sequence lengths
    if False and last_batch_size != 0:
      c  = [ctx.args.batch_size / len(ctx.valid_data)] * i
      c += [last_batch_size / len(ctx.valid_data)]
      loss = (torch.stack(loss) * torch.tensor(c)).sum()
    else:
      loss = torch.stack(loss).mean()
      top1 = torch.stack(top1).mean()
      top5 = torch.stack(top5).mean()
    
    ctx.valwrt.add_scalar('loss', loss, ctx.step)
    ctx.valwrt.add_scalar('top1', top1, ctx.step)
    ctx.valwrt.add_scalar('top5', top5, ctx.step)
    print('epoch', ctx.epoch+1, 'loss', loss.item(), 'top1', top1.item(), 'top5', top5.item())
  
  return loss.item()


def main(ctx):
  ctx.device = torch.device(('cuda:%d' % ctx.args.gpu) if ctx.args.gpu >= 0 else 'cpu')
  ctx.weight = torch.tensor([ # these were produced by compute_weights()
    0.4019, 0.6905, 0.9775, 0.4853, 0.9794, 0.4844, 1.2064, 3.7559, 1.1979,
    3.7369, 0.9859, 0.4932, 0.9836, 0.4942, 1.2775, 3.1449, 1.2775, 3.1499,
    0.2362, 1.2409, 1.4082, 3.2633, 1.3949, 3.2619, 1.4025, 3.2596, 1.4043,
    3.2751, 1.1174, 1.0896, 1.1461, 1.1464, 1.0697, 0.9966, 1.0950, 0.9026,
    1.1886, 1.1913, 0.8997, 1.0971, 0.9936, 1.0701, 1.1519, 1.1538, 0.9779,
    0.8465, 1.0233, 0.9267, 1.0868, 1.0867, 0.9258, 1.0230, 0.8480, 0.9739,
    1.2061, 1.1996, 0.7767, 0.8151, 0.8309, 1.0319, 1.0277, 1.0293, 1.0333,
    0.8339, 0.8127, 0.7783, 1.1848, 1.1963, 0.7802, 0.8100, 0.8306, 1.0380,
    1.0296, 1.0278, 1.0340, 0.8324, 0.8123, 0.7790, 1.0857, 1.0813, 1.0918,
    1.0871, 1.0919, 1.0834, 1.0926, 1.0927, 1.0823, 1.0881, 1.0836, 1.0877,
    1.0893, 1.0892, 1.0871, 1.0834, 1.0878, 1.0863, 1.0929, 1.0907, 1.0832,
    1.0878, 1.0866, 1.0883, 0.1051]).to(ctx.device)
  ctx.train_data, ctx.valid_data, ctx.test_data = make_data_split(ctx.args.data)
  
  for s in ('train', 'valid', 'test'):
    ds = getattr(ctx, s + '_data')
    ldr = DataLoader(ds, collate_fn=collate_noff1_unpacked, batch_size=ctx.args.batch_size)
    setattr(ctx, s + '_ldr', ldr)
  
  ctx.model = Merlin(len(commap), ctx.args.lstm_size).to(ctx.device)
  ctx.optimizer = torch.optim.Adam(ctx.model.parameters(), lr=ctx.args.lr)
  #ctx.optimizer = torch.optim.SGD(ctx.model.parameters(), lr=ctx.args.lr, 
  #                                momentum=ctx.args.momentum)
  ctx.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(ctx.optimizer, verbose=True, 
                                                             patience=3)
  ctx.logdir = os.path.join(ctx.args.logdir, ctx.args.name)
  ctx.writer = SummaryWriter(log_dir=os.path.join(ctx.logdir, 'train'))
  ctx.valwrt = SummaryWriter(log_dir=os.path.join(ctx.logdir, 'valid'))
  dump_setup(ctx)
  ctx.model_file = os.path.join(ctx.logdir, ctx.args.model_name + '.pt')
  best_loss = float('inf')
  
  for ctx.epoch in range(ctx.args.epochs):
    train(ctx)
    loss = validate(ctx)
    ctx.scheduler.step(loss)
    
    if loss <= best_loss:
      torch.save(ctx.model.state_dict(), ctx.model_file)


def decode(com):
  comlist = []
  EOS = commap.index('EOS')
  
  for c in com:
    if c == EOS:
      break
    
    comlist.append(commap[c])
  
  return '{0,0,0} | {"' + '", "'.join(comlist) + '"} | {}'


def generate(ctx):
  ctx.device = torch.device(('cuda:%d' % ctx.args.gpu) if ctx.args.gpu >= 0 else 'cpu')
  ctx.logdir = os.path.join(ctx.args.logdir, ctx.args.name)
  ctx.model_file_noex = os.path.join(ctx.logdir, ctx.args.model_name)
  ctx.args_model = load_setup(ctx.model_file_noex + '.txt')
  ctx.model = Merlin(len(commap), ctx.args_model.lstm_size).to(ctx.device)
  ctx.model.load_state_dict(torch.load(ctx.model_file_noex + '.pt', map_location=ctx.device))
  ctx.model.eval()
  eye = torch.eye(len(commap)).to(ctx.device)
  
  with torch.no_grad():
    for i in range(ctx.args.generate):
      x = torch.randint(len(commap)-1, (ctx.args.batch_size,))
      x = x.to(ctx.device).reshape(1, *x.shape)
      x_seq = [x]
      x = eye[x]
      hc = tuple(torch.zeros(2, 1, ctx.args.batch_size, ctx.args_model.lstm_size).to(ctx.device))
      
      for _ in range(ctx.args.maxlen):
        x, hc = ctx.model(x, hc)
        topk = x.topk(ctx.args.k, dim=2)[1]
        choice = torch.randint(ctx.args.k, (ctx.args.batch_size,))
        x = topk[:,torch.arange(ctx.args.batch_size),choice]
        x_seq.append(x)
        x = eye[x]
      
      com = torch.cat(x_seq, dim=0)
      
      for j in range(ctx.args.batch_size):
        print(decode(com[:,j]))


if __name__ == '__main__':
  torch.manual_seed(0)
  ctx = Object()
  ctx.args = create_argparser().parse_args()
  
  if ctx.args.generate is not None:
    generate(ctx)
  else:
    main(ctx)


def compute_weights():
  train_data, _, _ = make_data_split('melvinNoFF1Maxent.db')
  ldr = DataLoader(train_data, collate_fn=collate_noff1_unpacked, batch_size=4096)
  cnt = torch.zeros(len(commap))
  tot = 0
  
  for i, (com, maxent, srv, file) in enumerate(ldr):
    cnt[-1] += len(com)
    com = torch.cat(com)
    tot += len(com)
    cnt += torch.eye(len(commap))[com].sum(dim=0)
  
  print('frequencies', cnt / tot)
  print('class weights', (tot / cnt) / len(cnt))


if __name__ != '__main__':
  compute_weights()



