import argparse
import numpy as np
import datetime
import signal
import time
import sys
import io
import os
from multiprocessing import Process, Queue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.binomial import Binomial

from data import MelvinRandomDataLoader, TriggerLoader
from data import white_srv, commap, commap_sympy, ff1map, ff1max, decode
from loss import srv_soft_prediction, srv_prediction
from model import MelvinNet
from utils import load_setup, Object
from symbolic import check_setup, get_triggers_for_setup, MakeFF2


def print_result():
  print('mean inference time: %f ms' % (1000 * sum(exec_times) / len(exec_times)), file=sys.stderr)
  print('processed %d sequences' % (n_seq_batch * args.batch_size))
  print('processed %d trigger combinations, %d (%f%%) hot' % (sum(n_trig), 
    sum(n_trig_hot), 100 * sum(n_trig_hot) / (len(n_trig_hot) * args.batch_size)))
  #print('%d effective samples' % num_samples)
  #print('maxent confusion matrix: tp=%f tn=%f fp=%f fn=%f' % (tp/num_samples, tn/num_samples, 
  #  fp/num_samples, fn/num_samples))
  #print('SRV accuracy: %f' % (acc / num_samples))


def signal_handler(sig, frame):
  print_result()
  sys.exit(0)


#os.environ['NUMBAPRO_NVVM']='/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_NVVM']='/system/apps/biosoft/cuda-9.0/nvvm/lib64/libnvvm.so'
#os.environ['NUMBAPRO_LIBDEVICE']='/usr/local/cuda/nvvm/libdevice'
os.environ['NUMBAPRO_LIBDEVICE']='/system/apps/biosoft/cuda-9.0/nvvm/libdevice'

signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser(description='Evaluate model on a subset of the test set')
parser.add_argument('maxent', help='Path to maxent model without file extension. There must ' + 
                    'exist a [maxent].pt and a [maxent].txt file containing weights and ' + 
                    'hyperparameters respectively.')
parser.add_argument('srv', help='Path to SRV model without file extension. There must ' + 
                    'exist a [srv].pt and a [srv].txt file containing weights and ' + 
                    'hyperparameters respectively.')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for inference (default: 256, on a 4GB GPU a batch size' +
                    'of 24000 is possible with 600ms inference time per batch)')
parser.add_argument('--threshold', type=float, default=0.5, 
                    help='maxent threshold for subset (default: 0.5)')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 means CPU (default: 0)')
parser.add_argument('--seed', type=str, default='0xdeadbeef')
parser.add_argument('--dst-file', type=str, default='/publicwork/adler/positives')
parser.add_argument('--stats', action='store_true')

args = parser.parse_args()
models = []

device = torch.device(('cuda:%d' % args.gpu) if args.gpu >= 0 else 'cpu')
torch.manual_seed(int(args.seed, base=16))

binomial = Binomial(2 * ff1max, .5)
eye = torch.eye(len(ff1map), device=device)
zero = torch.zeros((len(ff1map),), device=device)


def consumer(queue, id):
  yey = np.arange(-ff1max, ff1max+1)
  #TODO: write checked samples into database
  #TODO: print time
  #TODO: print dim_vec vs SRV
  with open('consumer%d.txt' % id, 'w') as f:
    while True:
      c, t, s, sym = queue.get()
      setup = [commap_sympy[c_] for c_ in c]
      trigger = list(yey[t])
      f.write('%s trigger %s pred SRV %s\n' % (setup, trigger, s))
      f.flush()
      #true_maxent, true_srv, msg = check_setup(setup, trigger)
      true_maxent, true_srv, msg = check_setup(sym, trigger)
      f.write('true_maxent=%d true_srv=%s msg=%s\n' % (true_maxent, true_srv, msg))
      f.flush()
      
      if true_maxent:
        f.write(setup)
        f.write(trigger)
        f.write('true maxent=%s and srv=%s\n' % (true_maxent, true_srv))
        f.flush()
        
        if tuple(true_srv) in white_srv:
          f.write('\n----------------------------------------------\n')
          f.write('NEW MAXENT SRV FOUND\n')
          f.write('----------------------------------------------\n\n')
          f.write(setup, trigger)
          f.flush()


def ff1_to_torch(ff1_list):
  import random
  i = random.randint(0,len(ff1_list)-1)
  
  if i == 0 and ff1_list[0] == ():
    return zero
  else:
    return eye[list(ff1_list[i])].sum(dim=1)


def __ff1_to_torch(ff1_list):
  ff1_torch = []
  
  if ff1_list[0] == ():
    ff1_torch.append(zero)
    del(ff1_list[0])
  
  for ff1 in ff1_list:
    ff1_torch.append(eye[list(ff1)].sum(dim=1))
  
  return torch.stack(ff1_torch)


def get_ff1_for_com(com):
  #return com, eye[binomial.sample((com.batch_sizes[0],)).long()]
  com, com_len = nn.utils.rnn.pad_packed_sequence(com, batch_first=True)
  ff1 = []
  
  for i, (c, l) in enumerate(zip(com.data.cpu(), com_len)):
    ff1_list = get_triggers_for_setup([commap_sympy[c_] for c_ in c[:l]])
    ff1.append(ff1_to_torch(ff1_list))
  
  return com, torch.stack(ff1)


for model_name in [args.maxent, args.srv]:
  setup = load_setup(model_name + '.txt')
  ctx = Object()
  ctx.args = setup
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

white_srv_map = torch.zeros((12, 12, 12), dtype=torch.uint8).to(device)

for n, m, k in white_srv:
  white_srv_map[n, m, k] = 1

# ctx is set to SRV model context
model_maxent, model_srv = models
loader = MelvinRandomDataLoader(args.batch_size, device=device)
exec_times = []
n_trig_hot = []
n_trig = []
n_seq_batch = 0
n_seq_hot = 0
num_samples, acc = 0, 0
tp, tn, fp, fn = 0, 0, 0, 0

#queue = Queue()
#workers = [Process(target=consumer, args=(queue, i)) for i in range(10)]
#[w.start() for w in workers]

out_file = open(args.dst_file + '_gpu' + str(args.gpu) + '_' + datetime.datetime.now().strftime('%Y-%m-%d'), 'w')

mkff2 = MakeFF2(args.batch_size, device)

with torch.no_grad():
  for i, com in enumerate(loader):
    # load data to GPU and do inference
    start = time.perf_counter()
    
    # inference on LSTM part of the model (once for all FF1)
    hidden_maxent = model_maxent.forward_lstm(com, model.init_hidden(args.batch_size, device))
    hidden_srv = model_srv.forward_lstm(com, model.init_hidden(args.batch_size, device))
    
    trigger_loader = TriggerLoader(mkff2(com), args.batch_size)
    
    for j, (ff1, idx) in enumerate(trigger_loader):
      logit_maxent = model_maxent.forward_snn(ff1.float(), hidden_maxent[idx])
      logit_srv = model_srv.forward_snn(ff1.float(), hidden_srv[idx])
      
      # compute predictions
      pred_maxent = (logit_maxent >= 0.)
      
      if args.stats:
        pred_srv = srv_soft_prediction(ctx, logit_srv)+1
      else:
        pred_srv = srv_prediction(ctx, logit_srv).type(torch.int64)+1
      
      n, m, k = pred_srv.t()
      
      # create subset mask
      mask_thresh = (logit_maxent.squeeze(1).sigmoid() >= args.threshold)
      
      if args.stats:
        mask = torch.ones_like(mask_thresh)
      else:
        mask_inbounds = (n<12) & (m<12) & (k<12)
        mask_white = torch.ones_like(mask_inbounds)
        mask_white[mask_inbounds] = white_srv_map[n[mask_inbounds], m[mask_inbounds], k[mask_inbounds]]
        mask = mask_thresh & mask_white
      
      n_trig_hot.append(mask.long().sum().item())
      n_trig.append(ff1.shape[0])
      
      if n_trig_hot[-1] > 0:
        # unpack component sequences
        com_pad, com_len = nn.utils.rnn.pad_packed_sequence(com, batch_first=True)
        
        # download to CPU and convert to numpy
        srv = pred_srv[mask].data.cpu().numpy()
        com_pad = com_pad[idx][mask].data.cpu().numpy()
        com_len = com_len[idx][mask].data.cpu().numpy()
        ff1 = ff1[mask].data.cpu().numpy()
        num = torch.arange(mask.shape[0])[mask.cpu()].numpy()
        
        if args.stats:
          for s, c, l, f, n, p in zip(srv, com_pad, com_len, ff1, num, logit_maxent.squeeze(1).sigmoid()):
            out_file.write(decode(s, c[:l], f, True) + ' | {%f}\n' % p)
        else:
          for s, c, l, f, n in zip(srv, com_pad, com_len, ff1, num):
            out_file.write(decode(s, c[:l], f, False) + '\n')
          
        out_file.flush()
      exec_times.append(time.perf_counter() - start)
      
      if args.stats and sum(n_trig) > 2e7:
        # send SIGINT to self
        os.kill(os.getpid(), signal.SIGINT)
      
    n_seq_batch += 1

print_result()


