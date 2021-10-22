import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_noff1 import main, create_argparser
import valid_noff1
from multiprocessing import Process
from datetime import datetime
from time import sleep


def hidden_dim_search():
  # get default args and manipulate them
  args = create_argparser().parse_args([])
  args.valid_batches_max = 10
  args.save_model = True
  args.lambda_srv = 1.0
  args.logdir = '/publicwork/adler/melvin/runs'
  
  hidden_dim = [256, 128, 64, 32]
  lr = [0.01, 0.02, 0.05, 0.1]
  gpu = [0, 1, 2, 3]
  proc = []
  
  for h, l, g in zip(hidden_dim, lr, gpu):
    args.name = 'h%d_l%d' % (h, int(l*100))
    args.lstm_hidden_dim = h
    args.snn_hidden_dim = [h, h, h]
    args.lr = l
    args.gpu = 0 #g
    proc.append(Process(target=main, args=(args,)))
    proc[-1].start()
  
  [p.join() for p in proc]


def compare_srv_loss():
  # get default args and manipulate them
  args = create_argparser().parse_args([])
  
  args.valid_batches_max = 10
  args.save_model = True
  #args.logdir = '/publicwork/adler/melvin/runs'
  args.lstm_hidden_dim = 256
  args.snn_hidden_dim = [256] * 3
  args.epochs = 6
  args.lr = 0.01
  args.gpu = 0
  args.lambda_srv = 1.0
  args.srv_loss = 'poisson_binomial'
  proc = []
  
  for lambda_srv in [1.0]:
    args.name = args.srv_loss + '_%.1f' % lambda_srv
    args.lambda_srv = lambda_srv
    proc.append(Process(target=main, args=(args,)))
    proc[-1].start()
  
  [p.join() for p in proc]


def srvreg():
  # get default args and manipulate them
  args = create_argparser().parse_args([])
  
  name = datetime.now().strftime('%Y-%m-%dT%H-%M-%S') + '/srvreg_'
  args.logdir = '/home/tomte/projects/melvin/runs'
  args.lstm_hidden_dim = 256
  args.snn_hidden_dim = [256] * 3
  args.epochs = 6
  args.gpu = 0
  
  srv_loss = ['mse', 'poisson_binomial']
  lr = [0.001, 0.017]
  task = ['srv', 'srv']
  
  proc = []
  
  for t, s, l in zip(task, srv_loss, lr):
    args.name = name + s
    args.lr = l
    args.srv_loss = s
    args.task = t
    proc.append(Process(target=main, args=(args,)))
    proc[-1].start()
  
  [p.join() for p in proc]


def cross_validation():
  name = 'fold%02d'
  
  # training args
  args = create_argparser().parse_args([])
  args.logdir = '/publicwork/adler/melvin/ccv'
  args.lstm_hidden_dim = 1024
  args.lr = 0.1
  args.batch_size = 512
  args.log_interval = 100
  args.data = '/local00/bioinf/adler/melvinNoFF1Train.db'
  
  # validation args
  valid_args = valid_noff1.create_argparser().parse_args([])
  valid_args.batch_size = 4096
  valid_args.gpu = 0
  valid_args.data = '/local00/bioinf/adler/melvinNoFF1Train.db'
  
  train_job = []
  valid_job = []
  
  # start training jobs in parallel on 4 GPUs
  for fold in range(11):
    args.fold = fold
    args.name = name % fold
    args.gpu = fold % 4
    args.stdout = name % fold + '_train.log'
    
    # start training job
    train_job.append(Process(target=main, args=(args,)))
    train_job[-1].start()
  
  # wait until initial models are written to file
  sleep(15)
  
  # start validation jobs
  for fold in range(11):
    valid_args.gpu = fold % 4
    valid_args.stdout = name % fold + '_valid.log'
    valid_args.model_name = os.path.join(args.logdir, name % fold, 'model')
    valid_job.append(Process(target=valid_noff1.main, args=(valid_args,)))
    valid_job[-1].start()
  
  # wait for training jobs to finish and terminate validation jobs
  for t, v in zip(train_job, valid_job):
    t.join()
    sleep(300) # give validation job another 5 mins to finish
    v.terminate()
    v.join()
    
    if sys.version_info.major >= 3 and sys.version_info.minor >= 7:
      t.close()
      v.close()


def fold08_grid_search():
  #FIXME: get this to work!!!!
  name = 'f08gs_h%d_lr%f_bs%d'
  
  # training args
  args = create_argparser().parse_args([])
  args.logdir = '/publicwork/adler/melvin/grid'
  args.embedding_dim = 512
  h = [4096, 2048]
  lr = [0.1, 0.1]
  bs = [128, 128]
  args.log_interval = 100
  args.data = '/local00/bioinf/adler/melvinNoFF1Train.db'
  args.fold = 8
  
  train_job = []
  
  for i, (h_, lr_, bs_) in enumerate(zip(h, lr, bs)):
    args.name = name % (h_, lr_, bs_)
    args.lstm_hidden_dim = h_
    args.lr = lr_
    args.batch_size = bs_
    args.gpu = i % 4
    args.stdout = os.path.join(args.logdir, args.name, 'train.log')
    train_job.append(Process(target=main, args=(args,)))
    train_job[-1].start()
    print('job started')
  
  sleep(30)
  
  # validation args
  valid_args = valid_noff1.create_argparser().parse_args([])
  valid_args.batch_size = 4096
  valid_args.data = '/local00/bioinf/adler/melvinNoFF1Train.db'
  valid_job = []
  
  for i, (h_, lr_, bs_) in enumerate(zip(h, lr, bs)):
    valid_args.name = name % (h_, lr_, bs_)
    valid_args.gpu = i % 4
    valid_args.stdout = os.path.join(args.logdir, valid_args.name, 'valid.log')
    valid_args.model_name = os.path.join(args.logdir, valid_args.name, 'model')
    valid_job.append(Process(target=valid_noff1.main, args=(valid_args,)))
    valid_job[-1].start()
  
  # wait for training jobs to finish
  for j in train_job:
    j.join()
    
    if sys.version_info.major >= 3 and sys.version_info.minor >= 7:
      j.close()
  
  # give validation jobs another 10 mins
  sleep(600)
  
  # terminate validation jobs
  for j in valid_job:
    j.terminate()
    j.join()
    
    if sys.version_info.major >= 3 and sys.version_info.minor >= 7:
      j.close()


if __name__ == '__main__':
  fold08_grid_search()

