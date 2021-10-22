import torch
import sympy as sp

from .trigger import check_setup, get_triggers_for_setup, get_all_triggers
from .setup import a, b, c, d, e, f, sympify
from .makeff2 import commap, coef, path, oam, makeff2_kernel, to_sympy, TERMS_INIT

from .trigger import make_ff2, trigger

from numba import cuda

class MakeFF2(object):
  def __init__(self, batch_size, device=torch.device('cuda'), terms_max=2**16):
    # init makeff2 on device (must be a GPU). it only works on one device at a time. 
    # we do this so we can reuse memory across kernel calls to save some overhead
    cuda.select_device(device.index)
    self.batch_size = batch_size
    self.device = device
    self.coef = torch.zeros((batch_size, terms_max, 2), dtype=torch.float, device=device)
    self.path = torch.zeros((batch_size, terms_max, 4), dtype=torch.int8, device=device)
    self.oam  = torch.zeros((batch_size, terms_max, 4), dtype=torch.int8, device=device)
    self.ffn = torch.zeros((batch_size, 2 * 25 + 1), dtype=torch.uint8, device=device)
    self.n_terms = torch.tensor([TERMS_INIT] * batch_size, dtype=torch.int32, device=device)
  
  
  def __del__(self):
    cuda.close()


  def __call__(self, pseq):
    # clean memory
    self.coef[:,:TERMS_INIT] = coef.to(self.device).repeat(self.batch_size, 1, 1)
    self.coef[:,TERMS_INIT:] = 0
    self.path[:,:TERMS_INIT] = path.to(self.device).repeat(self.batch_size, 1, 1)
    self.path[:,TERMS_INIT:] = 0
    self.oam[:,:TERMS_INIT]  = oam.to(self.device).repeat(self.batch_size, 1, 1)
    self.oam[:,TERMS_INIT:]  = 0
    self.ffn[:,:] = 0
    self.n_terms[:] = TERMS_INIT
    
    # call makeff2 kernel
    makeff2_kernel[1,256](pseq.data.type(torch.uint8), 
                          pseq.batch_sizes.to(self.device).type(torch.long), 
                          commap.to(self.device), self.coef, self.path, 
                          self.oam, self.n_terms, self.ffn)
    return self.ffn
  
  
  def __len__(self):
    return self.batch_size
  
  
  def __getitem__(self, i):
    s = []
    
    for j in range(self.n_terms[i].item()):
      s.append('(%f+%fj)*%c[%d]*%c[%d]*%c[%d]*%c[%d]' % (self.coef[i,j,0], self.coef[i,j,1], 
        self.path[i,j,0], self.oam[i,j,0], self.path[i,j,1], self.oam[i,j,1], 
        self.path[i,j,2], self.oam[i,j,2], self.path[i,j,3], self.oam[i,j,3]))
    
    return eval('+'.join(s))


def benchmark(terms_max, batch_size, n_loops):
  import os
  from time import perf_counter
  import sys
  sys.path.append('..')
  from data import MelvinRandomDataLoader
  import numpy as np
  
  os.environ['NUMBAPRO_NVVM']='/usr/local/cuda/nvvm/lib64/libnvvm.so'
  os.environ['NUMBAPRO_LIBDEVICE']='/usr/local/cuda/nvvm/libdevice'
  
  device = torch.device('cuda')
  loader = MelvinRandomDataLoader(batch_size, device=device)
  exec_times = []
  trigger = []
  mkff2 = MakeFF2(batch_size, device, terms_max)
  
  for i, com in enumerate(loader):
    if i >= n_loops:
      break
    
    ffn = mkff2(com)
    # do not count the first call (compilation overhead)
    print(mkff2[0])
    exec_times.append(perf_counter())
  
  exec_times = np.array(exec_times)
  return (exec_times[1:] - exec_times[:-1]).mean()#, trigger


if __name__ == '__main__':
  print(benchmark(2**16, 2**11, 10))
  
  
  
  
  
  
  
