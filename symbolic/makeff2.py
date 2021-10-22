import os
import numba
from numba import cuda
import torch
import numpy as np
from cmath import pi, exp, sin, cos, sqrt

# integer constants
BS, LI, OAM_HOLO, REFLECTION, DP = range(5)
TERMS_INIT = 21
EPS = 1e-10

# map to functions and arguments according to data.commap
commap = torch.tensor([
  [BS,ord('a'),ord('b')], [LI,ord('a'),ord('b')], [BS,ord('a'),ord('c')], [LI,ord('a'),ord('c')],
  [BS,ord('a'),ord('d')], [LI,ord('a'),ord('d')], [BS,ord('a'),ord('e')], [LI,ord('a'),ord('e')],
  [BS,ord('a'),ord('f')], [LI,ord('a'),ord('f')], [BS,ord('b'),ord('c')], [LI,ord('b'),ord('c')],
  [BS,ord('b'),ord('d')], [LI,ord('b'),ord('d')], [BS,ord('b'),ord('e')], [LI,ord('b'),ord('e')],
  [BS,ord('b'),ord('f')], [LI,ord('b'),ord('f')], [BS,ord('c'),ord('d')], [LI,ord('c'),ord('d')],
  [BS,ord('c'),ord('e')], [LI,ord('c'),ord('e')], [BS,ord('c'),ord('f')], [LI,ord('c'),ord('f')],
  [BS,ord('d'),ord('e')], [LI,ord('d'),ord('e')], [BS,ord('d'),ord('f')], [LI,ord('d'),ord('f')],
  [BS,ord('e'),ord('f')], [LI,ord('e'),ord('f')], [REFLECTION,ord('a'),0], [DP,ord('a'),0],
  [OAM_HOLO,ord('a'),-5], [OAM_HOLO,ord('a'),-4], [OAM_HOLO,ord('a'),-3],
  [OAM_HOLO,ord('a'),-2], [OAM_HOLO,ord('a'),-1], [OAM_HOLO,ord('a'),1],
  [OAM_HOLO,ord('a'),2], [OAM_HOLO,ord('a'),3], [OAM_HOLO,ord('a'),4],
  [OAM_HOLO,ord('a'),5], [REFLECTION,ord('b'),0], [DP,ord('b'),0],
  [OAM_HOLO,ord('b'),-5], [OAM_HOLO,ord('b'),-4], [OAM_HOLO,ord('b'),-3],
  [OAM_HOLO,ord('b'),-2], [OAM_HOLO,ord('b'),-1], [OAM_HOLO,ord('b'),1],
  [OAM_HOLO,ord('b'),2], [OAM_HOLO,ord('b'),3], [OAM_HOLO,ord('b'),4],
  [OAM_HOLO,ord('b'),5], [REFLECTION,ord('c'),0], [DP,ord('c'),0],
  [OAM_HOLO,ord('c'),-5], [OAM_HOLO,ord('c'),-4], [OAM_HOLO,ord('c'),-3],
  [OAM_HOLO,ord('c'),-2], [OAM_HOLO,ord('c'),-1], [OAM_HOLO,ord('c'),1],
  [OAM_HOLO,ord('c'),2], [OAM_HOLO,ord('c'),3], [OAM_HOLO,ord('c'),4],
  [OAM_HOLO,ord('c'),5], [REFLECTION,ord('d'),0], [DP,ord('d'),0],
  [OAM_HOLO,ord('d'),-5], [OAM_HOLO,ord('d'),-4], [OAM_HOLO,ord('d'),-3],
  [OAM_HOLO,ord('d'),-2], [OAM_HOLO,ord('d'),-1], [OAM_HOLO,ord('d'),1],
  [OAM_HOLO,ord('d'),2], [OAM_HOLO,ord('d'),3], [OAM_HOLO,ord('d'),4],
  [OAM_HOLO,ord('d'),5], [REFLECTION,ord('e'),0], [DP,ord('e'),0],
  [OAM_HOLO,ord('e'),-5], [OAM_HOLO,ord('e'),-4], [OAM_HOLO,ord('e'),-3],
  [OAM_HOLO,ord('e'),-2], [OAM_HOLO,ord('e'),-1], [OAM_HOLO,ord('e'),1],
  [OAM_HOLO,ord('e'),2], [OAM_HOLO,ord('e'),3], [OAM_HOLO,ord('e'),4],
  [OAM_HOLO,ord('e'),5], [REFLECTION,ord('f'),0], [DP,ord('f'),0],
  [OAM_HOLO,ord('f'),-5], [OAM_HOLO,ord('f'),-4], [OAM_HOLO,ord('f'),-3],
  [OAM_HOLO,ord('f'),-2], [OAM_HOLO,ord('f'),-1], [OAM_HOLO,ord('f'),1],
  [OAM_HOLO,ord('f'),2], [OAM_HOLO,ord('f'),3], [OAM_HOLO,ord('f'),4],
  [OAM_HOLO,ord('f'),5]], dtype=torch.int8)

# OAMs of the initial state
oam = torch.tensor([[
  [-1, -1,  1,  1],
  [-1,  0,  0,  1],
  [-1,  1, -1,  1],
  [-1,  1, -1,  1],
  [-1,  1,  0,  0],
  [-1,  1,  1, -1],
  [ 0,  0,  0,  0],
  [ 0,  1, -1,  0],
  [ 0,  0, -1,  1],
  [ 0,  0,  0,  0],
  [ 0,  0,  1, -1],
  [ 1,  1, -1, -1],
  [ 1, -1, -1,  1],
  [ 1, -1,  0,  0],
  [ 1, -1,  1, -1],
  [-1, -1,  1,  1],
  [-1,  0,  0,  1],
  [-1,  1, -1,  1],
  [ 0,  0,  0,  0],
  [ 0,  1, -1,  0],
  [ 1,  1, -1, -1]]], dtype=torch.int8)

# coefficients [real, imag] of the initial state
coef = torch.tensor([[
  [1, 0], [2, 0], [2, 0], [2, 0], [2, 0], [2, 0], [1, 0], [2, 0], [2, 0], [2, 0], [2, 0], [1, 0], 
  [2, 0], [2, 0], [2, 0], [1, 0], [2, 0], [2, 0], [1, 0], [2, 0], [1, 0]]], dtype=torch.float)

path = torch.tensor([[
  [ord('a'), ord('a'), ord('b'), ord('b')],
  [ord('a'), ord('a'), ord('b'), ord('b')],
  [ord('a'), ord('a'), ord('b'), ord('b')],
  [ord('a'), ord('b'), ord('c'), ord('d')],
  [ord('a'), ord('b'), ord('c'), ord('d')],
  [ord('a'), ord('b'), ord('c'), ord('d')],
  [ord('a'), ord('a'), ord('b'), ord('b')],
  [ord('a'), ord('a'), ord('b'), ord('b')],
  [ord('a'), ord('b'), ord('c'), ord('d')],
  [ord('a'), ord('b'), ord('c'), ord('d')],
  [ord('a'), ord('b'), ord('c'), ord('d')],
  [ord('a'), ord('a'), ord('b'), ord('b')],
  [ord('a'), ord('b'), ord('c'), ord('d')],
  [ord('a'), ord('b'), ord('c'), ord('d')],
  [ord('a'), ord('b'), ord('c'), ord('d')],
  [ord('c'), ord('c'), ord('d'), ord('d')],
  [ord('c'), ord('c'), ord('d'), ord('d')],
  [ord('c'), ord('c'), ord('d'), ord('d')],
  [ord('c'), ord('c'), ord('d'), ord('d')],
  [ord('c'), ord('c'), ord('d'), ord('d')],
  [ord('c'), ord('c'), ord('d'), ord('d')]]], dtype=torch.int8)


@cuda.jit(device=True, inline=True)
def print_term(coef, path, oam, i, j):
  print(coef[i,j,0], coef[i,j,1], path[i,j,0], 
    oam[i,j,0], path[i,j,1], oam[i,j,1], path[i,j,2], oam[i,j,2], path[i,j,3], oam[i,j,3])


@cuda.jit(device=True, inline=True)
def copy(coef, path, oam, n_terms, i, j):
  if not n_terms[i] < oam.shape[1]:
    print('OOM')
    path[i,0,0] = -1 # misuse this element as stop signal
    return -1
  
  for k in range(4):
    path[i,n_terms[i],k] = path[i,j,k]
    oam[i,n_terms[i],k] = oam[i,j,k]
  
  coef[i,n_terms[i],0] = coef[i,j,0]
  coef[i,n_terms[i],1] = coef[i,j,1]
  n_terms[i] += 1
  return n_terms[i]-1


@cuda.jit(device=True, inline=True)
def delete(coef, path, oam, n_terms, i, j):
  if j != n_terms[i]-1:
    for k in range(4):
      path[i,j,k] = path[i,n_terms[i]-1,k]
      oam[i,j,k] = oam[i,n_terms[i]-1,k]
    
    coef[i,j,0] = coef[i,n_terms[i]-1,0]
    coef[i,j,1] = coef[i,n_terms[i]-1,1]
  n_terms[i] -= 1


@cuda.jit(device=True, inline=True)
def oam_holo(coef, path, oam, n_terms, i, arg1, arg2):
  for j in range(n_terms[i]):
    for k in range(4):
      if path[i,j,k] == arg1:
        oam[i,j,k] += arg2


@cuda.jit(device=True, inline=True)
def reflection(coef, path, oam, n_terms, i, arg1):
  for j in range(n_terms[i]):
    for k in range(4):
      if path[i,j,k] == arg1:
        c = complex(coef[i,j,0], coef[i,j,1]) * 1j
        
        if abs(c.real) < EPS and abs(c.imag) <  EPS:
          delete(coef, path, oam, n_terms, i, j)
          break
        
        coef[i,j,0], coef[i,j,1] = c.real, c.imag
        oam[i,j,k] *= -1


@cuda.jit(device=True, inline=True)
def dp(coef, path, oam, n_terms, i, arg1):
  for j in range(n_terms[i]):
    for k in range(4):
      if path[i,j,k] == arg1:
        l = oam[i,j,k]
        c = complex(coef[i,j,0], coef[i,j,1]) * 1j * exp(1j * l * pi)
        
        if abs(c.real) < EPS and abs(c.imag) <  EPS:
          delete(coef, path, oam, n_terms, i, j)
          break
        
        coef[i,j,0], coef[i,j,1] = c.real, c.imag
        oam[i,j,k] = -l


@cuda.jit(device=True, inline=True)
def __bs(coef, path, oam, n_terms, i, j, arg1, arg2):
# FIXME: BS and LI let the number of terms explode. this means that sequences with more BS and LI
# components have significantly higher memory demand than other samples. this calls for dynamic 
# memory management, which is not (yet) supported by numba.cuda, i.e. consider rewriting the code 
# in native cuda and add dynamic memory management. 
  term = cuda.local.array(16, numba.int32)
  term[0] = j
  term_len = 1
  
  for k in range(4):
    for t in range(term_len):
      j = term[t]
      
      if j == -1:
        continue
      
      p = path[i,j,k]
      
      if p == arg1:
        new_p = arg2
      elif p == arg2:
        new_p = arg1
      else:
        new_p = -1
      
      if new_p != -1:
        m = copy(coef, path, oam, n_terms, i, j)
        
        if m == -1:
          return False
        
        term[term_len] = m
        term_len += 1
        
        # go through, path name changes
        c = complex(coef[i,m,0], coef[i,m,1]) * sqrt(2)/2
        
        if abs(c.real) < EPS and abs(c.imag) <  EPS:
          n_terms[i] -= 1
          term_len -= 1
        else:
          coef[i,m,0], coef[i,m,1] = c.real, c.imag
          path[i,m,k] = new_p
        
        # reflection, keep path name
        c = complex(coef[i,j,0], coef[i,j,1]) * 1j * sqrt(2)/2
        
        if abs(c.real) < EPS and abs(c.imag) <  EPS:
          delete(coef, path, oam, n_terms, i, j)
          term[t] = -1
          term[term_len-1] = j
        else:
          coef[i,j,0], coef[i,j,1] = c.real, c.imag
          oam[i,j,k] *= -1
  
  return True


@cuda.jit(device=True, inline=True)
def bs(coef, path, oam, n_terms, i, arg1, arg2):
  n = n_terms[i]
  
  for j in range(n):
    if not __bs(coef, path, oam, n_terms, i, j, arg1, arg2):
      break


@cuda.jit(device=True, inline=True)
def __li(coef, path, oam, n_terms, i, j, arg1, arg2):
  term = cuda.local.array(16, numba.int32)
  
  term[0] = j
  term_len = 1
  
  for k in range(4):
    for t in range(term_len):
      j = term[t]
      
      if j == -1:
        continue
      
      p = path[i,j,k]
      
      if p == arg1:
        new_p = arg2
        flip_sign = False
      elif p == arg2:
        new_p = arg1
        flip_sign = True
      else:
        new_p = -1
      
      if new_p != -1:
        m = copy(coef, path, oam, n_terms, i, j)
        
        if m == -1:
          return False
        
        term[term_len] = m
        term_len += 1
        l = oam[i,j,k]
        
        # go through, keep path name
        c = complex(coef[i,m,0], coef[i,m,1])
        c *= cos(l * pi/2)**2 * (-1 if flip_sign else 1)
        
        if abs(c.real) < EPS and abs(c.imag) <  EPS:
          n_terms[i] -= 1
          term_len -= 1
        else:
          coef[i,m,0], coef[i,m,1] = c.real, c.imag
        
        # reflection, path name changes
        c = complex(coef[i,j,0], coef[i,j,1])
        c *= 1j * sin(l * pi/2)**2
        
        if abs(c.real) < EPS and abs(c.imag) <  EPS:
          delete(coef, path, oam, n_terms, i, j)
          term[t] = -1
          term[term_len-1] = j
        else:
          coef[i,j,0], coef[i,j,1] = c.real, c.imag
          path[i,j,k] = new_p
          oam[i,j,k] = -l
  
  return True


@cuda.jit(device=True, inline=True)
def li(coef, path, oam, n_terms, i, arg1, arg2):
  n = n_terms[i]
  
  for j in range(n):
    if not __li(coef, path, oam, n_terms, i, j, arg1, arg2):
      break


@cuda.jit
def makeff2_kernel(com, batch_size, commap, coef, path, oam, n_terms, ffn):
  # com and batch_size are data and batch_sizes from a torch.nn.utils.rnn.PackedSequence
  # commap is a uint8 array of size [101, 3] mapping com to function calls
  # coef is a float array of size [batch_size, terms_max, 2] representing complex numbers
  # path and oam are int8 arrays of size [batch_size, terms_max, 4]
  # n_terms is a long array of size [batch_size] holding the number non-zero terms per sample
  
  start = cuda.grid(1)
  stride = cuda.gridsize(1)
  p = 0
  
  # compute state
  for t in range(batch_size.shape[0]):
    for i in range(start, batch_size[t], stride):
      if path[i,0,0] == -1: # check if sample ran out of terms
        continue
      
      func, arg1, arg2 = commap[com[p+i]]
      
      if func == BS:
        bs(coef, path, oam, n_terms, i, arg1, arg2)
      elif func == LI:
        li(coef, path, oam, n_terms, i, arg1, arg2)
      elif func == OAM_HOLO:
        oam_holo(coef, path, oam, n_terms, i, arg1, arg2)
      elif func == REFLECTION:
        reflection(coef, path, oam, n_terms, i, arg1)
      elif func == DP:
        dp(coef, path, oam, n_terms, i, arg1)
    
    p += batch_size[t]
  
  # for all a,b,c,d terms, collect distinct OAMs in a
  for i in range(start, batch_size[t], stride):
    if path[i,0,0] == -1:
      continue
    
    for j in range(n_terms[i]):
      # FIXME: canceling coefficients are not (yet) detected
      # sum of squares even works with e,f
      sos = (path[i,j,0]-97)**2 + (path[i,j,1]-97)**2 + \
        (path[i,j,2]-97)**2 + (path[i,j,3]-97)**2 # ord('a') == 97
      
      if sos == 14: # a,b,c,d term
        for k in range(4):
          if path[i,j,k] == 97: # and (abs(coef[i,j,0]) > EPS or abs(coef[i,j,1]) > EPS):
            l = oam[i,j,k]
            
            #print_term(coef, path, oam, i, j)
            
            if abs(l) > 25:
              path[i,0,0] = -2
            else:
              ffn[i,l+25] = 1
            break


def to_sympy(coef, path, oam, n_terms, i):
  s = []
  from setup import a, b, c, d, e, f
  
  for j in range(n_terms[i]):
    s.append('(%f+%fj)*%c[%d]*%c[%d]*%c[%d]*%c[%d]' % (coef[i,j,0], coef[i,j,1], path[i,j,0], 
      oam[i,j,0], path[i,j,1], oam[i,j,1], path[i,j,2], oam[i,j,2], path[i,j,3], oam[i,j,3]))
  
  s = '+'.join(s)
  return sp.simplify(eval(s))


if __name__ == '__main__':
  os.environ['NUMBAPRO_NVVM']='/usr/local/cuda/nvvm/lib64/libnvvm.so'
  os.environ['NUMBAPRO_LIBDEVICE']='/usr/local/cuda/nvvm/libdevice'
  
  import sys
  sys.path.append('..')
  from data import commap_sympy
  from trigger import distinct_ffn, make_ff2, ff1
  from setup import sympify
  import sympy as sp
  
  #seq = ['Reflection(XXX,e)', 'OAMHolo(XXX,a,3)', 'LI(XXX,b,c)', 'OAMHolo(XXX,d,5)', 
  #       'OAMHolo(XXX,a,2)', 'OAMHolo(XXX,e,5)']
  #seq = ['OAMHolo(XXX,c,-3)', 'OAMHolo(XXX,a,3)', 'OAMHolo(XXX,f,-5)', 'OAMHolo(XXX,c,2)', 
  #       'LI(XXX,a,c)', 'BS(XXX,a,b)', 'OAMHolo(XXX,b,-3)', 'OAMHolo(XXX,e,-4)']
  
  # FIXME: this sample is not consistent
  seq = ['OAMHolo(XXX,c,-3)', 'BS(XXX,a,e)', 'LI(XXX,b,d)', 'OAMHolo(XXX,c,2)', 
         'LI(XXX,a,c)', 'BS(XXX,c,e)', 'OAMHolo(XXX,b,-3)', 'OAMHolo(XXX,e,-4)']
  #seq = ['LI(XXX,a,c)', 'BS(XXX,a,b)']
  
  # FIXME: states differ by only one term each
  #seq = ['LI(XXX,a,e)']
  
  seq = 'BS(XXX,a,b)'
  
  for i in range(1): 
  #for seq in commap_sympy:
    print(seq)
    #seq = [seq]
    com = torch.tensor([commap_sympy.index(s) for s in seq], dtype=torch.uint8, device='cuda')
    batch_size = torch.tensor([1] * len(seq), dtype=torch.uint8, device='cuda')
    terms_max = 2048
    n_terms = torch.tensor([21], dtype=torch.int32, device='cuda')
    
    commap = commap.to('cuda')
    _coef = torch.cat([coef.to('cuda'), torch.zeros((1, terms_max-21, 2), dtype=torch.float, device='cuda')], dim=1)
    _path = torch.cat([path.to('cuda'), torch.zeros((1, terms_max-21, 4), dtype=torch.int8, device='cuda')], dim=1)
    _oam = torch.cat([oam.to('cuda'), torch.zeros((1, terms_max-21, 4), dtype=torch.int8, device='cuda')], dim=1)
    #coef = torch.cat([coef[:,3:4].to('cuda'), torch.zeros((1, terms_max-1, 2), dtype=torch.float, device='cuda')], dim=1)
    #path = torch.cat([path[:,3:4].to('cuda'), torch.zeros((1, terms_max-1, 4), dtype=torch.int8, device='cuda')], dim=1)
    #oam = torch.cat([oam[:,3:4].to('cuda'), torch.zeros((1, terms_max-1, 4), dtype=torch.int8, device='cuda')], dim=1)
    ffn = torch.zeros((1, 51), dtype=torch.uint8, device='cuda')
    
    makeff2_kernel(com, batch_size, commap, _coef, _path, _oam, n_terms, ffn)
    psi_kernel = sp.expand(to_sympy(_coef, _path, _oam, n_terms, 0))
    psi_xuemei = sp.expand(sp.simplify(sympify(seq, 1)))
  
    #print(psi_kernel)
    #print('')
    #print(psi_xuemei)
    #print('')
    print('diff', psi_kernel - psi_xuemei)
    #print(sp.simplify(sympify(seq, 1)))
    
    if path[0,0,0] == -1:
      print('sample ran out of terms')
    
    #print(n_terms)
    #print(coef[:,:n_terms])
    #print(path[:,:n_terms])
    #print(oam[:,:n_terms])
    #print(ffn)
    print('kernel', set(torch.arange(-25, 26).to('cuda')[ffn[0]].cpu().numpy()))
    print('xuemei', distinct_ffn(make_ff2(sympify(seq, 1)), ff1))
  


# this is a minimal example of using numba.cuda with torch
if False : #__name__ == '__main__':
  # do this if not run in conda
  #os.environ['NUMBAPRO_NVVM']='/usr/local/cuda/nvvm/lib64/libnvvm.so'
  #os.environ['NUMBAPRO_LIBDEVICE']='/usr/local/cuda/nvvm/libdevice'

  # check if numba works on GPU with numpy and torch

  # this is how cuda kernel
  @cuda.jit(device=True, inline=True)
  def write(out, val, i):
    out[i] = val
    return 'asdf'

  @cuda.jit
  def add(x, y, out):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    for i in range(start, x.shape[0], stride):
      #out[i] = x[i] + y[i]
      ret = write(out, x[i] + y[i], i)
      print(ret)

  a = torch.ones((10,), device='cuda')
  b = a * 2
  out = torch.zeros_like(a)
  add[1,32](a, b, out)
  print(out)

