import numpy as np
from numba import jit
import argparse
import matplotlib.pyplot as plt
from check_fold import SrvCounter, predict
from data import white_srv as white_srv_data
from math import sqrt, ceil

def myload(file):
  a = []
  
  with open(file, 'rb') as f:
    while True:
      try:
        a.append(np.load(f))
      except OSError:
        break
  
  return np.vstack(a)


def make_white_srv(fold, special=False):
  mposs_srv = set()
  
  if fold == 9:
    fold = [9,10,11,12]
  
  if type(fold) is not list:
    fold = [fold]
  
  for n in fold:
    for m in range(ceil(sqrt(n)), n+1):
      for k in range(2, m+1):
        mposs_srv.add((n, m, k))
  
  return sorted(mposs_srv)


@jit
def validate_loop(data, threshold, radius, white_srv, zero_fold=False):
  tp = tn = fp = fn = 0
  
  ctr = SrvCounter()
  s_all = set()
  s_found = set()
  
  for i, (maxent, maxent_pred, n, m, k, n_pred, m_pred, k_pred) in enumerate(data):
    pred = predict(maxent_pred, n_pred, m_pred, k_pred, threshold, radius, 
                   white_srv_data if zero_fold else white_srv)
    #print('prediction:', pred, 'data:', maxent, maxent_pred, n, m, k, n_pred, m_pred, k_pred)
    
    if pred:
      if maxent:
        tp += 1
        
        ctr.add((n,m,k), True)
        #s_found.add((n, m ,k))
        #s_all.add((n, m, k))
      else:
        fp += 1
    else:
      if maxent:
        fn += 1
        
        ctr.add((n,m,k), False)
        #s_all.add((n, m, k))
      else:
        tn += 1
  
  rr = 0 if zero_fold else ctr() #len(s_found)/len(s_all)
  
  return tp, tn, fp, fn, ctr.good(), len(ctr)



#t_range = np.arange(0.4, 1.0, 0.05)
#r_range = np.arange(0.5, 5.0, 0.25)

thresh = 2 #corresponds to 0.5
srvrad = 10 # corresponds to 3
#srvrad = 2 # corresponds to 1

files = ['fold%02d_plot.npy' % i for i in range(1,9)] + ['testOOD_plot.npy']
files_raw = ['fold%02d.npy' % i for i in range(1,9)] + ['testOOD.npy']
data = [np.load(f) for f in files]
data_raw = [myload(f) for f in files_raw]
labels = ['0,1'] + ['%d' % i for i in range(2,9)] + ['9-12']

# data is a list of length 9 containing arrays of shape (5, 12, 18)
# dim one is tnr, tpr, speedup, rediscovery ratio, bacc
# dim two is sigmoid threshold
# dim three is SRV radius

x = np.arange(len(data))
x[-1] = 9
width = 0.2
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots()
data[0][1,thresh,srvrad] = 0


if False: # debug
  ax.bar(x + width * 0, x, width, label='TNR', color='C1')
  ax.bar(x + width * 1, x, width, label='TPR', color='C0')
  ax.bar(x + width * 2, x, width, label='Precision', color='C2')
  ax.bar(x + width * 3, x,  width, label='Rediscovery Ratio', color='C3')

  ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  ax.set_xticks(x + 3/2 * width)
  ax.set_xlabel('Cluster')
  ax.set_xticklabels(labels)
  ax.set_aspect(5)
  #ax.set_ylim(0, 1.19)

  plt.savefig('histogram_debug.pdf', bbox_inches='tight')

  exit()



# calc bar heights
tnr_bar = [d[0,thresh,srvrad] for d in data]
tpr_bar = [d[1,thresh,srvrad] for d in data]
ppv_bar = [d[2,thresh,srvrad] for d in data]
rr_bar  = [d[3,thresh,srvrad] for d in data]

# calc error bars
conf = [validate_loop(r, 0.5, 3.0, make_white_srv(i+1), i+1 < 2) for i, r in enumerate(data_raw)]
# structure of conf is [(TP, TN, FP, FN, RR_GOOD, RR_TOTAL), ...]
z = 1.96
tnr_n = [c[1] + c[2] for c in conf]
tnr_error_bar = [z / n * sqrt(c[1]*c[2]/n) for c, n in zip(conf, tnr_n)]
tpr_n = [c[0] + c[3] for c in conf]
tpr_error_bar = [z / n * sqrt(c[0]*c[3]/n) for c, n in zip(conf, tpr_n)]
ppv_n = [c[0] + c[2] for c in conf]
ppv_error_bar = [z / n * sqrt(c[0]*c[2]/n) for c, n in zip(conf, ppv_n)]
rr_error_bar = [z / c[5] * sqrt(c[4]*(c[5]-c[4]) / c[5]) for c in conf]

tnr_error_bar_plus = [e if v+e <= 1 else 1-v for e, v in zip(tnr_error_bar, tnr_bar)]
tpr_error_bar_plus = [e if v+e <= 1 else 1-v for e, v in zip(tpr_error_bar, tpr_bar)]
ppv_error_bar_plus = [e if v+e <= 1 else 1-v for e, v in zip(ppv_error_bar, ppv_bar)]
rr_error_bar_plus = [e if v+e <= 1 else 1-v for e, v in zip(rr_error_bar, rr_bar)]

tnr_error_bar_minus = [e if v-e >= 0 else v for e, v in zip(tnr_error_bar, tnr_bar)]
tpr_error_bar_minus = [e if v-e >= 0 else v for e, v in zip(tpr_error_bar, tpr_bar)]
ppv_error_bar_minus = [e if v-e >= 0 else v for e, v in zip(ppv_error_bar, ppv_bar)]
rr_error_bar_minus = [e if v-e >= 0 else v for e, v in zip(rr_error_bar, rr_bar)]

#tpr_error_bar_plus[0] = None
#tpr_error_bar_minus[0] = None
#rr_error_bar_plus[0] = None
#rr_error_bar_minus[0] = None

tnr_error_bar = [tnr_error_bar_minus, tnr_error_bar_plus]
tpr_error_bar = [tpr_error_bar_minus, tpr_error_bar_plus]
ppv_error_bar = [ppv_error_bar_minus, ppv_error_bar_plus]
rr_error_bar = [rr_error_bar_minus, rr_error_bar_plus]

print('x', x)


#ax.bar(x + width * 0, tnr_bar, width, yerr=tnr_error_bar, capsize=3, label='TNR', color='C1')
#ax.bar(x + width * 1, tpr_bar, width, yerr=tpr_error_bar, capsize=3, label='TPR', color='C0')
#ax.bar(x + width * 2, ppv_bar, width, yerr=ppv_error_bar, capsize=3, label='Precision', color='C2')
#ax.bar(x + width * 3,  rr_bar, width, yerr=rr_error_bar,  capsize=3, label='Rediscovery Ratio', color='C3')

print(x + width * 0, tnr_bar, width)
print(x + width * 1, tpr_bar, width)
print(x + width * 2, ppv_bar, width)
print(x + width * 3,  rr_bar, width)

ax.bar(x + width * 0, tnr_bar, width, label='TNR', color='C1')
ax.bar(x + width * 1, tpr_bar, width, label='TPR', color='C0')
ax.bar(x + width * 2, ppv_bar, width, label='Precision', color='C2')
ax.bar(x + width * 3,  rr_bar, width, label='Rediscovery Ratio', color='C3')


ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xticks(x + 3/2 * width)
ax.set_xlabel('Cluster')
ax.set_xticklabels(labels)
ax.set_aspect(5)
#ax.set_ylim(0, 1.19)

plt.savefig('histogram_with_ppv_r3.pdf', bbox_inches='tight')

