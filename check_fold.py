import numpy as np
from numba import jit, njit
import argparse
import sys
from math import sqrt, ceil
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from data import white_srv as white_srv_data


class SrvCounter(object):
  def __init__(self, thresh=0.2):
    self.thresh = thresh
    self.d = dict()
  
  def add(self, key, found):
    if not key in self.d:
      self.d[key] = [0, 0]
    
    self.d[key][0] += 1
    
    if found:
      self.d[key][1] += 1
  
  def __call__(self):
    return self.good() / len(self)
  
  
  def __len__(self):
    return len(self.d)
  
  
  def good(self):
    good = 0
    
    for v in self.d.values():
      if v[1] / v[0] >= self.thresh:
        good += 1
    
    return good
  
  def __str__(self):
    return self.d.__str__()


@jit
def dist(n, m, k, x, y, z):
  return sqrt((n-x)**2 + (m-y)**2 + (k-z)**2)


def make_white_srv(fold, special=False):
  if fold==99:
    return sorted([(6,3,3), (6,5,2), (7,5,3), (7,5,4), (7,7,4)])
  
  mposs_srv = set()
  
  if type(fold) is not list:
    fold = [fold]
  
  for n in fold:
    for m in range(ceil(sqrt(n)), n+1):
      for k in range(2, m+1):
        mposs_srv.add((n, m, k))
  
  return sorted(mposs_srv)


@jit
def predict(maxent, n, m, k, threshold, radius, white_srv):
  is_white = False
  
  for n_, m_, k_ in white_srv:
    if dist(n, m, k, n_, m_, k_) <= radius:
      is_white = True
      break
  
  if not is_white:
    return 0.
  
  return float(maxent >= threshold)


def _predict_(data, thresh, radius, white_srv):
  is_white = np.zeros((data.shape[0],), dtype=np.bool)
  
  for n, m, k in white_srv:
    is_white |= (np.linalg.norm(data[:,5:8] - np.array(((n, m, k),)), axis=1) < radius)
  
  return is_white & (data[:,1] >= thresh)


def _validate_loop_(data, thresh, radius, white_srv, zero_fold=False):
  pred = _predict_(data, thresh, radius, white_srv)
  truth = data[:,0].astype(np.bool)
  is_white = np.zeros((data.shape[0],), dtype=np.bool)
  
  for n, m, k in white_srv:
    is_white |= (data[:,2:5] == np.array(((n,m,k),))).all(1)
  
  truth &= is_white
  
  
  tp = (pred & truth).sum()
  fp = (pred & np.logical_not(truth)).sum()
  fn = (np.logical_not(pred) & truth).sum()
  tn = (np.logical_not(pred) & np.logical_not(truth)).sum()
  
  if tp > 0:
    rr = np.unique(data[pred & truth,2:5], axis=0) / len(white_srv)
  
  return tp, tn, fp, fn, 0


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
  
  return tp, tn, fp, fn, rr


def validate(args, data, threshold, radius):
  #tp = tn = fp = fn = 0
  white_srv = make_white_srv(args.fold)
  return _validate_loop_(data, threshold, radius, white_srv, all(f < 2 for f in args.fold))


@jit
def conf_table(data, threshold, radius, white_srv):
  tp = tn = fp = fn = 0
  
  for maxent, maxent_pred, n, m, k, n_pred, m_pred, k_pred in data:
    pred = predict(maxent_pred, n_pred, m_pred, k_pred, \
                   threshold, radius, white_srv)
    
    if pred:
      if maxent:
        tp += 1
      else:
        fp += 1
    else:
      if maxent:
        fn += 1
      else:
        tn += 1
  
  return tp, tn, fp, fn


#def mean_avg_ppv(data, white_srv, t_step=0.02, r_start=0.5, r_stop=12.0, r_num=46):
#def mean_avg_ppv(data, white_srv, t_step=0.02, r_start=0.5, r_stop=5.0, r_num=46):
def mean_avg_ppv(data, white_srv, t_step=0.01, r_start=0.5, r_stop=7.0, r_num=66):
  mAP = 0
  
  for radius in np.linspace(r_start, r_stop, r_num):
    ap = 0
    tpr_old = 0
    x = [0]
    y = [1]
    
    for thresh in np.linspace(1-t_step, 0, int(round(1/t_step))):
      tp, tn, fp, fn = conf_table(data, thresh, radius, white_srv)
      
      if tp+fp <= 0:
        print('avoid division by 0 for t=%f r=%f' % (thresh, radius))
        continue
      
      ppv = tp/(tp+fp)
      tpr = tp/(tp+fn)
      ap += (tpr - tpr_old) * ppv
      tpr_old = tpr
      x.append(tpr)
      y.append(ppv)
    
    plt.plot(x, y)
    plt.savefig('pr_%f.pdf' % radius, bbox_inches='tight')
    plt.clf()
    print('radius=%f AP:' % radius, ap)
    mAP += ap
  
  return mAP / r_num


def myload(file):
  a = []

  with open(file, 'rb') as f:
    while True:
      try:
        a.append(np.load(f))
      except OSError:
        break
  
  return np.vstack(a)


def main(args):
  t_range = np.arange(0., 1., 0.1) #np.arange(0.4, 1.0, 0.05)
  r_range = np.arange(0.5, 10., 0.5) #np.arange(0.5, 5.0, 0.25)
  
  if args.mean_avg_ppv:
    a = myload(args.in_file)
    white_srv = make_white_srv(args.fold)
    print('mAP: ', mean_avg_ppv(a, white_srv))
    return
  
  if not args.no_plot:
    #t_range = np.arange(0.05, 1.0, 0.05)
    #r_range = np.arange(1, 10, 0.25)
    
    tnr_map, tpr_map, sup_map, f_map, bac_map, pr_map = np.load(args.plot_file)
    
    
    plt.rcParams.update({'font.size': 18}) # 4
    titles = ['Speedup', 'TNR', 'Rediscovery Ratio', 'TPR', 'Balanced Accuracy', 'Precision']
    subs = range(321, 326)
    imgs = [sup_map, tnr_map, f_map, tpr_map, bac_map, pr_map]
    vmaxs = [40, 1, 1, 1, 1, 1]
    vmins = [0, 0.6, 0, 0, 0, 0]
    
    for sub, tit, img, vmax, vmin in zip(subs, titles, imgs, vmaxs, vmins):
      if all(f < 2 for f in args.fold) and (tit == 'TPR' or tit == 'Rediscovery Ratio'):
        continue
      
      plt.subplot(sub)
      plt.title(tit)
      #im = plt.imshow(img, cmap=plt.cm.Blues, interpolation='nearest', vmin=vmin, vmax=vmax)
      im = plt.imshow(img, cmap=plt.cm.rainbow, interpolation='nearest', vmin=vmin, vmax=vmax)
      print('doing ticks')
      plt.xticks(list(range(len(r_range)))[::4], np.round(r_range[::4], 4))
      plt.yticks(list(range(len(t_range)))[::2], np.round(t_range[::2], 2))
      plt.xlabel('SRV Radius')
      plt.ylabel('Sigmoid Threshold')
      
      if not args.no_colorbar:
        plt.colorbar(im)
    
    plt.savefig(args.out_file)
    
    fname_prefix = ['speedup_', 'tnr_', 'rediscr_', 'tpr_', 'bacc_', 'pr_']
    
    for tit, img, vmax, vmin, fpfix in zip(titles, imgs, vmaxs, vmins, fname_prefix):
      if all(f < 2 for f in args.fold) and (tit == 'TPR' or tit == 'Rediscovery Ratio'):
        continue
      
      plt.clf()
      plt.title(tit)
      #im = plt.imshow(img, cmap=plt.cm.Blues, interpolation='nearest', vmin=vmin, vmax=vmax)
      im = plt.imshow(img, cmap=plt.cm.rainbow, interpolation='nearest', vmin=vmin, vmax=vmax)
      plt.xticks(list(range(len(r_range)))[::4], np.round(r_range[::4], 4))
      plt.yticks(list(range(len(t_range)))[::2], np.round(t_range[::2], 2))
      plt.xlabel('SRV Radius')
      plt.ylabel('Sigmoid Threshold')
      
      if not args.no_colorbar:
        plt.colorbar(im)
      
      plt.savefig(fpfix + args.out_file, bbox_inches='tight')
  else:
    print('no-plot')
    a = myload(args.in_file)
    print('data loaded')
    
    tnr_map, tpr_map, sup_map, f_map, bac_map, pr_map = [np.zeros((len(t_range), len(r_range))) for i in range(6)]
    
    for i, t in enumerate(t_range):
      for j, r in enumerate(r_range):
        tp, tn, fp, fn, f_map[i,j] = validate(args, a, t, r)
        # TODO: write these figures to disk and make plots in separate script
        print('t', t, 'r', r, 'tp', tp, 'tn', tn, 'fp', fp, 'fn', fn)
        tnr_map[i,j] = tn/(tn+fp)
        tpr_map[i,j] = 1 if all(f < 2 for f in args.fold) else tp/(tp+fn)
        sup_map[i,j] = tpr_map[i,j]/(1-tnr_map[i,j])
        bac_map[i,j] = 1/2 * (tnr_map[i,j] + tpr_map[i,j])
        pr_map[i,j] = tp / (tp + fp)
    
    np.save(args.plot_file, [tnr_map, tpr_map, sup_map, f_map, bac_map, pr_map])
    
    # TODO np save
    #tp, tn, fp, fn, s = validate(args, a, args.threshold, args.radius)
    #print(sorted(s))
    #print('tp:', tp, 'fp:', fp, 'fn:', fn, 'tn:', tn)
    #tnr = tn/(tn+fp)
    #tpr = tp/(tp+fn)
    #print('tnr:', tnr, 'tpr:', tpr, 'speedup:', tpr/(1-tnr))


def create_argparser():
  parser = argparse.ArgumentParser(description='melvin args')
  parser.add_argument('--in-file', type=str, default=None, help='npy file to analyze')
  parser.add_argument('--out-file', type=str, default=None, help='pdf file to plot to')
  parser.add_argument('--threshold', type=float, default=0.5, help='maxent threshold')
  parser.add_argument('--radius', type=float, default=2.0, help='SRV radius')
  parser.add_argument('--fold', type=int, nargs='+', default=8, help='which fold to check')
  parser.add_argument('--plot-file', type=str, default=None, help='npy file containing plot data')
  parser.add_argument('--no-plot', action='store_true')
  parser.add_argument('--no-colorbar', action='store_true')
  parser.add_argument('--mean-avg-ppv', action='store_true')
  return parser


if __name__ == '__main__':
  main(create_argparser().parse_args())
