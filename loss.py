import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.poisson import Poisson
from torch.distributions.binomial import Binomial


def binomial_nll_loss(input, target, n, reduction='mean'):
  """input must be logits"""
  loss = target * F.softplus(-input) + (n-target) * F.softplus(input)
  
  if reduction == 'none':
      ret = loss
  elif reduction == 'mean':
      ret = torch.mean(loss)
  else:
      ret = torch.sum(loss)
  return ret


def __poisson_nll_loss(input, target, reduction='mean'):
  # FIXME: this implementation has a bug; reuduction='none' does reduction='sum'
  return F.poisson_nll_loss(input, target, reduction=reduction)


def poisson_nll_loss(input, target, reduction='mean'):
  loss = torch.exp(input) - target * input
  
  if reduction == 'none':
      ret = loss
  elif reduction == 'mean':
      ret = torch.mean(loss)
  else:
      ret = torch.sum(loss)
  return ret


def poisson_binomial_nll_loss(input, target, reduction='mean'):
  n, m, k = target.t()
  l, p, q = input.t()
  
  loss  = poisson_nll_loss(l, n, reduction='none')
  loss += binomial_nll_loss(p, m, n=n, reduction='none')
  loss += binomial_nll_loss(q, k, n=m, reduction='none')
  
  if reduction == 'none':
      ret = loss
  elif reduction == 'mean':
      ret = torch.mean(loss)
  else:
      ret = torch.sum(loss)
  return ret


def mse_loss(input, target, reduction='mean'):
  return F.mse_loss(input.exp(), target, reduction=reduction)


def srv_criterion(input, target, loss='poisson_binomial', reduction='mean'):
  # only induce loss for non-zero SRVs
  mask = (target.sum(dim=1) > 0)
  loss_fn = {'mse': mse_loss,
             'poisson': poisson_nll_loss,
             'poisson_binomial': poisson_binomial_nll_loss}
  l = loss_fn[loss](input[mask], target[mask]-1, reduction=reduction)
  return l, loss_fn[loss].__name__, mask


def srv_soft_prediction(ctx, input):
  if ctx.args.srv_loss == 'mse':
    ret = input.exp()
  elif ctx.args.srv_loss == 'poisson':
    # FIXME: poisson has mode floor(lambda) but round seems to work better
    ret = input.exp()
  elif ctx.args.srv_loss == 'poisson_binomial':
    l, p, q = input.t()
    l = l.exp()
    p = p.sigmoid()
    q = q.sigmoid()
    ret = torch.stack([l, l*p, l*p*q]).t()
  else:
    raise ValueError('Unknown SRV loss %s' % ctx.args.srv_loss)
  
  return ret


def srv_prediction(ctx, input):
  return srv_soft_prediction(ctx, input).round()


def maxent_criterion(input, target, reduction='mean'):
  return F.binary_cross_entropy_with_logits(input, target, reduction=reduction), 'cross_entropy_loss'


def log_joint(l, p, q, n, m, k):
  return Poisson(l).log_prob(n) + Binomial(n, p).log_prob(m) + Binomial(m, q).log_prob(k)


def find_mode(input):
  """This is a bruteforce method to find the mode of the joint. It serves as preliminary 
     implementation until the mode is found analytically and as a sanity check for candidates."""
  # TODO: test
  l, p, q = input.t()
  l, p, q = l.exp(), p.sigmoid(), q.sigmoid()
  prob_max = torch.tensor([-float('Inf')]).to(input.device).repeat(input.shape[0])
  mode = torch.zeros_like(input)
  
  for n in range(11):
    for m in range(n+1):
      for k in range(m+1):
        n_, m_, k_ = [torch.tensor([t], dtype=torch.float).to(input.device) for t in [n, m, k]]
        prob = log_joint(l, p, q, n_, m_, k_)
        mask = (prob_max <= prob)
        
        if mask.sum().item() > 0:
          prob_max[mask] = prob[mask]
          mode[mask] = torch.tensor([n, m, k], dtype=torch.float).to(input.device).repeat(mask.sum(), 1)
  
  return mode


def num_correct_srv(output, target, mask, ctx, radius):
  if radius == 0.:
    pred = srv_prediction(ctx, output)
    ret = (target[mask] - 1 == pred[mask]).all(dim=1).long().sum().item()
  else:
    pred = srv_soft_prediction(ctx, output)
    dist = (target[mask] - 1 - pred[mask]).norm(dim=1)
    ret = (dist <= radius).long().sum().item()
  
  return ret

