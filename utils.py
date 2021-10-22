import os
import git
import yaml
import torch
import time as __time__


class Object(object):
  """dummy class that has a __dict__ to collect objects"""
  pass


class Metrics(object):
  def reset():
    # TODO: implement
    pass
  
  def add(output, target, srv):
    output, target, srv = [x.cpu().data.numpy() for x in [output, target, srv]]
    maxent_pred = (output[:,:1] >= 0)
    srv_pred = output[:,1:].round()


class Result(object):
  def __init__(self, n):
    self.df = pd.DataFrame(index=range(n), columns=list('nmktp'))
    self.i = 0
  
  def add(self, srv, target, prediction):
    for s, t, p in zip(srv, target, prediction):
      self.df.loc[self.i] = list(s.cpu().numpy()) + list(t.cpu().numpy()) + list(p.cpu().numpy())
      self.i += 1
  
  def save(self, fname):
    self.df.sort_values(list('nmk')).to_csv(fname)


def dump_setup(ctx):
  """Make experiments reproducible by dumping commit id, git diff, and args to file.
     An experiment can be reproduced by checking out the commit specified in setup.txt, applying 
     setup.patch, and specifying the (non-default) command line arguments in setup.txt."""
  repo = git.Repo(os.path.dirname(os.path.realpath(__file__)))
  ctx.args.commit = repo.head.commit.hexsha
  
  with open(os.path.join(ctx.logdir, ctx.args.model_name + '.txt'), 'w') as f:
    f.write(yaml.dump(ctx.args.__dict__))
  
  with open(os.path.join(ctx.logdir, ctx.args.model_name + '.patch'), 'w') as f:
    f.write(repo.git.diff(repo.head.commit.tree))


def load_setup(file):
  with open(file, 'r') as f:
    if yaml.__version__ >= '5.1':
      setup_dict = yaml.load(f, Loader=yaml.FullLoader)
    else:
      setup_dict = yaml.load(f)
  
  setup = Object()
  
  for k, v in setup_dict.items():
    setattr(setup, k, v)
  
  return setup


def log_scalars(writer, names, values, step, prefix=None):
  for n, v in zip(names, values):
    name = n if prefix is None else '/'.join(prefix, n)
    value = v.item() if type(v) is torch.Tensor else v
    writer.add_scalar(name, value, step)


def time(f, *args):
  start = __time__.perf_counter()
  ret = f(*args)
  end = __time__.perf_counter()
  return end-start, ret




