import pandas as pd
import numpy as np
import sqlite3
import gzip
import shutil
import io
from os.path import isfile, join, basename
from glob import glob
import matplotlib.pyplot as plt
from math import ceil, sqrt
import numpy as np
from random import random, shuffle
import itertools

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from torch.distributions.multinomial import Multinomial
from torch.utils.data.sampler import SequentialSampler, WeightedRandomSampler

"""
make new ARRAY data type in sqlite3 for loading/storing numpy arrays
https://stackoverflow.com/a/18622264
"""

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return out.read()

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("ARRAY", convert_array)

# mathematically possible SRVs (except trivial n,m,1)
mposs_srv = set()

for n in range(2, 12):
  for m in range(ceil(sqrt(n)), n+1):
    for k in range(2, m+1):
      mposs_srv.add((n, m, k))

# observed SRVs (green), generated by 'SELECT DISTINCT n, m, k FROM data WHERE maxent=1'
green_srv = {
  (2,2,2), (3,2,2), (3,3,2), (3,3,3), (4,2,2), (4,3,2), (4,3,3), (4,4,2), (4,4,3), (5,3,2), 
  (5,3,3), (5,4,2), (5,4,3), (5,5,2), (5,4,4), (5,5,3), (6,3,3), (6,4,3), (6,4,4), (6,5,2), 
  (6,5,3), (6,6,2), (6,4,2), (6,5,4), (6,6,3), (7,6,3), (7,4,3), (7,4,4), (7,5,2), (7,5,3), 
  (7,5,4), (7,6,2), (7,7,2), (7,6,4), (8,5,4), (8,5,5), (8,6,3), (8,6,4), (8,7,2), (8,7,3), 
  (8,8,2), (8,6,2), (9,6,4), (9,6,5), (9,7,3), (9,9,2), (9,6,3), (9,7,4), (9,8,2), (10,10,2), 
  (10,6,6), (10,7,5)}

# unobserved SRVs (white)
white_srv = mposs_srv - green_srv

commap = [
  'BS[XXX,a,b]', 'LI[XXX,a,b]', 'BS[XXX,a,c]', 'LI[XXX,a,c]',
  'BS[XXX,a,d]', 'LI[XXX,a,d]', 'BS[XXX,a,e]', 'LI[XXX,a,e]',
  'BS[XXX,a,f]', 'LI[XXX,a,f]', 'BS[XXX,b,c]', 'LI[XXX,b,c]',
  'BS[XXX,b,d]', 'LI[XXX,b,d]', 'BS[XXX,b,e]', 'LI[XXX,b,e]',
  'BS[XXX,b,f]', 'LI[XXX,b,f]', 'BS[XXX,c,d]', 'LI[XXX,c,d]',
  'BS[XXX,c,e]', 'LI[XXX,c,e]', 'BS[XXX,c,f]', 'LI[XXX,c,f]',
  'BS[XXX,d,e]', 'LI[XXX,d,e]', 'BS[XXX,d,f]', 'LI[XXX,d,f]',
  'BS[XXX,e,f]', 'LI[XXX,e,f]', 'Reflection[XXX,a]', 'DP[XXX,a]',
  'OAMHolo[XXX,a,-5]', 'OAMHolo[XXX,a,-4]', 'OAMHolo[XXX,a,-3]',
  'OAMHolo[XXX,a,-2]', 'OAMHolo[XXX,a,-1]', 'OAMHolo[XXX,a,1]',
  'OAMHolo[XXX,a,2]', 'OAMHolo[XXX,a,3]', 'OAMHolo[XXX,a,4]',
  'OAMHolo[XXX,a,5]', 'Reflection[XXX,b]', 'DP[XXX,b]',
  'OAMHolo[XXX,b,-5]', 'OAMHolo[XXX,b,-4]', 'OAMHolo[XXX,b,-3]',
  'OAMHolo[XXX,b,-2]', 'OAMHolo[XXX,b,-1]', 'OAMHolo[XXX,b,1]',
  'OAMHolo[XXX,b,2]', 'OAMHolo[XXX,b,3]', 'OAMHolo[XXX,b,4]',
  'OAMHolo[XXX,b,5]', 'Reflection[XXX,c]', 'DP[XXX,c]',
  'OAMHolo[XXX,c,-5]', 'OAMHolo[XXX,c,-4]', 'OAMHolo[XXX,c,-3]',
  'OAMHolo[XXX,c,-2]', 'OAMHolo[XXX,c,-1]', 'OAMHolo[XXX,c,1]',
  'OAMHolo[XXX,c,2]', 'OAMHolo[XXX,c,3]', 'OAMHolo[XXX,c,4]',
  'OAMHolo[XXX,c,5]', 'Reflection[XXX,d]', 'DP[XXX,d]',
  'OAMHolo[XXX,d,-5]', 'OAMHolo[XXX,d,-4]', 'OAMHolo[XXX,d,-3]',
  'OAMHolo[XXX,d,-2]', 'OAMHolo[XXX,d,-1]', 'OAMHolo[XXX,d,1]',
  'OAMHolo[XXX,d,2]', 'OAMHolo[XXX,d,3]', 'OAMHolo[XXX,d,4]',
  'OAMHolo[XXX,d,5]', 'Reflection[XXX,e]', 'DP[XXX,e]',
  'OAMHolo[XXX,e,-5]', 'OAMHolo[XXX,e,-4]', 'OAMHolo[XXX,e,-3]',
  'OAMHolo[XXX,e,-2]', 'OAMHolo[XXX,e,-1]', 'OAMHolo[XXX,e,1]',
  'OAMHolo[XXX,e,2]', 'OAMHolo[XXX,e,3]', 'OAMHolo[XXX,e,4]',
  'OAMHolo[XXX,e,5]', 'Reflection[XXX,f]', 'DP[XXX,f]',
  'OAMHolo[XXX,f,-5]', 'OAMHolo[XXX,f,-4]', 'OAMHolo[XXX,f,-3]',
  'OAMHolo[XXX,f,-2]', 'OAMHolo[XXX,f,-1]', 'OAMHolo[XXX,f,1]',
  'OAMHolo[XXX,f,2]', 'OAMHolo[XXX,f,3]', 'OAMHolo[XXX,f,4]',
  'OAMHolo[XXX,f,5]', 'EOS']

commap_sympy = [
  'BS(XXX,a,b)', 'LI(XXX,a,b)', 'BS(XXX,a,c)', 'LI(XXX,a,c)',
  'BS(XXX,a,d)', 'LI(XXX,a,d)', 'BS(XXX,a,e)', 'LI(XXX,a,e)',
  'BS(XXX,a,f)', 'LI(XXX,a,f)', 'BS(XXX,b,c)', 'LI(XXX,b,c)',
  'BS(XXX,b,d)', 'LI(XXX,b,d)', 'BS(XXX,b,e)', 'LI(XXX,b,e)',
  'BS(XXX,b,f)', 'LI(XXX,b,f)', 'BS(XXX,c,d)', 'LI(XXX,c,d)',
  'BS(XXX,c,e)', 'LI(XXX,c,e)', 'BS(XXX,c,f)', 'LI(XXX,c,f)',
  'BS(XXX,d,e)', 'LI(XXX,d,e)', 'BS(XXX,d,f)', 'LI(XXX,d,f)',
  'BS(XXX,e,f)', 'LI(XXX,e,f)', 'Reflection(XXX,a)', 'DP(XXX,a)',
  'OAMHolo(XXX,a,-5)', 'OAMHolo(XXX,a,-4)', 'OAMHolo(XXX,a,-3)',
  'OAMHolo(XXX,a,-2)', 'OAMHolo(XXX,a,-1)', 'OAMHolo(XXX,a,1)',
  'OAMHolo(XXX,a,2)', 'OAMHolo(XXX,a,3)', 'OAMHolo(XXX,a,4)',
  'OAMHolo(XXX,a,5)', 'Reflection(XXX,b)', 'DP(XXX,b)',
  'OAMHolo(XXX,b,-5)', 'OAMHolo(XXX,b,-4)', 'OAMHolo(XXX,b,-3)',
  'OAMHolo(XXX,b,-2)', 'OAMHolo(XXX,b,-1)', 'OAMHolo(XXX,b,1)',
  'OAMHolo(XXX,b,2)', 'OAMHolo(XXX,b,3)', 'OAMHolo(XXX,b,4)',
  'OAMHolo(XXX,b,5)', 'Reflection(XXX,c)', 'DP(XXX,c)',
  'OAMHolo(XXX,c,-5)', 'OAMHolo(XXX,c,-4)', 'OAMHolo(XXX,c,-3)',
  'OAMHolo(XXX,c,-2)', 'OAMHolo(XXX,c,-1)', 'OAMHolo(XXX,c,1)',
  'OAMHolo(XXX,c,2)', 'OAMHolo(XXX,c,3)', 'OAMHolo(XXX,c,4)',
  'OAMHolo(XXX,c,5)', 'Reflection(XXX,d)', 'DP(XXX,d)',
  'OAMHolo(XXX,d,-5)', 'OAMHolo(XXX,d,-4)', 'OAMHolo(XXX,d,-3)',
  'OAMHolo(XXX,d,-2)', 'OAMHolo(XXX,d,-1)', 'OAMHolo(XXX,d,1)',
  'OAMHolo(XXX,d,2)', 'OAMHolo(XXX,d,3)', 'OAMHolo(XXX,d,4)',
  'OAMHolo(XXX,d,5)', 'Reflection(XXX,e)', 'DP(XXX,e)',
  'OAMHolo(XXX,e,-5)', 'OAMHolo(XXX,e,-4)', 'OAMHolo(XXX,e,-3)',
  'OAMHolo(XXX,e,-2)', 'OAMHolo(XXX,e,-1)', 'OAMHolo(XXX,e,1)',
  'OAMHolo(XXX,e,2)', 'OAMHolo(XXX,e,3)', 'OAMHolo(XXX,e,4)',
  'OAMHolo(XXX,e,5)', 'Reflection(XXX,f)', 'DP(XXX,f)',
  'OAMHolo(XXX,f,-5)', 'OAMHolo(XXX,f,-4)', 'OAMHolo(XXX,f,-3)',
  'OAMHolo(XXX,f,-2)', 'OAMHolo(XXX,f,-1)', 'OAMHolo(XXX,f,1)',
  'OAMHolo(XXX,f,2)', 'OAMHolo(XXX,f,3)', 'OAMHolo(XXX,f,4)',
  'OAMHolo(XXX,f,5)']

ff1max = 25
ff1map = ['FF1[%d]' % i for i in range(-ff1max, ff1max+1)]

filemap = glob(join('/home/tomte/data/melvin/txt_files', '*.txt'))
filemap.sort()
lblmap = [0 if basename(s).lower().startswith('wrong') else 1 for s in filemap]
lsetmap = [1 if 'large' in basename(s).lower() else 0 for s in filemap]

def encode(line, noff1=True):
  srv, com, ff1 = [s.strip()[1:-1] for s in line.split('|')[:3]]
  srv = [abs(int(i)) for i in srv.split(', ')]
  com = com[1:-1].split('", "')
  com = np.array([commap.index(c) for c in com], dtype=np.uint8)
  
  if not noff1:
    ff1 = ff1.split(', ')
    
    if ff1[0] in ff1map:
      ff1 = [ff1map.index(c) for c in ff1]
      ff1 = np.eye(len(ff1map))[ff1].sum(0).astype(np.uint8)
    else:
      ff1 = np.zeros((len(ff1map),), dtype=np.uint8)
    
    return srv, com, ff1
  else:
    return srv, com, np.zeros((len(ff1map),), dtype=np.uint8)


def decode(srv, com, ff1, stats=False):
  if stats:
    srv = [str(float(s)) for s in srv]
  else:
    srv = [str(int(s)) for s in srv]
  
  
  comlist = [commap[c] for c in com]
  ff1list = []
  
  for i, f in enumerate(ff1):
    if f==1:
      ff1list.append(ff1map[i])
  
  return '{' + ', '.join(tuple(srv)) + '} | {"' + '", "'.join(comlist) + '"} | {' + \
         ', '.join(ff1list) + '}'


def create_sqlite_database_from_textfiles(dbfile, txtdir):
  if isfile(dbfile):
    print('cannot create database, file', dbfile, 'already exists')
    return
  
  print('creating database', dbfile)
  con = sqlite3.connect(dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  c.execute('CREATE TABLE data (n INTEGER, m INTEGER, k INTEGER, \
             file INTEGER, maxent INTEGER, com ARRAY, ff1 ARRAY)')

  for i, (fin, maxent) in enumerate(zip(filemap, lblmap)):
    print('processing %s' % fin)
    with open(fin, 'r') as f:
      records = []
      for line in f:
        if len(line) > 1:
          srv, com, ff1 = encode(line)
          records.append((srv[0], srv[1], srv[2], i, maxent, com, ff1))
      
      c.executemany('INSERT INTO data (n, m, k, file, maxent, com, ff1) \
                     VALUES (?, ?, ?, ?, ?, ?, ?)', records)
  c.execute('CREATE INDEX idx_n_maxent ON data (n, maxent)')
  c.execute('CREATE INDEX idx_file ON data (file)')
  con.commit()
  con.close()


def delete_duplicates(dbfile):
  print('deleting duplicates from', dbfile)
  con = sqlite3.connect(dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  s = 'DELETE FROM data WHERE rowid NOT IN (SELECT MIN(rowid) FROM data GROUP BY n, m, k, com, ff1)'
  r = c.execute(s)
  print('deleted', r.rowcount, 'rows')
  
  print('regenerating column rowid')
  n = c.execute('SELECT COUNT(*) FROM data').fetchone()[0]
  
  from tqdm import tqdm
  
  for i in tqdm(range(1, n+1)):
    c.execute('UPDATE data SET rowid=' + str(i) + ' WHERE rowid=\
              (SELECT MIN(rowid) FROM data WHERE rowid >= ' + str(i) + ')')
  
  c.execute('VACUUM')
  con.commit()
  con.close()


def sanity_check(dbfile):
  print('sanity check on', dbfile, '...', end=' ')
  checks = []
  con = sqlite3.connect(dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  
  s = 'SELECT COUNT(*) FROM data WHERE rowid NOT IN (SELECT MIN(rowid) FROM data \
       GROUP BY n, m, k, maxent, com, ff1)'
  checks.append(c.execute(s).fetchone()[0] == 0)
  
  s = 'SELECT COUNT(*) FROM data WHERE rowid NOT IN (SELECT MIN(rowid) FROM data \
       GROUP BY com, ff1)'
  checks.append(c.execute(s).fetchone()[0] == 0)
  
  s = 'SELECT MAX(rowid), COUNT(rowid) FROM data'
  m, n = c.execute(s).fetchone()
  checks.append(m==n)
  
  if all(checks):
    print('SUCCESS')
  else:
    print('FAILED', checks)
    # TODO: print report
  
  con.close()
  return n


def create_sqlite_database_subset(src_dbfile, dst_dbfile, where_clause, 
  values = 'n, m, k, file, maxent, com, ff1', select = 'n, m, k, file, maxent, com, ff1'):
  if isfile(dst_dbfile):
    print('cannot create database, file', dst_dbfile, 'already exists')
    return
  
  print('creating database', dst_dbfile)
  con = sqlite3.connect(dst_dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  c.execute('ATTACH DATABASE \'' + src_dbfile + '\' AS src')
  c.execute('CREATE TABLE main.data (n INTEGER, m INTEGER, k INTEGER, \
             file INTEGER, maxent INTEGER, com ARRAY, ff1 ARRAY)')
  c.execute('INSERT INTO main.data (' + values + ') \
             SELECT ' + select + ' FROM src.data WHERE ' + where_clause)
  c.execute('CREATE INDEX idx_n_maxent ON data (n, maxent)')
  c.execute('CREATE INDEX idx_file ON data (file)')
  con.commit()
  con.close()


def create_sqlite_database_noff1():
  con = sqlite3.connect('melvinFull.db', detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  c.execute('ATTACH DATABASE \'melvinNoFF1.db\' as dst')
  c.execute('CREATE TABLE dst.data (n INTEGER, m INTEGER, k INTEGER, \
             maxent INTEGER, com ARRAY, file INTEGER)')
  c.execute('INSERT INTO dst.data (n, m, k, maxent, com, file) \
             SELECT n, m, k, MAX(maxent), com, file FROM \
             (SELECT max(16*16*n + 16*m + k), n, m, k, maxent, com, file \
             FROM main.data WHERE maxent == 1 GROUP BY com \
             UNION \
             SELECT max(16*16*n + 16*m + k), n, m, k, maxent, com, file \
             FROM main.data WHERE maxent == 0 GROUP BY com) \
             GROUP BY com')
  con.commit()
  con.close()


def create_g6_sqlite_databases(dbFull):
  dbG6Train = 'melvinG6Train.db'
  where = 'n != 6'
  create_sqlite_database_subset(dbFull, dbG6Train, where)
  sanity_check(dbG6Train)
  
  dbG6Test = 'melvinG6Test.db'
  where = 'n = 6'
  create_sqlite_database_subset(dbFull, dbG6Test, where)
  sanity_check(dbG6Test)


def create_all_sqlite_databases():
  dbFull = 'melvinFull.db'
  create_sqlite_database_from_textfiles(dbFull, '/home/tomte/data/melvin/txt_files')
  delete_duplicates(dbFull)
  sanity_check(dbFull)
  
  #create_g6_sqlite_databases(dbFull)


def create_test_split_databases(full_db):
  # make random train-valid-test split by approximately 80/5/15 percent
  con = sqlite3.connect(full_db, detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  s = 'SELECT COUNT(rowid) FROM data'
  num_rows, = c.execute(s).fetchone()
  con.close()
  
  np.random.seed(int('0xdeadbeef', base=16))
  idx = np.random.permutation(num_rows)+1
  test_offset = ceil(num_rows*0.05)
  train_offset = test_offset + ceil(num_rows*0.15)
  valid_idx = idx[:test_offset]
  test_idx = idx[test_offset:train_offset]
  train_idx = idx[train_offset:]
  
  # full contains all data, split into train, valid, test
  train_db = 'melvinTrain.db'
  valid_db = 'melvinValid.db'
  test_db = 'melvinTest.db'
  create_sqlite_database_subset(full_db, train_db, 'rowid in ' + str(tuple(train_idx)))
  create_sqlite_database_subset(full_db, valid_db, 'rowid in ' + str(tuple(valid_idx)))
  create_sqlite_database_subset(full_db, test_db, 'rowid in ' + str(tuple(test_idx)))


def create_positives_database():
  create_sqlite_database_subset('melvinFull.db', 'melvinPos.db', 'maxent=1')


def stats(dbfile):
  con = sqlite3.connect(dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  stats = [('maxent', 'n', 'm', 'k', 'count')]
  
  c.execute('SELECT maxent, \'*\', \'*\', \'*\', COUNT(*) FROM data GROUP BY maxent')
  stats += list(c).copy()
  c.execute('SELECT maxent, n, \'*\', \'*\', COUNT(*) FROM data GROUP BY maxent, n')
  stats += list(c).copy()
  c.execute('SELECT maxent, n, m, k, COUNT(*) FROM data GROUP BY maxent, n, m, k')
  stats += list(c).copy()
  
  return stats


def visualize_stats(dbfile):
  plt.rcParams.update({'font.size': 14})
  
  def count(df, n, m, k):
    subneg = df[(df.n==n) & (df.m==m) & (df.k==k) & (df.maxent==0)]
    subpos = df[(df.n==n) & (df.m==m) & (df.k==k) & (df.maxent==1)]
    return (subneg.iloc[0]['count'] if len(subneg)==1 else 0), \
      (subpos.iloc[0]['count'] if len(subpos)==1 else 0)
  
  data = stats(dbfile)
  df = pd.DataFrame(data[1:], columns=list(data[0]))
  srv = pd.unique(df[list('nmk')].values)
  n01 = srv[0]
  byn = srv[1:14]
  srv = srv[14:]
  
  for i, (n, m, k) in enumerate(srv):
    if i % 40 == 0:
      print('\nn|m|k|neg|pos')
      print('---:| ---:| ---:| ---:| ---:')
    print(' | '.join([str(x) for x in [n, m, k, *count(df, n, m, k)]]))
  
  plt.pie(count(df, *n01), labels=('negatives', 'positives'), autopct='%1.1f%%', startangle=90)
  plt.axis('equal')
  plt.savefig('doc/pie.pdf', bbox_inches='tight')
  plt.clf()
  
  counts = []
  
  for i, (n, m, k) in enumerate(byn):
    counts.append(count(df, n, m, k))
  
  ind = np.arange(len(byn))
  neg = [c[0] for c in counts]
  pos = [c[1] for c in counts]
  p_neg = plt.bar(ind*2-.1, neg)
  p_neg = plt.bar(ind*2-.9, pos)
  
  plt.xlabel('leading Schmidt rank')
  plt.ylabel('number of samples')
  plt.xticks(ind*2-.5, [s[0] for s in byn])
  plt.yscale('log')
  plt.legend(('negative', 'positive'))
  plt.axes().set_aspect(16/9)
  plt.savefig('doc/counts_by_n.pdf', bbox_inches='tight')


class MelvinDataset(Dataset):
  # FIXME: this version supports only num_workers=1 because cursors will be shared among workers
  def __init__(self, dbfile, table='data', noff1=True):
    # FIXME: self.query is very unsafe wrt SQL injections
    self.noff1 = noff1
    self.connection = sqlite3.connect(dbfile, detect_types=sqlite3.PARSE_DECLTYPES)
    self.cursor = self.connection.cursor()
    self.query = 'SELECT rowid, n, m, k, file, maxent, com' + \
      (' ' if noff1 else ', ff1') + 'FROM ' + table + ' WHERE rowid = '
    self.len = self.cursor.execute('SELECT MAX(rowid) FROM data').fetchone()[0]
    n_pos = self.cursor.execute('SELECT COUNT(rowid) FROM data WHERE maxent = 1').fetchone()[0]
    self.positive_rate = n_pos / self.len
  
  def __len__(self):
    return self.len
  
  def __getitem__(self, i):
    if isinstance(i, torch.Tensor):
      i = i.item()
    i = int(i)
    sample = self.cursor.execute(self.query + str(i % self.len + 1)).fetchone()
    com = torch.tensor(sample[6], dtype=torch.long)
    maxent = torch.tensor([sample[5]], dtype=torch.float)
    srv = torch.tensor(sample[1:4], dtype=torch.long)
    file = torch.tensor(sample[4], dtype=torch.long)
    
    if self.noff1:
      ret = (com, maxent, srv, file)
    else:
      ff1 = torch.tensor(sample[7], dtype=torch.float)
      ret = (com, ff1, maxent, srv, file)
    
    return ret
  
  def __del__(self):
    self.connection.close()
  
  def getWeightedRandomSampler(self):
    labels = self.cursor.execute('SELECT maxent FROM data ORDER BY rowid').fetchall()
    labels = np.array(labels, dtype=np.float).reshape(-1)
    weights = -(labels.copy() - 1)
    labels *= 1/self.positive_rate
    weights *= 1/(1-self.positive_rate)
    return WeightedRandomSampler(list(weights+labels), int(self.positive_rate * self.len * 2))


def collate(batch):
  """this implementation of collate sorts com by descending lengths and pads shorter
  sequences with zeros"""
  batch.sort(key=lambda x: x[0].shape[0], reverse=True)
  com, ff1, maxent, srv, file = zip(*batch)
  return pack_sequence(com), torch.stack(ff1, 0), torch.stack(maxent, 0), \
    torch.stack(srv, 0), torch.stack(file, 0)


def collate_noff1(batch):
  """this implementation of collate sorts com by descending lengths and pads shorter
  sequences with zeros"""
  batch.sort(key=lambda x: x[0].shape[0], reverse=True)
  com, maxent, srv, file = zip(*batch)
  return pack_sequence(com), torch.stack(maxent, 0), \
    torch.stack(srv, 0), torch.stack(file, 0)


def collate_noff1_unpacked(batch):
  """this implementation of collate sorts com by descending lengths and pads shorter
  sequences with zeros"""
  batch.sort(key=lambda x: x[0].shape[0], reverse=True)
  com, maxent, srv, file = zip(*batch)
  return com, torch.stack(maxent, 0), \
    torch.stack(srv, 0), torch.stack(file, 0)


def get_melvin_data_loader(batch_size, split='train'):
  db_file = {'train': 'melvinTrain.db',
             'valid': 'melvinValid.db',
             'test': 'melvinTest.db'}[split]
  return DataLoader(MelvinDataset(db_file), 
                    batch_size=batch_size, 
                    collate_fn=collate, 
                    shuffle=True, 
                    drop_last=True)


def create_melvin_data_loader(batch_size, g6=True):
  if g6: # train, val from G6 train; test is G6 test
    data = MelvinDataset('melvinG6Train.db')
    test_data = MelvinDataset('melvinG6Test.db')
    len_train = int(len(data) * 0.85)
    split = [len_train, len(data) - len_train]
    train_data, valid_data = random_split(data, split)
  else: # make a random split
    data = MelvinDataset('melvinFull.db')
    len_train = int(len(data) * 0.7)
    len_valid = int(len(data) * 0.1)
    split = [len_train, len_valid, len(data) - len_train - len_valid]
    train_data, valid_data, test_data = random_split(data, split)
  
  train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate, 
                            shuffle=True, drop_last=True)
  valid_loader = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate, 
                            shuffle=True, drop_last=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate, 
                           shuffle=True, drop_last=True)
  return train_loader, valid_loader, test_loader


class TxtDataset(object):
  def __init__(self, file, maxent=None, noff1=True):
    self.file_is_io = (type(file) is io.TextIOWrapper)
    self.f = (file if self.file_is_io else open(file))
    self.maxent = maxent
    self.noff1 = noff1
  
  def __del__(self):
    if not self.file_is_io:
      self.f.close()
  
  def __iter__(self):
    return self
  
  def __next__(self):
    line = '\n'
    
    while len(line) == 1:
      line = self.f.readline()
    
    if len(line) == 0:
      raise StopIteration
    
    if self.maxent is None:
      srv = line.strip()[1:-1].split('|')[0]
      maxent = torch.tensor([int(srv.split(', ')[0]) > 0], dtype=torch.float)
    else:
      maxent = torch.tensor([self.maxent], dtype=torch.float)
    
    srv, com, ff1 = encode(line)
    com = torch.tensor(com, dtype=torch.long)
    ff1 = torch.tensor(ff1, dtype=torch.float)
    srv = torch.tensor(srv, dtype=torch.long)
    file = torch.tensor(-1, dtype=torch.long)
    
    if self.noff1:
      ret = (com, maxent, srv, file)
    else:
      ret = (com, ff1, maxent, srv, file)
    
    return ret


class TxtDataLoader(object):
  def __init__(self, file, maxent=-1, batch_size=1, collate_fn=collate, drop_last=False):
    self.batch_size = batch_size
    self.collate_fn = collate_fn
    self.drop_last = drop_last
    self.dataset = TxtDataset(file, maxent)
    self.stop_iter = False
  
  def __iter__(self):
    return self
  
  def __next__(self):
    batch = []
    
    for _, sample in zip(range(self.batch_size), self.dataset):
      batch.append(sample)
    
    if len(batch) == 0 or (self.drop_last and len(batch) < self.batch_size):
      raise StopIteration
    
    return self.collate_fn(batch)


class MelvinRandomDataLoader(object):
  """This data loader directly produces PackedSequence objects consisting of randomly chosen 
  components. The sequences may have different lengths of 6, 7, ..., 15. We avoid sampling on a 
  per-sequence basis, i.e. for each sequence sample a sequence length l and then l components. 
  Instead, we sample a whole batch at once. The sequence lengths in a batch of size n follow a 
  multinomial distribution with n trials and 10 categories (all with probability 0.1). From this 
  distribution we sample the length frequencies of the next batch and construct a PackedSequence 
  object from them. 
  
  When device='cuda' is selected, the data is sampled directly on GPU. 
  """
  
  def __init__(self, batch_size=1, min_len=6, max_len=15, device='cpu'):
    self.batch_size = torch.tensor(batch_size).long()
    self.device = device
    self.min = min_len
    self.max = max_len
    self.n = max_len - min_len + 1
    self.multinomial = Multinomial(batch_size, torch.ones(self.n))
  
  def __iter__(self):
    return self
  
  def __next__(self):
    len_freq = self.multinomial.sample().long()
    # batch_sizes has self.max entries, the first self.min must have value batch_size
    batch_sizes = self.batch_size - len_freq[:self.n-1].cumsum(0)
    batch_sizes = batch_sizes[batch_sizes>0]
    batch_sizes = torch.cat([self.batch_size.repeat(self.min), batch_sizes])
    data_size = (torch.arange(self.min, self.max+1) * len_freq).sum().item()
    data = torch.randint(len(commap), (data_size,), device=self.device)
    return PackedSequence(data, batch_sizes)


class TriggerLoader(object):
  def __init__(self, ff2, batch_size):
    self.i = 1
    self.j = 0
    self.batch_size = batch_size
    self.ff2 = ff2
    self.done = (ff2.long().sum().item() == 0)
  
  
  def __iter__(self):
    return self
  
  
  def __next__(self):
    if self.done:
      raise StopIteration
    
    num_samples = 0
    batch = []
    idx = []
    do_break = False
    
    for j in range(self.j, self.ff2.shape[0]):
      n_bits = self.ff2[j].long().sum().item()
      
      for i in range(self.i, 2**n_bits):
        if num_samples == self.batch_size:
          self.i = i
          self.j = j
          do_break = True
          break
        
        bits = torch.tensor([int(k) for k in ('{0:0%db}' % n_bits).format(i)], 
                            dtype=self.ff2.dtype, device=self.ff2.device)
        sample = self.ff2[j].clone()
        sample[sample.type(torch.uint8)] = bits
        batch.append(sample)
        idx.append(j)
        num_samples += 1
      
      if do_break:
        break
      else:
        self.i = 1
    
    if not do_break:
      self.done = True
    
    return torch.stack(batch), idx


def test_txt_data_loader():
  file = '/home/tomte/data/melvin/txt_files/WrongLarge_NonMaxEnt_20190117_5.txt'
  ldr = TxtDataLoader(file, batch_size=16)
  
  for com, ff1, maxent, srv, file in ldr:
    print(com)


def create_noff1_split(test_prob=0.3):
  con = sqlite3.connect('melvinNoFF1.db', detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  c.execute('ATTACH DATABASE \'melvinNoFF1Train.db\' as train')
  c.execute('CREATE TABLE train.data (n INTEGER, m INTEGER, k INTEGER, \
             maxent INTEGER, com ARRAY, file INTEGER)')
  c.execute('ATTACH DATABASE \'melvinNoFF1Test.db\' as test')
  c.execute('CREATE TABLE test.data (n INTEGER, m INTEGER, k INTEGER, \
             maxent INTEGER, com ARRAY, file INTEGER)')
  num_samples = c.execute('SELECT count(*) FROM main.data').fetchone()[0]
  
  for i in range(num_samples):
    db = 'test' if random() <= test_prob else 'train'
    c.execute('INSERT INTO ' + db + '.data (n, m, k, maxent, com, file) \
               SELECT n, m, k, maxent, com, file FROM main.data WHERE rowid = ' + str(i+1))
  
  con.commit()
  con.close()


def create_noff1_ccv():
  c = sqlite3.connect('melvinNoFF1Train.db', detect_types=sqlite3.PARSE_DECLTYPES).cursor()
  where = ['n < 2'] + ['n = %d' % i for i in range(2, 10)] + ['n > 9']
  folds = []
  label = []
  
  for w in where:
    res = c.execute('SELECT rowid, maxent FROM data WHERE ' + w).fetchall()
    folds.append([i[0] - 1 for i in res])
    label.append([i[1] for i in res])
  
  # split the first where clause in half becoming the first two folds
  perm = torch.randperm(len(folds[0])).tolist()
  
  for l in [folds, label]:
    l0 = l[0]
    off = int(len(l0)/2)
    l[0] = [l0[i] for i in perm][:off]
    l.insert(0, [l0[i] for i in perm][off:])
  
  for i in range(2, len(folds)):
    perm = torch.randperm(len(folds[i])).tolist()
    folds[i] = [folds[i][j] for j in perm]
    label[i] = [label[i][j] for j in perm]
  
  return folds, label


def create_noff1_ccv__sanity_check():
  #fold, _ = create_noff1_ccv()
  fold, _ = create_noff1_special_fold()
  
  # check if folds are disjoint
  for i in range(len(fold)):
    for j in range(i+1, len(fold)):
      assert(len(set(fold[i]) & set(fold[j])) == 0)
  
  # check if no samples are missing using Gauss' summation formula
  l = sum([len(f) for f in fold]) # total number of samples in CCV
  assert(sum([sum(f) for f in fold]) == l * (l-1) / 2)


def create_noff1_special_fold():
  c = sqlite3.connect('melvinNoFF1Train.db', detect_types=sqlite3.PARSE_DECLTYPES).cursor()
  w1 = 'n = 6 and m = 3 and k = 3 or ' + \
       'n = 6 and m = 5 and k = 2 or ' + \
       'n = 7 and m = 5 and k = 3 or ' + \
       'n = 7 and m = 5 and k = 4 or ' + \
       'n = 7 and m = 7 and k = 4'
  w2 = '(n != 6 or m != 3 or k != 3) and ' + \
       '(n != 6 or m != 5 or k != 2) and ' + \
       '(n != 7 or m != 5 or k != 3) and ' + \
       '(n != 7 or m != 5 or k != 4) and ' + \
       '(n != 7 or m != 7 or k != 4)'
  folds = []
  label = []
  
  for w in [w1, w2]:
    res = c.execute('SELECT rowid, maxent FROM data WHERE ' + w).fetchall()
    folds.append([i[0] - 1 for i in res])
    label.append([i[1] for i in res])
  
  for i in range(2):
    perm = torch.randperm(len(folds[i])).tolist()
    folds[i] = [folds[i][j] for j in perm]
    label[i] = [label[i][j] for j in perm]
  
  return folds, label


def create_noff1_no6xx_ccv():
  c = sqlite3.connect('melvinNoFF1Train.db', detect_types=sqlite3.PARSE_DECLTYPES).cursor()
  where = ['n < 2'] + ['n = %d' % i for i in [2, 3, 4, 5, 7, 8, 9]] + ['n > 9']
  folds = []
  label = []
  
  for w in where:
    res = c.execute('SELECT rowid, maxent FROM data WHERE ' + w).fetchall()
    folds.append([i[0] - 1 for i in res])
    label.append([i[1] for i in res])
  
  # split the first where clause in half becoming the first two folds
  perm = torch.randperm(len(folds[0])).tolist()
  
  for l in [folds, label]:
    l0 = l[0]
    off = int(len(l0)/2)
    l[0] = [l0[i] for i in perm][:off]
    l.insert(0, [l0[i] for i in perm][off:])
  
  for i in range(2, len(folds)):
    perm = torch.randperm(len(folds[i])).tolist()
    folds[i] = [folds[i][j] for j in perm]
    label[i] = [label[i][j] for j in perm]
  
  return folds, label


def create_noff1_n9_split():
  con = sqlite3.connect('melvinNoFF1.db', detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  
  # create tmp database with n < 9
  c.execute('ATTACH DATABASE \'melvinNoFF1N9Tmp.db\' as train')
  c.execute('CREATE TABLE train.data (n INTEGER, m INTEGER, k INTEGER, \
             maxent INTEGER, com ARRAY, file INTEGER)')
  c.execute('INSERT INTO train.data (n, m, k, maxent, com, file) \
             SELECT n, m, k, maxent, com, file FROM main.data WHERE n < 9')
  
  # create test database with n >= 9
  c.execute('ATTACH DATABASE \'melvinNoFF1N9Test.db\' as test')
  c.execute('CREATE TABLE test.data (n INTEGER, m INTEGER, k INTEGER, \
             maxent INTEGER, com ARRAY, file INTEGER)')
  c.execute('INSERT INTO test.data (n, m, k, maxent, com, file) \
             SELECT n, m, k, maxent, com, file FROM main.data WHERE n >= 9')
  
  con.commit()
  con.close()


def create_noff1_n9_split2(test_prob=0.2):
  con = sqlite3.connect('melvinNoFF1N9Tmp.db', detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  c.execute('ATTACH DATABASE \'melvinNoFF1N9Train.db\' as train')
  c.execute('CREATE TABLE train.data (n INTEGER, m INTEGER, k INTEGER, \
             maxent INTEGER, com ARRAY, file INTEGER)')
  c.execute('ATTACH DATABASE \'melvinNoFF1N9Test2.db\' as test')
  c.execute('CREATE TABLE test.data (n INTEGER, m INTEGER, k INTEGER, \
             maxent INTEGER, com ARRAY, file INTEGER)')
  num_samples = c.execute('SELECT count(*) FROM main.data').fetchone()[0]
  
  for i in range(num_samples):
    db = 'test' if random() <= test_prob else 'train'
    c.execute('INSERT INTO ' + db + '.data (n, m, k, maxent, com, file) \
               SELECT n, m, k, maxent, com, file FROM main.data WHERE rowid = ' + str(i+1))
  
  con.commit()
  con.close()


def create_noff1_n9_ccv():
  c = sqlite3.connect('melvinNoFF1N9Train.db', detect_types=sqlite3.PARSE_DECLTYPES).cursor()
  where = ['n < 2'] + ['n = %d' % i for i in range(2, 9)]
  folds = []
  label = []
  
  for w in where:
    res = c.execute('SELECT rowid, maxent FROM data WHERE ' + w).fetchall()
    folds.append([i[0] - 1 for i in res])
    label.append([i[1] for i in res])
  
  # split the first where clause in half becoming the first two folds
  perm = torch.randperm(len(folds[0])).tolist()
  
  for l in [folds, label]:
    l0 = l[0]
    off = int(len(l0)/2)
    l[0] = [l0[i] for i in perm][:off]
    l.insert(0, [l0[i] for i in perm][off:])
  
  for i in range(2, len(folds)):
    perm = torch.randperm(len(folds[i])).tolist()
    folds[i] = [folds[i][j] for j in perm]
    label[i] = [label[i][j] for j in perm]
  
  return folds, label


def create_noff1_positives():
  con = sqlite3.connect('melvinNoFF1.db', detect_types=sqlite3.PARSE_DECLTYPES)
  c = con.cursor()
  
  c.execute('ATTACH DATABASE \'melvinNoFF1Maxent.db\' as other')
  c.execute('CREATE TABLE other.data (n INTEGER, m INTEGER, k INTEGER, \
             maxent INTEGER, com ARRAY, file INTEGER)')
  c.execute('INSERT INTO other.data (n, m, k, maxent, com, file) \
             SELECT n, m, k, maxent, com, file FROM main.data WHERE maxent = 1')
  
  con.commit()
  con.close()


class CrossValidationDataset(Dataset):
  def __init__(self, dataset, folds, label, leave_out=-1, training=True, task='maxent'):
    if leave_out < 0: # if there's no validation fold, there's no validation!
      assert(training)
    
    self.ds = dataset
    self.leave_out = leave_out
    folds = folds.copy()
    
    if leave_out >= 0:
      self.valid_idx = folds.pop(leave_out)
    
    self.train_idx = list(itertools.chain(*folds))
    label = label.copy()
    
    if leave_out >= 0:
      self.valid_lbl = label.pop(leave_out)
    
    self.train_lbl = list(itertools.chain(*label))
    self.training = training
    self.train_folds = folds
    self.task = task
  
  
  def __getitem__(self, idx):
    return self.ds[self.train_idx[idx] if self.training else self.valid_idx[idx]]
  
  
  def __len__(self):
    return len(self.train_idx if self.training else self.valid_idx)
  
  
  def getSampler(self):
    if not self.training:
      return SequentialSampler(self)
    
    if self.task == 'maxent':
      labels = self.train_lbl
      num_all = len(labels)
      num_pos = sum(labels)
      labels = np.array(labels, dtype=np.float)
      weights = -(labels.copy() - 1)
      labels *= num_all / num_pos
      weights *= num_all / (num_all - num_pos)
      ret = WeightedRandomSampler(list(weights+labels), num_pos * 2)
    elif self.task == 'srv':
      weights = 1 / np.array([len(f) for f in self.train_folds], dtype=np.float)
      weights = [[w] * len(f) for w, f in zip(weights, self.train_folds)]
      weights = list(itertools.chain(*weights))
      # FIXME: any suggestions for the num_samples parameter?
      ret = WeightedRandomSampler(weights, 3000000)
    
    return ret


def save_folds():
  torch.save(create_noff1_n9_ccv(), 'folds_n9.pt')

if __name__ == '__main__':
  folds, label = torch.load('/publicwork/adler/folds.pt')
  ds = CrossValidationDataset(MelvinDataset('melvinNoFF1Train.db'), folds, label, 
                              8, task='srv')
  ldr = DataLoader(ds, collate_fn=collate_noff1, batch_size=50, sampler=ds.getSampler())
  
  fold_freq = [0] * 11
  
  for i, (com, maxent, srv, file) in enumerate(ldr):
    for s in srv:
      s = s[0].item()
      
      if s >= 10:
        s = 10
      
      fold_freq[s] += 1
    
    if i >= 100:
      break
  
  print(fold_freq)
  exit()
  
  
  ds = MelvinDataset('melvinNoFF1.db')
  test_set, train_set = [Subset(ds, idx) for idx in torch.load('split.pt')]
  
  print(test_set, train_set)
  
  ldr = DataLoader(train_set, collate_fn=collate_noff1, batch_size=10, 
                   sampler=ds.getWeightedRandomSampler())
  l = []
  
  
  for i, (com, maxent, srv, file) in enumerate(ldr):
    if i >= 1000:
      break
    
    l.append(maxent.sum().item())
  
  print(sum(l) / (10 * len(l)))


if False: #__name__ == '__main__':
  cmdmap = {
    'create_sqlite_database_from_textfiles': create_sqlite_database_from_textfiles,
    'delete_duplicates': delete_duplicates,
    'sanity_check': sanity_check,
    'create_sqlite_database_subset': create_sqlite_database_subset,
    'create_g6_sqlite_databases': create_g6_sqlite_databases,
    'create_all_sqlite_databases': create_all_sqlite_databases,
    'create_test_split_databases': create_test_split_databases,
    'stats': stats,
    'visualize_stats': visualize_stats,
    'test_txt_data_loader': test_txt_data_loader,
  }
  
  import argparse
  parser = argparse.ArgumentParser(description='data reading routines')
  parser.add_argument('cmd', type=str, choices=list(cmdmap.keys()), help='command')
  parser.add_argument('--dbfile', type=str, default='melvinFull.db', help='SQLite file')
  parser.add_argument('--src', type=str, default='melvinFull.db', help='SQLite file')
  parser.add_argument('--dst', type=str, default='melvinFull.db', help='SQLite file')
  parser.add_argument('--txtdir', type=str, default='/tomte/home/data/melvin/txt_files', 
                      help='directory where txt files are located')
  parser.add_argument('--where', type=str, default='', help='SQL WHERE clause defining the subset')
  args = parser.parse_args()
  
  # assemble arguments for function call
  if args.cmd == 'create_sqlite_database_from_textfiles':
    cmd_args = (args.dbfile, args.txtdir)
  elif args.cmd == 'create_sqlite_database_subset':
    cmd_args = (args.src, args.dst, args.where)
  elif args.cmd in ['create_all_sqlite_databases', 'test_txt_data_loader']:
    cmd_args = ()
  else:
    cmd_args = (args.dbfile,)
  
  cmdmap[args.cmd](*cmd_args)


if False: #__name__ == '__main__':
  line = '{-4, 4, -2} | {"OAMHolo[XXX,d,1]", "BS[XXX,a,d]", "OAMHolo[XXX,c,-1]", ' + \
    '"OAMHolo[XXX,b,-1]", "BS[XXX,c,d]", "DP[XXX,d]"} | {FF1[1]}'
  print(line)
  srv, com, ff1 = encode(line)
  print(srv, com, ff1)
  line = decode(srv, com, ff1)
  print(line)


if False: #__name__ == '__main__':
  import time
  
  start = time.clock()
  

  #con.close()
  
  melvin = MelvinDataset('melvin.db', 'data')
  ldr = DataLoader(melvin, batch_size=4, shuffle=True, num_workers=1, 
                   collate_fn=MelvinDataset.collate)
  
  print('init took', time.clock() - start, 'secs')
  
  start = time.clock()
  
  for i, batch in enumerate(ldr):
    com, ff1, maxent = batch
    print(com.data)
    if i >= 1000:
      break
  
  print('reading 1000 size-4 batches took', time.clock() - start, 'secs')
  
  con.close()










