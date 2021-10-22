if __name__ == '__main__':
  import sys
  sys.path.append('..')


import sympy as sp
from itertools import combinations, chain
from symbolic.setup import a, b, c, d, e, f, l, l1, l2, l3, l4, l5, l6, l7, l8, args, sympify


XXX = sp.symbols('XXX')
ff1, ff2, ff3, ff4, ffn = sp.symbols('ff1, ff2, ff3, ff4, ffn', cls=sp.IndexedBase)
hh, gg1, gg2, gg3, gg4 = sp.symbols('hh, gg1, gg2, gg3, gg4', cls=sp.IndexedBase)
x1, x2, x3, x4, x5, x6, w, n = sp.symbols('x1, x2, x3, x4, x5, x6, w, n', cls=sp.Wild)
zero, psi = sp.symbols('zero, psi')


def replace_map(expr, query_value_map):
  for q, v in query_value_map.items():
    expr = expr.replace(q, v, **args)
  
  return expr


def make_ff2(expr): 
  repl = {w * a[l1] * a[l2]: 0, 
          w * b[l1] * b[l2]: 0, 
          w * c[l1] * c[l2]: 0, 
          w * d[l1] * d[l2]: 0}
  expr1 = replace_map(expr, repl)
  expr2 = replace_map(expr1, {w * a[l2] * b[l3] * c[l4] * d[l5]: 0})
  expr = expr1 - expr2
  q = w * a[l1] * b[l2] * c[l3] * d[l4]
  v = w * ff1[l1] * ff2[l2] * ff3[l3] * ff4[l4]
  return expr.replace(q, v)


def trigger(expr, p1, n_list):
  lt = sp.Wild('lt', exclude=n_list)
  expr = expr.replace(p1[lt], 0)
  return expr.replace(p1[l], 1)


def get_all_triggers(ffn_types):
  return list(chain.from_iterable(combinations(ffn_types,n) for n in range(len(ffn_types)+1)))


def check_coeffs(expr):
  coeff = list(sp.collect(expr, [ff2[x1] * ff3[x2] * ff4[x3]], evaluate=False).values())
  
  for i, v in enumerate(coeff):
    coeff[i] = sp.expand(v * sp.conjugate(v))
    
    if i > 0 and coeff[i] != coeff[i-1]:
      return len(coeff), False
  
  return len(coeff), True


def distinct_ffn(expr, ffn):
  term = set(sp.collect(expr, [ffn[x1]], evaluate=False).keys())
  return set(t.indices[0] for t in term)


def to_hh(expr):
  expr1 = expr.replace(w * ff2[l1] * ff3[l2] * ff4[l3], 
                       sp.conjugate(w) * gg2[l1] * gg3[l2] * gg4[l3])
  rho0 = sp.expand(expr * expr1)
  return rho0.replace(w * ff2[l1] * ff3[l2] * ff4[l3] * gg2[l4] * gg3[l5] * gg4[l6], 
                      w * hh[l1, l2, l3, l4, l5, l6])


def partial_trace(expr, n):
  term_coeff = sp.collect(expr, [hh[l1, l2, l3, l4, l5, l6]], evaluate=False)
  particle1, particle2 = [], []
  
  for t in term_coeff.keys():
    if t.indices[n-1] == t.indices[n+2]:
      ll = list(t.indices[:6])
      del(ll[n-1], ll[n+1])
      particle1.append(ll[0])
      particle2.append(ll[1])
  
  upper1 = max(particle1) + 1
  lower1 = min(min(particle1), 0)
  range1 = upper1 - lower1
  upper2 = max(particle2) + 1
  lower2 = min(min(particle2), 0)
  range2 = upper2 - lower2
  
  m_size = range1 * range2
  m = sp.SparseMatrix(m_size, m_size, {(0, 0): 0})
  
  for t, c in term_coeff.items():
    if t.indices[n-1] == t.indices[n+2]:
      ll = list(t.indices[:6])
      del(ll[n-1], ll[n+1])
      dim_row = (ll[0] - lower1) * range2 + (ll[1] - lower2)
      dim_col = (ll[2] - lower1) * range2 + (ll[3] - lower2)
      m += sp.SparseMatrix(m_size, m_size, {(dim_row, dim_col): c})
  
  return m.rank()


def schmidt_rank_vector(expr):
  rho = to_hh(expr)
  
  if rho == 0:
    return [0, 0, 0]
  else:
    return sorted([partial_trace(rho, i) for i in range(1, 4)], reverse=True)


def check_higher_order(com, triggers, atoff_lower, reduced_vv_lower):
  reduced_vv = trigger(make_ff2(sympify(com, 5)), ff1, triggers)
  atoff = [distinct_ffn(reduced_vv, ff) for ff in [ff2, ff3, ff4]]
  compl = [a.difference(a_lo) for a_lo, a in zip(atoff_lower, atoff)]
  
  for compl_, ff in zip(compl, [ff2, ff3, ff4]):
    for i in compl_:
      reduced_vv = reduced_vv.replace(ff[i], 0, **args)
  
  return reduced_vv == reduced_vv_lower


def check_setup(com, triggers):
  #reduced_vv = trigger(make_ff2(sympify(com, 1)), ff1, triggers)
  reduced_vv = trigger(make_ff2(sp.simplify(com)), ff1, triggers)
  
  atoff = [distinct_ffn(reduced_vv, ff) for ff in [ff2, ff3, ff4]]
  dim_vec = sorted([len(a) for a in atoff], reverse=True)
  n_terms, maxent = check_coeffs(reduced_vv)
  more_terms = (dim_vec[0] < n_terms)
  bisep = (dim_vec[2] <= 1)
  
  if not maxent or more_terms or bisep:
    msg = 'maxent=%s more_terms=%s bisep=%s' % (maxent, more_terms, bisep)
    return False, dim_vec, msg
  
  srv = schmidt_rank_vector(reduced_vv)
  
  if n_terms != srv[0] or srv[2] <= 1:
    return False, srv, 'no good SRV'
  
  higher_order = check_higher_order(com, triggers, atoff, reduced_vv)
  msg = 'positive' if higher_order else 'higher_order'
  return higher_order, srv, msg


def get_triggers_for_setup(com):
  return get_all_triggers(distinct_ffn(make_ff2(sympify(com, 1)), ff1))


if __name__ == '__main__':
  if False:
    com = ['Reflection(XXX,e)', 'OAMHolo(XXX,a,3)', 'LI(XXX,b,c)', 'OAMHolo(XXX,d,5)', 
           'OAMHolo(XXX,a,2)', 'OAMHolo(XXX,e,5)']
    triggers = [5, 6]
    
    from time import perf_counter
    
    print(com)
    start = perf_counter()
    x = sympify(com, 1)
    
    #exit()
    
    print(perf_counter() - start)
    print(x)
    
    start = perf_counter()
    x = make_ff2(x)
    print(perf_counter() - start)
    print(x)
    
    start = perf_counter()
    x = distinct_ffn(x, ff1)
    print(perf_counter() - start)
    print(x)
    
    start = perf_counter()
    triggers = get_all_triggers(x)
    print(perf_counter() - start)
    print(triggers)
    
    print(check_setup(com, triggers))
    
    exit()
  
  
  #com = ['Reflection(XXX,e)', 'OAMHolo(XXX,a,3)', 'LI(XXX,b,c)', 'OAMHolo(XXX,d,5)', 
  #       'OAMHolo(XXX,a,2)', 'OAMHolo(XXX,e,7)']
  #triggers = [4, 5]
  
  com = ['OAMHolo(XXX,b,4)', 'BS(XXX,a,b)', 'OAMHolo(XXX,c,2)', 'OAMHolo(XXX,b,1)', 'BS(XXX,c,d)', 
         'Reflection(XXX,c)', 'OAMHolo(XXX,f,-1)', 'OAMHolo(XXX,b,2)', 'DP(XXX,c)', 
         'OAMHolo(XXX,f,-2)', 'Reflection(XXX,b)', 'BS(XXX,a,d)', 'OAMHolo(XXX,b,1)', 
         'DP(XXX,f)', 'OAMHolo(XXX,f,-3)']
  triggers = [1]
  
  print(com)
  print(check_setup(com, triggers))
  exit()
  
  
  #com = ['OAMHolo(XXX,c,-3)', 'OAMHolo(XXX,a,3)', 'OAMHolo(XXX,f,-5)', 'OAMHolo(XXX,c,2)', 
  #       'LI(XXX,a,c)', 'BS(XXX,a,b)', 'OAMHolo(XXX,b,-3)', 'OAMHolo(XXX,e,-4)']
  com = ['OAMHolo(XXX,c,-3)', 'BS(XXX,b,d)', 'OAMHolo(XXX,f,-5)', 'OAMHolo(XXX,c,2)', 
         'LI(XXX,a,c)', 'BS(XXX,a,b)', 'OAMHolo(XXX,b,-3)', 'OAMHolo(XXX,e,-4)']
  print(distinct_ffn(make_ff2(sympify(com, 1)), ff1))
  #triggers = [1]
  
  #print(com)
  #print(check_setup(com, triggers))





