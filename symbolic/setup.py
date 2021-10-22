import sympy as sp


a, b, c, d, e, f = sp.symbols('a, b, c, d, e, f', cls=sp.IndexedBase)
l, l1, l2, l3, l4, l5, l6, l7, l8 = sp.symbols('l, l1, l2, l3, l4, l5, l6, l7, l8', cls=sp.Wild)
sq2 = sp.sqrt(2)/2
args = {'map': False, 'simultaneous': True, 'exact': False}


def DownConvOAM(l_order, p1, p2):
  psi = 0
  for i in range(-l_order, l_order+1):
    psi = p1[i] * p2[-i] + psi
  return psi


def __BS(expr, p1, p2): 
  if expr.base == p1:
    return expr.replace(p1[l], sq2 * (p2[l] + sp.I * p1[-l]), **args)
  else: 
    return expr.replace(p2[l], sq2 * (p1[l] + sp.I * p2[-l]), **args)


def BS(psi, p1, p2):
  return sp.expand(psi.replace(lambda expr: expr.base in [p1, p2], 
                               lambda expr: __BS(expr, p1, p2)))


def __LI(expr, p1, p2): 
  if expr.base == p1:
    x = (sp.cos(l1 * sp.pi/2)**2) * p1[l1] + sp.I * (sp.sin(l1 * sp.pi/2)**2) * p2[-l1]
    return expr.replace(p1[l1], x, **args)
  else: 
    x = -(sp.cos(l1 * sp.pi/2)**2) * p2[l1] + sp.I * (sp.sin(l1 * sp.pi/2)**2) * p1[-l1]
    return expr.replace(p2[l1], x, **args)


def LI(psi, p1, p2):
  return sp.expand(psi.replace(lambda expr: expr.base in [p1, p2], 
                               lambda expr: __LI(expr, p1, p2)))


def Reflection(expr, p):
  return expr.replace(p[l1], sp.I * p[-l1], **args)


def OAMHolo(expr, p, n):
  return expr.replace(p[l1], p[l1+n], **args)


def DP(expr, p):
  return expr.replace(p[l1], sp.I * sp.exp(sp.I * l1 * sp.pi) * p[-l1], **args)


def sympify(com, oam_order):
  state = '(DownConvOAM(%d, a, b) + DownConvOAM(%d, c, d))**2' % (oam_order, oam_order)
  
  for s in com:
    state = s.replace('XXX', state)
  
  return sp.expand(eval(state))


if __name__ == '__main__':
  #print(sp.expand(eval('(DownConvOAM(1, a, b) + DownConvOAM(1, c, d))**2')))
  #print('--------------------------------------------------------------')
  
  w = sp.symbols('w')
  term = w * a[l1] * b[l2] * c[l3] * d[l4]
  print('term')
  print(term, '\n')
  
  print('BS')
  print(BS(term, a, b), '\n')
  
  #print('LI')
  #print(LI(term, a, b), '\n')
  
  #print('Reflection')
  #print(Reflection(term, a), '\n')
  
  #print('OAMHolo')
  #print(OAMHolo(term, f, -5), '\n')
  
  #print('DP')
  #print(DP(term, a), '\n')
  
  
  
  

