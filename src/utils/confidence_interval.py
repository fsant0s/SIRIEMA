import numpy as np 
from scipy.stats import t

def confidence_interval(x):
  m = np.mean(x)
  s = np.std(x)
  dof = len(x)-1 
  confidence = 0.95
  t_crit = np.abs(t.ppf((1-confidence)/2,dof))
  return (m-s*t_crit/np.sqrt(len(x)), m,  m+s*t_crit/np.sqrt(len(x))) 