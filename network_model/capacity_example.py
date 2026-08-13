# -*- coding: utf-8 -*-
"""
Minimal example of pattern storage. 

@author: Adrian
"""

import numpy as np
from cvxopt import matrix, solvers
import tools


def check_feasibility(H, y, bias=0.5):
    """
    Check whether there exists a nonnegative weight vector w satisfying
    the linear constraints induced by H and y.
    """
    
    opts = {'maxiters' : 30}
    
    _, d = H.shape

    c = matrix(np.zeros(d))

    G = -np.diag(y) @ H
    h = -bias * y.reshape(len(y), 1)

    G2 = -np.eye(d)
    h2 = np.zeros((d, 1))

    G = matrix(np.vstack([G, G2]))
    h = matrix(np.vstack([h, h2]))
    
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h, options=opts)
    
    if sol['status'] == 'optimal':
        w = np.array(sol['x']).flatten()
        return True, w
    else:
        return False, None
    
    
### set network parameters

n = 25 # input layer size (mossy fibers)
m = 1000 # expansion layer size (granule cells)
n_ags = 20 # ensemble layer size (Purkinje cells)

k = n # in-degree of the expansion layer
f = 0.1 # coding-level of the expansion layer

p = 50 # number of task patterns
d = n # dimension of task patterns

s = 0.5 # label redraw probability
    

### start simulation

# generate patterns
D, y = gen_patterns(d, p)    

# generate embedding matrix
Om = genhaarorthmatrix(n)
A = Om[:, :d].astype(np.float32)

# embed patterns in input layer
N = (A @ D.T).T 

# generate projection
p_mat = gen_projection(n, m, k)

# project to expansion layer
M = create_embedding(N, p_mat, f)

W_dicts = []

for ag in range(n_ags):
    
    # redraw labels with probability s
    mask = np.random.rand(p) < s
    new_labels = np.random.choice([-1, 1], size=p)
    y_ag = np.where(mask, new_labels, y)
    
    p_ag = m // 3
    skip = p_ag // 20
    
    limit = False
    ever_feasible = False
    
    while skip > 1:
        feasible, w_prelim = check_feasibility(M[:p_ag], y_ag[:p_ag])
        
        if feasible:
            ever_feasible = True
            w = w_prelim
            
            if limit:
                skip //= 2
            p_ag += skip
            
        elif not ever_feasible:
            p_ag = p_ag * 3 // 4    
            
        elif skip > 1:
            limit = True
            
            skip //= 2
            p_ag -= skip
            
    feasible, w_prelim = check_feasibility(M[:p_ag], y_ag[:p_ag])
    
    if feasible:
        w = w_prelim
    
    else:
        p_ag -= skip
        
    W_dicts.append({'ag' : ag,
                    'w' : w, 
                    'p' : p_ag})

W_DF = pd.DataFrame(W_dicts)
