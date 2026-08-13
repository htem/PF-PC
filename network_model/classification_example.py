# -*- coding: utf-8 -*-
"""
Minimal example of a classification task. 

@author: Adrian Holtrup
"""

import numpy as np
import tools


### set network parameters

n = 25 # input layer size (mossy fibers)
m = 1000 # expansion layer size (granule cells)
n_ags = 20 # ensemble layer size (Purkinje cells)

k = n # in-degree of the expansion layer
f = 0.1 # coding-level of the expansion layer

p = 50 # number of task patterns
d = n # dimension of task patterns
sigma = 0.5 # generalization difficulty
test_ratio = 5 # test patterns per training patterns


### start simulation

# generate patterns
X, y = gen_patterns(d, p)    
D = np.tile(X, (test_ratio+1,1))
D[p:] = jam_patterns(D[p:], sigma)
D_train, D_test = D[:p], D[p:]
y = np.tile(y, (test_ratio+1))
y_train, y_test = y[:p], y[p:]

# generate embedding matrix
Om = genhaarorthmatrix(n)
A = Om[:, :d].astype(np.float32)

# embed patterns in input layer
N_train = (A @ D_train.T).T 
N_test = (A @ D_test.T).T

# generate projection
p_mat = gen_projection(n, m, k)

# project to expansion layer
M_train = create_embedding(N_train, p_mat, f)
M_test = create_embedding(N_test, p_mat, f)

# train and test network iterating through subsampling coefficient rho (r)
for r in np.linspace(0.1, 1.0, 10): 
     
    ens = np.zeros((n_ags, m), dtype=bool)
    W = []
    
    for ag in range(n_ags):
        
        # create subsampling masks
        rand_subs = np.random.random(size=m) < r
        
        ens[ag] = rand_subs
        
        # initialize readout weights
        w_ag = np.random.normal(
            size=rand_subs.sum(),
            scale=0.001
            )
        
        W.append(w_ag)
        
    train_classification(
        ens, 
        W, 
        M_train, 
        y_train, 
        l_rate=0.1,
        )
    
    ens_ER, ind_ER = test_classification(
        ens, 
        W, 
        M_test, 
        y_test
        )
    
    print(r, ens_ER, ind_ER)
        