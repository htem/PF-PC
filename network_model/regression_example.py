# -*- coding: utf-8 -*-
"""
Minimal example of a Ridge regression task. 

@author: Adrian Holtrup
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE
import tools


n = 25 # input layer size (mossy fibers)
m = 1000 # expansion layer size (granule cells)
n_ags = 20 # ensemble layer size (Purkinje cells)

k = n # in-degree of the expansion layer
f = 0.2 # coding-level of the expansion layer

p = 3000 # number of training samples (underparameterized region)
d = 5 # dimension of task patterns
gamma = 0.5 # RBF kernel length scale
sigma = 0.2 # training noise (generalization difficulty)
lambda_ = 0.01 # Ridge penalty 
test_ratio = 5 # test patterns per training patterns


### start simulation

# generate patterns
D = np.random.normal(
    size=(p * (test_ratio + 1), d),
    scale=1,
)

D /= np.sqrt(np.sum(D ** 2, axis=1, keepdims=True))

D_train, D_test = D[:p], D[p:]

targets = draw_GP_functions(D, gamma)
targets_train = targets[:p].copy()
targets_test = targets[p:].copy()

targets_train += np.random.normal(
    scale=np.sqrt(sigma),
    size=targets_train.shape,
)

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
    
    test_preds = np.empty((D_test.shape[0], n_ags))
    
    for ag in range(n_ags):

        # create subsampling masks
        rand_subs = np.random.random(size=m) < r
        
        ag_reg = Ridge(
            fit_intercept=False, 
            alpha=lambda_
            )  
        ag_reg.fit(M_train[:, rand_subs], targets_train)
        
        test_pred = ag_reg.predict(M_test[:, rand_subs])
        test_preds[:,ag] = test_pred
            
    ind_MSEs = np.mean((test_preds - targets_test[:, None]) ** 2, axis=0)
    ind_MSE = np.mean(ind_MSEs)
    
    # take uniform average
    ens_MSE = MSE(targets_test, np.mean(test_preds, axis=1))  
    
    print(r, ens_MSE, ind_MSE)
                                              
