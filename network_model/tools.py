# -*- coding: utf-8 -*-
"""
Tools to run network simulations. 

@author: Adrian Holtrup
"""

import numpy as np
import random
from scipy.spatial.distance import cdist


def sign(r):  
    """
    Returns the sign of r; sign(0) := -1.
    """
    
    out = np.where(np.asarray(r) > 0, 1, -1)
    
    return out.item() if np.ndim(out) == 0 else out


def gen_patterns(d, n_pats):   
    """
    Draws Gaussian patterns of dimension d with normalized scale.
    Binary labels are assigned randomly.
    """
    
    pats = np.random.normal(scale=1/np.sqrt(d), size=(n_pats, d))
    labels = np.random.choice([-1,1], size=(n_pats), replace=True)
    
    return pats, labels


def RBF_kernel(X, gamma): 
    """
    Computes pairwise RBF kernel values with length scale gamma for an 
    array of vectors X.
    """
    
    squared_norm = cdist(X, X, metric='sqeuclidean')
    
    return np.exp(-squared_norm / (2 * gamma ** 2))


def draw_GP_functions(X, gamma):
    """
    Maps vectors in X to targets by drawing functions from a Gaussian process.
    The covariance matrix of the GP is given by an RBF kernel with 
    length scale gamma.
    Adapted from Xie et al. (Elife) 2024.
    """
    
    p = X.shape[0]
    C = RBF_kernel(X, gamma)
    constant = 1e-10
    
    L = np.linalg.cholesky(C + constant * np.eye(p))
    y = np.dot(L, np.random.randn(p)).T
    
    y = y / y.std()
    
    return y.T


def jam_patterns(X, s):
    """
    Trades off signal to Gaussian noise for an array of vectors X.
    """
    
    d = X.shape[-1]
    
    X_scaled = np.sqrt(1-s**2) * X 
    noise = s * np.random.normal(scale=1/np.sqrt(d), size=X.shape)
    
    X_jammed = X_scaled + noise
    
    return X_jammed


def genhaarorthmatrix(n):
    """
    Generates a random orthogonal matrix of dimension n from the Haar group.
    Adapted from Muscinelli et al. (Nat Neuroscience) 2023.
    """
    
    H = np.diag(np.ones(n))
    D = np.ones(n)
    
    for l in range(n - 1):   
        x = np.random.randn(n - l, 1)
        D[l] = np.sign(x[0,0])
        x[0] = x[0] - D[l] * np.linalg.norm(x)
        
        Hx = np.diag(np.ones(n - l)) - (2 * x) @ x.T / (np.linalg.norm(x) ** 2)
        mat = np.diag(np.ones(n))
        mat[l:, l:] = Hx[:, :]
        H = H @ mat
        
    D[-1] = (-1) ** (1 - (n % 2)) * np.prod(D)
    H = np.dot(np.diag(D), H)
    
    return H


def gen_projection(d_in, d_out, k):
    """
    Generates a d_in x d_out matrix with k non-zero values per column.
    Non-zero elements are determined independently by sampling k indices
    per column without replacement. The value of the non-zero elements is
    drawn from a scaled Gaussian. 
    """
    
    p_mat = np.zeros((d_in, d_out))
    
    for i in range(d_out):  
        indices = np.random.choice(d_in, size=k, replace=False)
        values = np.random.normal(scale=1, size=k) * np.sqrt(1/k)
        
        p_mat[indices, i] = values
        
    return p_mat


def create_embedding(X, p_mat, f):
    """
    Maps an array of vectors X onto a sparse representational space where 
    fraction f of the elements are non-zero. The mapping comprises
    a linear transformation defined by p_mat followed by a non-linear 
    sparsening. 
    """
    
    _, d_out = p_mat.shape
    
    raw_emb = X @ p_mat
    raw_emb = np.maximum(raw_emb, 0)
    
    k = int(f * d_out)
    
    threshold = np.partition(raw_emb, -k, axis=1)[:, -k][:, np.newaxis]
    
    emb = np.where(raw_emb >= threshold, raw_emb-threshold, 0)  
    
    return emb


def ens_predict(ens, W, x):
    """
    Computes the majority-vote prediction of an ensemble.

    ens : boolean masks
    W   : weight vectors
    x   : input vector
    """
    
    outs = np.array([w @ x[mask] for w, mask in zip(W, ens)])
    
    preds = sign(outs)
    ens_pred = sign(np.sum(preds))
    
    return ens_pred, preds, outs


def sign_err(pred, target):
    """
    Computes a signed error signal for a prediction pred and a true target. 
    """
    
    if pred > target:
        error = 1
        
    elif pred < target:
        error = -1
        
    else:
        error = 0
        
    return error


def train_classification(ens, W, X, y, delta_s=0, n_epochs=20, l_rate=0.001):
    """
    Trains an ensemble of classification predictors defined by subsampling 
    mask ens. 
    W is the list of weight vectors for each predictor. 
    X, y are patterns and labels. 
    Each individual pattern is visited n_epochs times. 
    """
    
    c = len(X)
    n_ags = len(ens)
    
    for e in range(n_epochs):
        a = list(range(c))
        random.shuffle(a)
        
        for n in a:
            inp = jam_patterns(X[n], delta_s)
            target = y[n]
            
            ens_pred, preds, _ = ens_predict(ens, W, inp)
            error = sign_err(ens_pred, target)
            
            for i in range(n_ags):
                ag_inp = inp[ens[i]]
                loc_err = error + 1/2 * (preds[i] - ens_pred)
                
                W[i] -= l_rate * ag_inp * loc_err #update in place
              

def test_classification(ens, W, X, y):
    """
    Calculates the error rate of an ensemble of classification predictors 
    defined by subsampling mask ens and weight vectors W on a test set X, y. 
    """
    
    c = len(X)
    
    ens_EC = 0
    ind_EC = 0
    
    for i in range(c):
        inp = X[i]
        target = y[i]   
             
        ens_pred, preds, _ = ens_predict(ens, W, inp)
        ens_EC += sign_err(ens_pred, target)**2
        ind_EC += np.mean([sign_err(pred, target)**2 for pred in preds])
        
    ens_ER = ens_EC / c
    ind_ER = ind_EC / c
    
    return ens_ER, ind_ER 


def train_regression(ens, W, X, y, l_rate=0.1):
    """
    Trains an ensemble of regression predictors defined by subsampling 
    mask ens.
    W is the list of weight vectors for each predictor. 
    X and y are the training inputs and targets.
    The learning rate starts at l_rate and 
    is repeatedly halved until it falls below 1e-6.
    """
   
    c = len(X)
    n_ags = len(ens)
    
    e = 0
    c_rate = l_rate
    
    while c_rate >= 1e-6:
        EH = [float("inf"), float("inf"), 0]
        counter = 0
        iters = 0
        
        while counter < 3 and iters < 10e4:
            if EH[-2] < EH[-1]:
                counter += 1
                
            iters += 1    
            c_err = 0
            
            indices = list(range(c))
            random.Random(e).shuffle(indices)
            
            for n in indices:
                inp = X[n]
                target = y[n]
                
                _, _, outs = ens_predict(ens, W, inp)
                ens_pred = np.mean(outs)
                
                error = ens_pred - target
                c_err += error ** 2
                
                for i in range(n_ags):
                    ag_inp = inp[ens[i]]
                    loc_err = error + 1 * (outs[i] - ens_pred)

                    W[i] -= c_rate * ag_inp * loc_err #updated in place
                    
            EH.append(c_err)  
            e += 1
            
        c_rate *= 1/2     
                

def test_regression(ens, W, X, y):
    """
    Calculates the mean squared error of an ensemble of regression predictors 
    defined by subsampling mask ens and weight vectors W on a test set X, y. 
    """
    
    c = len(X)
    
    ens_EC = 0
    ind_EC = 0
    
    for i in range(c):
        inp = X[i]
        target = y[i] 
               
        _, _, outs = ens_predict(ens, W, inp)
        ens_pred = np.mean(outs)
        
        ens_EC += (ens_pred - target) ** 2
        ind_EC += np.mean((outs - target) ** 2)
        
    ens_MSE = ens_EC / c
    ind_MSE = ind_EC / c

    return ens_MSE, ind_MSE               