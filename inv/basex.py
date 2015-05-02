# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 01:27:08 2014

@author: felix
"""

import numpy as np
from scipy.special import factorial as fact
# basex basis set

def R_k(r, k, sigma):
    k_sq = (k ** 2).astype(np.float_)
    u = r / sigma
    u_sq = u ** 2
    exponent = k_sq - u_sq + k_sq * np.log(u_sq / k_sq)
    return np.exp(exponent)

def X(r, k, sigma):
    k_sq = np.float(k ** 2)
    u = r / sigma
    u_sq = u ** 2

    ll = np.arange(0, k + 1).astype(np.float_) ** 2
    gam = gamma_approx(ll)
    alph = aleph(ll)
    X_mat = np.zeros((k_sq, r.shape[0]))
    k_vec = np.arange(k + 1)
    X_mat =  R_k(r, k_vec[:, None], sigma)
    X_mat /= gam[::-1, None]
    return np.sum(X_mat[:-1] * alph[::-1, None], axis=0)

def gamma(ll):
#    ll = np.arange(0, l + 1) ** 2
    return (np.e / ll) ** ll * fact(ll)

def gamma_approx(ll):
    coeffs = np.array([1, 1/12., 1/288., -139/51840., -571/2488320.])
    exps = np.arange(5) * -1
    return np.sqrt(2 * np.pi * ll) * (coeffs[:, None] * (ll ** exps[:, None])).sum(0)

def aleph(m):
    ll_a = np.ones(m.shape[0] - 1)
#    m = np.arange(1, l + 1).astype(np.float) ** 2
    ll_a = ll_a - 1 / (2 * m[1::])
    return ll_a.cumprod()

###
###
###
# old attempts
def rho_k(r, k, sma):
    ksq = np.float(k ** 2)
    res = ((np.e / ksq) ** ksq) * ((r / sma) ** (2 * ksq) ) * \
    np.exp(-(r / sma) ** 2)
    return res

def iter_prod(ksq):
    k_vec = np.arange(ksq)
    res_prod = np.empty(k_vec.shape)
    for l in k_vec:
        l_prod = 1
        for m in np.arange(1, l + 1):
            l_prod *= ((ksq + 1 - m) * (m - 0.5)) / m
        res_prod[l] = l_prod

    return res_prod

def iter_sum(x, ksq, sma):
    try: type(ksq) == int
    except: TypeError('k must be an int')
#    res_sum = np.ones(ksq, ksq, x.shape[0])

    k_vec = np.arange(1, ksq + 1)
#    res_prod = np.empty([k_vec.shape[0], x.shape[0]])
#    for i, l in enumerate(k_vec):
    l_prod = np.empty(ksq)
    tmp_prod = 1.
    for i, m in enumerate(k_vec):
        tmp_prod *= ((ksq + 1 - m) * (m - 0.5)) / m
        l_prod[i] = tmp_prod
#    res_prod[i] = l_prod * (x / sma) ** (-2 * l)
#    res_sum = res_prod.sum(axis=0)

#    for l in np.arange(1, ksq + 1):
#        res_prod = iter_prod(ksq, l)
#        res_sum += (x / sma) ** (-2 * l) * res_prod

    return l_prod

def chi_k(x, k, sma):
    ksq = k ** 2
    res_chi = np.empty(x.shape)
    rho_x = rho_k(x, ksq, sma)
    res_sum = iter_sum(x, ksq, sma)
    res_chi = 2 * sma * rho_x * (1 + res_sum)

    return res_chi

