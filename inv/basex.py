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
    k_sq = k_sq[k_sq > 0.0, None]
    u = r / sigma
    u_sq = u ** 2
    exponent = np.zeros((len(k), len(r)))
    exponent[:] =  -1 * u_sq
    exponent[k.ravel() > 0] += k_sq + k_sq * np.log(u_sq / k_sq)
    return np.exp(exponent)

def X(r, k, sigma):
    k_sq = np.float(k ** 2)
    u = r / sigma

    ll = np.arange(0, k ** 2 + 1)
    print 'Setting up the coefficients'
    gam = gamma_approx(ll)
    alph = aleph(ll.astype(np.float_))
    R_mat = np.zeros((k_sq, r.shape[0]))
    X_mat = np.zeros((k + 1, r.shape[0]))
    k_vec = np.arange(k + 1)
    R_mat =  R_k(r, np.sqrt(ll), sigma)
#   X_mat /= gam[::-1, None]
#   return np.sum(X_mat * alph[::-1, None], axis=0) * 2 * gam[-1]
#   frac = alph[::-1] / gam
#   frac = frac[:, None]
    print 'Filling the Abel-transformed matrix'
    for i, k2 in enumerate(k_vec**2):
        X_mat[i] = np.sum(alph[k2::-1, None] * R_mat[:k2+1] / gam[:k2+1, None], axis=0)
        X_mat[i] *= 2 * gam[k2]
        print('%i -' % i)

    return X_mat

def gamma(ll):
#    ll = np.arange(0, l + 1) ** 2
    return (np.e / ll) ** ll * fact(ll)

def gamma_approx(ll):
    coeffs = np.array([1, 1/12., 1/288., -139/51840., -571/2488320.])
    exps = np.arange(5) * -1
    ll =  np.sqrt(2 * np.pi * ll) * (coeffs[:, None] * (ll ** exps[:, None])).sum(0)
    ll[0] = 1
    return ll

def aleph(m):
    ll_a = np.ones(m.shape[0])
#    m = np.arange(1, l + 1).astype(np.float) ** 2
    ll_a[1::] = ll_a[1::] - (1 / (2 * m[1::]))
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
    k_vec = np.arange(ksq) + 1
    k_vec = k_vec[:, None]
    res_prod = np.ones(k_vec.shape)
    res_prod += ksq - k_vec
    m = k_vec - 0.5
    res_prod = (res_prod * m) / k_vec
    return np.cumproduct(res_prod.ravel())

def iter_sum(x, ksq, sma):
    try: type(ksq) == int
    except: TypeError('k must be an int')
#    res_sum = np.ones(ksq, ksq, x.shape[0])

    k_vec = np.arange(ksq) + 1
    chi = np.zeros((ksq, len(x)))
#    res_prod = np.empty([k_vec.shape[0], x.shape[0]])
#    for i, l in enumerate(k_vec):
    l_prod = iter_prod(ksq)[:, None]
    chi = (x / sma) ** (-2 * k_vec[:, None])

    return chi, l_prod

def chi_k(x, k, sma):
    ksq = k ** 2
    res_chi = np.empty(x.shape)
    rho_x = rho_k(x, ksq, sma)
    res_sum = iter_sum(x, ksq, sma)
    res_chi = 2 * sma * rho_x * (1 + res_sum)

    return res_chi

