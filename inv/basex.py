# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 01:27:08 2014

@author: felix
"""

# basex basis set

def rho_k(r, ksq, sma):
#    ksq = k ** 2
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

