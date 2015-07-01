# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 19:17:56 2014

@author: brausse
"""
def GCV(b, AT, lam):
    gcv = 0
    for i, aline in enumerate(AT):
        den = (aline.conj() * aline + lam)       
        for j, bline in enumerate(b):
            num = bline / den
            
            gcv += norm(num) ** 2 
        gcv /= norm(den) ** 2
        print(i)
    return gcv