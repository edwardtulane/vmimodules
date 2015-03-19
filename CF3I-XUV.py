# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 04:55:35 2014

@author: felix
"""
import numpy as np
import scipy as sp
import vmiproc as vmp
import vmiclass as vmic
import pylab as pl

close('all')

VMIdir = '/home/brausse/vmi/'

#if 'ab' not in dir():
#    ab = np.load(commpath + 'storage/ab-250-18.npy', mmap_mode='r'
#                )
#                
#if 'bs' not in dir():
#    bs = np.load(commpath + 'storage/bs-250-18.npy' #, mmap_mode='r'
#                )
#                
#if 'FtF' not in dir():
#    FtF = np.load(commpath + 'storage/FtF-250-18.npy', mmap_mode='r'
#                )    
#
#


#broadband = {}
#broadb_ions = {}
#ser = np.arange(9,27+1), np.arange(7,33+1), np.arange(18, 29+1), np.arange(6,25+1)
#broadband[0] = vmic.VMIseries('2014-06-27',ser[0])
#broadband[1] = vmic.VMIseries('2014-06-30',ser[1])
#broadband[2] = vmic.VMIseries('2014-07-18',ser[2])
#broadband[3] = vmic.VMIseries('2014-07-21',ser[3])
##%%
#
#summo = {}
#
#for i in broadband[0].serPrist:
#    broadband[0].serPrist[i] = (broadband[0].serPrist[i] / np.sum(broadband[0].serPrist[i], dtype=float)) *6.2E9
#
#for i in broadband[1].serPrist:
#    broadband[1].serPrist[i] = (broadband[1].serPrist[i] / np.sum(broadband[1].serPrist[i], dtype=float)) *5.3E9
#
#for i in broadband[2].serPrist:
#    broadband[2].serPrist[i] = (broadband[2].serPrist[i] / np.sum(broadband[2].serPrist[i], dtype=float)) *5.3E9
#
#for i in broadband[3].serPrist:
#    broadband[3].serPrist[i] = (broadband[3].serPrist[i] / np.sum(broadband[3].serPrist[i], dtype=float)) *7.5E9
#
#
#
#sumbb = {}
#for i in broadband:
#    sumbb[i] = broadband[i].sumSeries()
#
##%%
#al1bb = vmic.TWimage(sumbb[0][0], xcntr=249 , ycntr=233, radius=170)
#al1bb.disp = [+1.2,0,0]
#al1bb.find_centre()
#an1bb = vmic.TWimage(sumbb[0][1], xcntr=249 , ycntr=233, radius=170)
#an1bb.disp = al1bb.disp
#an1bb.evalrect(501)
#
##%%
#al2bb = vmic.TWimage(sumbb[1][0], xcntr=249 , ycntr=233, radius=170)
#al2bb.disp = [+1.2,0,0]
#al2bb.find_centre()
#an2bb = vmic.TWimage(sumbb[1][1], xcntr=249 , ycntr=233, radius=170)
#an2bb.disp = al2bb.disp
#an2bb.evalrect(501)
#
##%%
#al3bb = vmic.TWimage(sumbb[2][0], xcntr=226 , ycntr=234, radius=210)
#al3bb.disp = [-2,0,0]
#al3bb.find_centre()
#an3bb = vmic.TWimage(sumbb[2][1], xcntr=226 , ycntr=234, radius=210)
#an3bb.disp = al3bb.disp
#an3bb.evalrect(501)
#
##%%
#
#al4bb = vmic.TWimage(sumbb[3][0], xcntr=227, ycntr=237, radius=210)
#al4bb.disp = [+0,0,0]
#al4bb.find_centre()
#an4bb = vmic.TWimage(sumbb[3][1], xcntr=227, ycntr=237, radius=210)
#an4bb.disp = al4bb.disp
#an4bb.evalrect(501)

##%%
#
#ab = np.load('storage/ab-200-16.npy', mmap_mode='r')
#bs = np.load('storage/bs-200-16.npy', mmap_mode='r')
#FtF = np.load('storage/FtF-200-16.npy')
#
##%%
#al1bb_img=vmp.halves(al1bb.rect)[1].ravel()
#al1bb_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(1800)), np.dot(ab,al1bb_img))
#
#an1bb_img=vmp.halves(an1bb.rect)[1].ravel()
#an1bb_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(1800)), np.dot(ab,an1bb_img))
#
#pl.plot(al1bb_inv - 0.82*an1bb_inv)
#
##%%
#al2bb_img=vmp.halves(al2bb.rect)[1].ravel()
#al2bb_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(1800)), np.dot(ab,al2bb_img))
#
#an2bb_img=vmp.halves(an2bb.rect)[1].ravel()
#an2bb_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(1800)), np.dot(ab,an2bb_img))
#
#pl.plot(al2bb_inv - 0.81*an2bb_inv)
#
##%%
#al3bb_img=vmp.halves(al3bb.rect)[1].ravel()
#al3bb_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(1800)), np.dot(ab,al3bb_img))
#
#an3bb_img=vmp.halves(an3bb.rect)[1].ravel()
#an3bb_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(1800)), np.dot(ab,an3bb_img))
#
#pl.plot(al3bb_inv - 0.90*an3bb_inv)
#
##%%
#al4bb_img=vmp.halves(al4bb.rect)[1].ravel()
#al4bb_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(1800)), np.dot(ab,al4bb_img))
#
#an4bb_img=vmp.halves(an4bb.rect)[1].ravel()
#an4bb_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(1800)), np.dot(ab,an4bb_img))
#
#pl.plot(al4bb_inv - 1.055*an4bb_inv)


###
###
###

bg1mo = vmic.TWimage(VMIdir+'terawatt/2014-07-25/2014-07-25-39.raw', xcntr=228, ycntr=235, radius=100)


bg2mo = vmic.TWimage(VMIdir+'terawatt/2014-07-28/2014-07-28-25.raw', xcntr=228, ycntr=235, radius=100)


bg3mo=vmic.TWimage(VMIdir+'terawatt/2014-07-28/2014-07-28-49.raw', xcntr=228, ycntr=235, radius=140)

##%%
#
monochrom = {}
ser2 = np.arange(29,38+1), np.arange(4,23+1), np.arange(26,47+1)
monochrom[0] = vmic.VMIseries('2014-07-25', ser2[0])
monochrom[1] = vmic.VMIseries('2014-07-28', ser2[1])
monochrom[2] = vmic.VMIseries('2014-07-28', ser2[2])

#%%
summo = {}

for i in monochrom[0].serPrist:
    monochrom[0].serPrist[i] = (monochrom[0].serPrist[i] / np.sum(monochrom[0].serPrist[i], dtype=float)) *2.8E8 - bg1mo.prist
for i in monochrom[1].serPrist:
    monochrom[1].serPrist[i] = (monochrom[1].serPrist[i] / np.sum(monochrom[1].serPrist[i], dtype=float)) *2.0E8 - bg2mo.prist
for i in monochrom[2].serPrist:
    monochrom[2].serPrist[i] = (monochrom[2].serPrist[i] / np.sum(monochrom[2].serPrist[i], dtype=float)) *1.0E9 - bg3mo.prist

for i in monochrom:
    summo[i] = monochrom[i].sumSeries()
mono_ions = {}
#
##%%
#
#%%

al1mo = vmic.TWimage(summo[0][0], xcntr=227, ycntr=235, radius=100)
al1mo.disp = [+0,0,0]
al1mo.find_centre()
an1mo = vmic.TWimage(summo[0][1], xcntr=227, ycntr=235, radius=100)
an1mo.disp = al1mo.disp
an1mo.evalrect(501)

#%%

al2mo = vmic.TWimage(summo[1][0], xcntr=228, ycntr=235, radius=100)
al2mo.disp = [+0,0,0]
al2mo.find_centre()
an2mo = vmic.TWimage(summo[1][1], xcntr=228, ycntr=235, radius=100)
an2mo.disp = al2mo.disp
an2mo.evalrect(501)


#%%

al3mo = vmic.TWimage(summo[2][0], xcntr=229, ycntr=235, radius=140)
al3mo.disp = [+0,0,0]
al3mo.find_centre()
an3mo = vmic.TWimage(summo[2][1], xcntr=229, ycntr=235, radius=140)
an3mo.disp = al3mo.disp
an3mo.evalrect(501)


#
##%%
#
#ab = np.load('storage/ab-200-16.npy', mmap_mode='r')
#FtF = np.load('storage/FtF-200-16.npy')
#
#%%
al1mo_img=vmp.fold(al1mo.rect, h=1, v=1).ravel()
al1mo_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(2500)), np.dot(ab,al1mo_img))

an1mo_img=vmp.fold(an1mo.rect, h=1, v=1).ravel()
an1mo_inv = np.dot(np.linalg.inv(FtF + 1 * np.eye(2500)), np.dot(ab,an1mo_img))

pl.plot(al1mo_inv - 1.00*an1mo_inv)


#%%
al2mo_img=vmp.fold(al2mo.rect, h=1, v=1).ravel()
al2mo_inv = np.dot(np.linalg.inv(FtF + 100 * np.eye(2500)), np.dot(ab,al2mo_img))

an2mo_img=vmp.fold(an2mo.rect, h=1, v=1).ravel()
an2mo_inv = np.dot(np.linalg.inv(FtF + 100 * np.eye(2500)), np.dot(ab,an2mo_img))

pl.plot(al2mo_inv - 1.025*an2mo_inv)


#%%
al3mo_img=vmp.fold(al3mo.rect, h=1, v=1).ravel()
al3mo_inv = np.dot(np.linalg.inv(FtF + 100 * np.eye(2500)), np.dot(ab,al3mo_img))

an3mo_img=vmp.fold(an3mo.rect, h=1, v=1).ravel()
an3mo_inv = np.dot(np.linalg.inv(FtF + 100 * np.eye(2500)),np.dot(ab,an3mo_img))

pl.plot(1.03*al3mo_inv-an3mo_inv)
#%%








##%%
#close('all')
##pinv=np.load('storage/pBinv.npy')
#inp, res = np.zeros((6,251,251)), np.zeros((6,1750))
#inp[0] = vmp.fold(sums[1][0],h=True,v=True)
#inp[1] = vmp.fold(sums[1][1],h=True,v=True)
#
#res[0] = np.dot(pinv,inp[0].ravel())
#res[1] = np.dot(pinv,inp[1].ravel())
#
#diff = -1*(res[0]- 1.028 * res[1])
#
#figure()
#plot(diff.reshape(7,250)[2,72:90].T / diff.reshape(7,250)[0,72:90].T)
#print sp.integrate.simps(diff.reshape(7,250)[2,72:90].T) / \
#sp.integrate.simps(diff.reshape(7,250)[0,72:90].T)
#
#print np.average(diff.reshape(7,250)[2,72:90].T / diff.reshape(7,250)[0,72:90].T)
#
#figure()
#plot(diff[:250] * np.arange(250)**2)
#
#
####
#
#inp[2] = vmp.fold(sums[2][0],h=True,v=True)
#inp[3] = vmp.fold(sums[2][1],h=True,v=True)
#
#res[2] = np.dot(pinv,inp[2].ravel())
#res[3] = np.dot(pinv,inp[3].ravel())
#
#
#diff = res[2]- 0.965 * res[3]
#
#figure()
#plot(diff.reshape(7,250)[2,104:121].T / diff.reshape(7,250)[0,104:121].T)
#print sp.integrate.simps(diff.reshape(7,250)[2,104:121].T) / \
#sp.integrate.simps(diff.reshape(7,250)[0,104:121].T)
#
#print np.average(diff.reshape(7,250)[2,104:121].T) / np.average(diff.reshape(7,250)[0,104:121].T)
#
#figure()
#plot(diff[:250] * np.arange(250)**2)
#
####
#
#inp[4] = vmp.fold(sums[0][0],h=True,v=True)
#inp[5] = vmp.fold(sums[0][1],h=True,v=True)
#
#res[4] = np.dot(pinv,inp[4].ravel())
#res[5] = np.dot(pinv,inp[5].ravel())
#
#diff = res[4]- 0.998 * res[5]
#
#figure()
#plot(diff.reshape(7,250)[2,72:90].T / diff.reshape(7,250)[0,72:90].T)
#print sp.integrate.simps(diff.reshape(7,250)[2,72:90].T) / \
#sp.integrate.simps(diff.reshape(7,250)[0,72:90].T)
#
#print np.average(diff.reshape(7,250)[2,72:90].T) / np.average(diff.reshape(7,250)[0,72:90].T)
#
#figure()
#plot(diff[:250] * np.arange(250)**2)
#
##bsx = [0,0]
##bsx[0] = vmp.Basex(sums[0][0],50,0,M1,M2)
##bsx[1] = vmp.Basex(sums[0][1],50,0,M1,M2)
##figure()
##imshow(bsx[0]- 0.96 * bsx[1])
##
