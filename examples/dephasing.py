#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of autocorrelation function and dephasing time calculation

@author: mariacm
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import autocorrelation as ac
 
# Random energy gap data as example (in eV)
en_data = np.random.uniform(low=1.5, high=1.6, size=(50,))

dt = 4 #ps

npoints = len(en_data)
times1 = np.linspace(0,(npoints-1)*dt,npoints)

autocA = ac.correlation(en_data,en_data,norm=True)
autocA_norm = autocA/np.max(autocA)

deph_dat1 = [ac.deph_func(autocA,times1)(t) for t in range(1,npoints)]
skip1 = 5 #ignoring last few datapoints in autocorrelation
depht3A = ac.deph_time(deph_dat1[:-skip1],times1[:-skip1])

#Plot results

colormap = 'viridis'
font_size = 28
font_family = 'helvetica'  

font = {'family': font_family, 'size': font_size}
mpl.rc('font', **font)   

#data
fig, ax1 = plt.subplots(figsize=(8,4))
ax1.set_xlabel(r'$t (fs)$')
ax1.set_xlim(0,times1[-1])
ax1.plot(times1,en_data,color='blue',linewidth=2.5,alpha=0.8)
plt.show()

##autocorrelation
fig, ax1 = plt.subplots(figsize=(8,4))
ax1.set_ylim(-1,1)
ax1.set_xlim(0,times1[-skip1-1])
ax1.set_xlabel(r'$t (fs)$')
ax1.plot(times1[:-1],autocA_norm,color='red',linewidth=2.0,alpha=0.8)
plt.show()

##Dephasing function
fig, ax2 = plt.subplots(figsize=(6,3.5))
ax2.set_xlabel(r'$t (fs)$')
ax2.set_ylabel(r'$D(t)$')
ax2.set_xlim(0,times1[-skip1-1])
ax2.plot(times1[:-skip1-1],deph_dat1[:-skip1],color='green',linewidth=2.0,alpha=0.8)

plt.show()
