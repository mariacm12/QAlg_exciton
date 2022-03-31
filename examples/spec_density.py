#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of autocorrelation function and dephasig time calculation

@author: mariacm
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import autocorrelation as ac

hb = 5308.8 
hbeV = 0.65821 #eV*fs
kbT = 8.617333*10**(-5)*300 #eV at 300K
ev_to_fs = 4.14
 

# Random energy gap data as example (in eV)
en_data = np.random.uniform(low=1.5, high=1.6, size=(50,))

dt = 4 #ps

npoints = len(en_data)
times1 = np.linspace(0,(npoints-1)*dt,npoints)

autocA = ac.correlation(en_data,en_data,norm=True)
autocA_norm = autocA/np.max(autocA)

skip1 = 5 #ignoring last few datapoints in autocorrelation
omegas = np.arange(0,2.5,0.01)

specA = [ac.spec_dens(autocA[:-skip1],times1[:-skip1-1])(om) for om in omegas]
maxA = max(specA)

# Reorganization energy
lam3A = ac.reorg_en(specA,omegas)
print(lam3A) 

# The resulting spectral density fitted to function described in the paper
def f_spec(K):
    tan= hbeV/2/kbT
    
    def f_spec2(w,a,b,g1,g2):
        w = w/ev_to_fs #convert to fs^-1
        const = 2 * K/hbeV/math.pi
        J_fn = const*np.tanh(tan*w)*(a*g1/(1 + g1**2 * w**2) + b*g2/(1 + g2**2 * w**2))
        return J_fn
    
    return f_spec2

from scipy.optimize import curve_fit
parsA, pcov = curve_fit(f_spec(maxA), omegas, specA, bounds=(0, [1., 1., 10, 10]))
spec_fitA = f_spec(maxA)(omegas,parsA[0],parsA[1],parsA[2],parsA[3])


#Plotting the spectral density and its fit
plt.plot(omegas,specA,color='blue',linewidth=2.0,alpha=0.8)
plt.plot(omegas,spec_fitA,color='red',linewidth=2.0,alpha=0.8)
plt.show()
