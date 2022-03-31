#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For calculating the dephasng time from QM-MM data

Created on Wed May 27 13:47:04 2020

@author: mariacm
"""

import numpy as np
from numpy import linalg as LA
import scipy.linalg
import math
from math import sqrt,pi

hb = 5308.8 
hbeV = 0.65821 #eV*fs
kbT = 8.617333*10**(-5)*300 #eV at 300K
ev_conv = 27.2113961 
ev_to_fs = 4.14


# =============================================================================
# Autocorrelation function
# =============================================================================
def avg_correlation(datas):
    
    N_T = len(datas[0])
    def corr_i(k,data):
        tot = 0
        it = 0
        Eavg = np.average(data,axis=0)
        for i in range(N_T-k):
            tot += (data[int(i)]-Eavg)*(data[i+k]-Eavg)
            it += 1
        if it == 0: it=1
        return tot/it,it   
    
    ac = []
    for k in range(1,N_T):
        cor_acum = 0
        for j in range(4):
            cor_acum += corr_i(k,datas[j])[0]
        ac.append(cor_acum/4)
    return np.array(ac)

def correlation(data1,data2,norm=False):
    """
    Calculates correlation function between A and B, with data given by data1 and data2

    Parameters
    ----------
    data1, data2 : Numpy arrays of the same length, N.
    norm : Set to True for calculating the correation function across a trajectory.
           If false will calculate average correlation. 

    Returns
    -------
    Numpy array. Correlation function with length N-1

    """
    N_T = len(data1)
    if not norm:
        def corr_k(lag):
            avg1 = np.average(data1,axis=0)
            avg2 = np.average(data2,axis=0)
            nk = data1[:N_T-lag]
            nkp1 = data2[lag:]
            num = np.dot(nk-avg1,nkp1-avg2)
            res = num
            return res,0
    else:
        def corr_i(k):
            tot = 0
            it = 0
            Eavg = np.average(data1,axis=0)
            for i in range(N_T-k):
                tot += (data1[int(i)]-Eavg)*(data2[i+k]-Eavg)
                it += 1
            if it == 0: it=1
            return tot/it,it   
    
    ac = []
    for k in range(1,N_T):
        cor = corr_i(k)
        ac.append(cor[0])
    return np.array(ac)

# =============================================================================
# Dephasing time calculation
# =============================================================================


def deph_func(autocorr,tim):
    """
    

    Parameters
    ----------
    autocorr : Numpy array.
        Autocorrelation function
    tim : Numpy array
        Time array

    Returns
    -------
    dephasing as a function of t

    """
    
    def integ(ti): #integration
        t = tim[ti]
        timesi = tim[:ti]
        autocorri = autocorr[:ti]
        val = np.multiply(autocorri,(t-timesi))
        result = np.exp(-(1/(hbeV**2))*abs(scipy.integrate.simps(val,timesi,even='last')))

        return result
    
    return integ
    
def deph_time(deph_data,tim):
    """
    Dephasing time from dephasing function

    Parameters
    ----------
    deph_data : Numpy array
        Dephasing function from time array.
    tim : Numpy array
        Time array.

    Returns
    -------
    Dephasing time

    """
    integ = scipy.integrate.simps(deph_data,tim[1:],even='last')
    return 2/sqrt(pi)*integ

# =============================================================================
# Spectral density calculation
# =============================================================================

def spec_dens(autocorr,tim):
    """
    Cacluates spectral density from autocorrelation array

    Parameters
    ----------
    autocorr : Ndarray
        Autocorrelation function.
    tim : Ndarray
        time array.

    Returns
    -------
    Numpy array. Spectral density.

    """
    
    def integ(omega):
        t = tim
        dt = t[1]-t[0]
        val = np.multiply(autocorr,np.cos(omega/ev_to_fs*t))
        res = abs(scipy.integrate.simps(val,t,dt,even='last')) 
        return (omega/ev_to_fs)/kbT/math.pi * res
    
    return integ

def reorg_en(data,omega):
    """
    Calculates reorganization energy
    """
    val = np.multiply(data[1:],1/omega[1:])
    return scipy.integrate.simps(val,omega[1:])



