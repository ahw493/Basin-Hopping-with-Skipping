#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import warnings
import logging
import os
import time
import math

import scipy
from scipy import optimize
from scipy.optimize import basinhopping
from scipy.stats import gengamma,t
from math import *

def basinQ(x):
    if norm(optimal-x)<=0.001:
        return 1
    else:
        return 0
    
def monotonicityconstraintQ(x, function, threshold):
    # checks threshold value constraints
    return function(x) < threshold

def boundaryconstraintQ(x, boxsize):
    # checks only boundary constraints
    return bool(np.all(x <= boxsize)) and bool(np.all(x >= -boxsize))

def bothconstraintsQ(x, boxsize, function, threshold):
    # checks for both boundary and threshold value constraints
    return boundaryconstraintQ(x, boxsize) and monotonicityconstraintQ(x, function, threshold)

def bothconstraintsQindependent(x, boxsize, function, threshold):
    # checks for both boundary and threshold value constraints independently
    return boundaryconstraintQ(x, boxsize), monotonicityconstraintQ(x, function, threshold)

def checkcoordinates(x, boxsize):
    newx = x
    for i in range(x.shape[0]):
        if x[i] > boxsize:
            newx[i] = x[i] - 2*boxsize
        elif x[i] < -boxsize:
            newx[i] = x[i] + 2*boxsize
    return newx

class ClassicalTakeStep(object):
    
    def __init__(self, boxsize, periodicboundary=False, proposeoutsidebounds=False):
        #self.stepsize = stepsize
        self.boxsize = boxsize
        self.periodicboundary = periodicboundary
        self.proposeoutsidebounds = proposeoutsidebounds
        
    def __call__(self, x):
        sigma = aa#self.stepsize
        boxsize = self.boxsize
        periodicboundary = self.periodicboundary
        proposeoutsidebounds = self.proposeoutsidebounds
        global startingpoints
        global proposedpoints
        global emptymoves
        global iterations
        
        iterations +=1
        d = x.shape[0]
        
        phi = np.random.normal(loc=0.0, scale=sigma, size=d)
        phi = phi/np.linalg.norm(phi)
        beta = math.sqrt(2 * sigma)
        r = gengamma.rvs(a = d/2., c = 2., loc = 0., scale = beta, size = 1)
        xnew = x + phi*r[0]
        
        if proposeoutsidebounds or boundaryconstraintQ(xnew, boxsize):
            startingpoints.append(x.tolist())
            proposedpoints.append(xnew.tolist())
            return xnew
        elif periodicboundary:
            xnew = checkcoordinates(xnew, boxsize)
            startingpoints.append(x.tolist())
            proposedpoints.append(xnew.tolist())
            return xnew
        else:
            emptymoves += 1
            return x

class MonotonicSkippingTakeStep(object):
    
    def __init__(self, function, boxsize, maxjumps=50, periodicboundary=False, 
                 proposeoutsidebounds=False):
        self.function = function
        self.maxjumps = maxjumps
        #self.stepsize = 3
        self.boxsize = boxsize
        self.periodicboundary = periodicboundary
        self.proposeoutsidebounds = proposeoutsidebounds
        
    def __call__(self, x):
        sigma = aa#self.stepsize
        #print('sigma is:' ,sigma)
        maxjumps = self.maxjumps
        boxsize = self.boxsize
        periodicboundary = self.periodicboundary
        proposeoutsidebounds = self.proposeoutsidebounds
        function = self.function
        global sss
        global arraynfval
        global startingpoints
        global proposedpoints
        global emptymoves
        global iterations
        global nfval
        
        
        iterations+=1
        d = x.shape[0]
       # nfval = 1
        fval = function(x)
        
        phi = np.random.normal(loc=0.0, scale=sigma, size=d)
        phi = phi/np.linalg.norm(phi)
        beta = math.sqrt(2 * sigma)
        r = gengamma.rvs(a = d/2., c = 2., loc = 0., scale = beta, size = maxjumps)
        xnew = x + phi*r[0]
        j = 1
        
        if proposeoutsidebounds:
            xnew = checkcoordinates(xnew, boxsize)
            monotonicityflag = monotonicityconstraintQ(xnew, function, fval)
            nfval += 1
            while (j < maxjumps and not(monotonicityflag)):
                xnew += phi*r[j]
                j += 1
                monotonicityflag = monotonicityconstraintQ(xnew, function, fval)
                nfval += 1
            
            arraynfval.append(nfval)
            
            if j>1 and j < maxjumps and monotonicityflag:
                sss += 1
         #       print('BHMS skipped for',j,'steps and obtained',xnew,'at distance',np.linalg.norm(xnew-x),' current sigma=',sigma)
            
            if monotonicityflag:
                startingpoints.append(x.tolist())
                proposedpoints.append(xnew.tolist())
                return xnew
            else:
                emptymoves += 1
                return x
            
        elif periodicboundary:
            xnew = checkcoordinates(xnew, boxsize)
            monotonicityflag = monotonicityconstraintQ(xnew, function, fval)
            nfval += 1
            while (j < maxjumps and not(monotonicityflag)):
                xnew += phi*r[j]
                xnew = checkcoordinates(xnew, boxsize)
                j += 1
                monotonicityflag = monotonicityconstraintQ(xnew, function, fval)
                nfval += 1
                
            arraynfval.append(nfval)
            
            if j>1 and j < maxjumps and monotonicityflag:
                sss += 1
              #  print('BHMS skipped for',j,'steps and obtained',xnew,'at distance',np.linalg.norm(xnew-x),' current sigma=',sigma)
            
            if monotonicityflag:
                startingpoints.append(x.tolist())
                proposedpoints.append(xnew.tolist())
                return xnew
            else:
#                 print('Point',xnew,'with value',eggholder(xnew),' rejected as not monotone')
                emptymoves += 1
                return x
            
        else:
            boundaryflag, monotonicityflag = bothconstraintsQindependent(xnew, boxsize, function, fval)
            nfval += 1
            while (j < maxjumps and boundaryflag and not(monotonicityflag)):
                xnew += phi*r[j]
                j += 1
                boundaryflag, monotonicityflag = bothconstraintsQindependent(xnew, boxsize, function, fval)
                nfval += 1
                
            arraynfval.append(nfval)
        
            if j>1 and j < maxjumps and boundaryflag and monotonicityflag:
                sss += 1
              #  print('BHMS skipped for',j,'steps and obtained',xnew,'at distance',np.linalg.norm(xnew-x),' current sigma=',sigma)
            
            if boundaryflag and monotonicityflag:
                startingpoints.append(x.tolist())
                proposedpoints.append(xnew.tolist())
                return xnew
            else:
                emptymoves += 1
                return x
    
class MyBounds(object):
    
    def __init__(self, box):
        self.xmax = np.array(box)
        self.xmin = np.array(-box)
        
    def __call__(self, **kwargs):
      #  global iterations
        x = kwargs["x_new"]
       # iterations +=1
        return bool(np.all(x <= self.xmax)) and bool(np.all(x >= self.xmin))
    
def print_fun(x, f, accept):
#Capture intermediate states of optimization
    global xs
    xs.append(x.tolist())
#     print("point (%.3f,%.3f) at minimum %.3f accepted %d" % (x[0],x[1], f, int(accept)))

def interleave(arrays, axis=0, out=None):
    shape = list(np.asanyarray(arrays[0]).shape)
    if axis < 0:
        axis += len(shape)
    assert 0 <= axis < len(shape), "'axis' is out of bounds"
    if out is not None:
        out = out.reshape(shape[:axis+1] + [len(arrays)] + shape[axis+1:])
    shape[axis] = -1
    return np.stack(arrays, axis=axis+1, out=out).reshape(shape)

def show_doubletrajectory(startingpoints, proposedpoints, x, boxsize):
    margin = boxsize * 1.05
    sp = np.array(startingpoints)
    pp = np.array(proposedpoints)
    tj = interleave((sp,pp), axis=0)
    tj = np.vstack([tj,x])
    plt.figure()
    plt.xlim(-margin,margin)
    plt.ylim(-margin,margin)
    plt.plot(tj[:, 0], tj[:, 1], '-o', sp[:, 0], sp[:, 1], 's', x[0], x[1], 'r*') 
    plt.gca().set_aspect('equal', adjustable='box') 
    return plt.show()
    
def show_trajectory(trajectory, x, boxsize):
    margin = boxsize * 1.05
    tj = np.array(trajectory)
    tj = np.vstack([tj,x])
    plt.figure()
    plt.xlim(-margin,margin)
    plt.ylim(-margin,margin)
    plt.plot(tj[:, 0], tj[:, 1], '-o', x[0], x[1], 'r*') 
    plt.gca().set_aspect('equal', adjustable='box') 
    return plt.show()

class callbackiguess(object):
    #This will stop the basin hopping routine when the returned test is true
    def __init__(self):
        self.time_taken = 0

    def __call__(self,x,f,accept):
       # print(limit)
        self.time_taken = (time.time() - start)>limit
        
       # print(self.time_taken)
        
        return self.time_taken


# In[36]:


def schwefel(x):
    y = sum(-x*np.sin(np.sqrt(np.abs(x))))
    return y

print(schwefel(420.9687*np.array([1,1,1]))/3)


# In[43]:


boxsize = 500
seed = math.floor(time.time())

dimension_list = np.array(range(2,12,2))
x_initial = np.loadtxt('x0list_Sch_Var.dat')#np.random.uniform(low=-boxsize, high=boxsize, size=[200000,dimension_list[-1]])
variance = np.array([1,5.,10., 20., 30., 40., 50., 60.,  70.,  80.])

niter = 100000
maxjumps = 200
len_dim = len(dimension_list)
len_var = len(variance)
T = 1.0
method="L-BFGS-B"

limit_evals = 10000000 #maximum number of function evaluations
limit_time = 300
limit_iterations = 100000
same_place_iter = 50
callbackfunc = callbackiguess()

classicmytakestep = ClassicalTakeStep(boxsize=boxsize, periodicboundary=False, 
                                      proposeoutsidebounds=True)
skippingmytakestep = MonotonicSkippingTakeStep(function=schwefel, maxjumps=maxjumps,
                                               boxsize=boxsize, 
                                               periodicboundary=False, proposeoutsidebounds=True)


# In[44]:


#CONFIDENCE INTERVAL FUNCTIONS

 

#CLASSIC BASIN HOPPING

from numpy.linalg import norm 
import numpy.linalg as a 
import statistics 

T= 1

total_trials_c=np.ones([len_dim,len_var])
total_in_basin_c=np.ones([len_dim,len_var])
total_evaluations_c=np.ones([len_dim,len_var])
total_iterations_c=np.ones([len_dim,len_var])
total_time_c=np.ones([len_dim,len_var])

PIB_c=np.ones([len_dim,len_var])
EpS_c=np.ones([len_dim,len_var])
IpS_c=np.ones([len_dim,len_var])
TpS_c=np.ones([len_dim,len_var])

for z in range(len_dim):
    
    dimension = dimension_list[z]
    #print(dimension)
    x0 = x_initial[:,0:dimension]

    xmin = -boxsize*np.ones(dimension)#[-boxsize, -boxsize, -boxsize]
    xmax = boxsize*np.ones(dimension)#[boxsize, boxsize, boxsize]
    bounds = [(low, high) for low, high in zip(xmin, xmax)] #bounds for local minimizer (rewritten in the way required by L-BFGS-B, which can be used as the problem is smooth and bounded)
    minimizer_kwargs = dict(bounds=bounds, method=method)
    mybounds = MyBounds(np.array([boxsize]*dimension)) #bounds for accept_test
   # print(bounds)
    optimal = 420.9687*np.ones(dimension)
        
    for y in range(len_var):
               
        ft_minimum_pos_c = []
        ft_minimum_vals_c = []
        ft_minimums_dist_c = []
        ft_function_evals_c = []
        ft_basin_iterations_c = []
        ft_end_time_c = []
        ft_in_basin_c = []
        i= 1
        total_iterations = 0
        total_evaluations = 0
        iterations = 0
        duration = 0
        each_ctart = time.time()

        aa = variance[y]
               
        while duration<=limit_time:

            nfval = 0
            arraynfval = []
            xs = [x0[i].tolist()]
            proposedpoints = [x0[i].tolist()]
            startingpoints = [x0[i].tolist()]
            sss = 0
            emptymoves = 0
            #each_ctart = time.time()
            #print(x0[i])
            res3 = basinhopping(schwefel, x0[i], niter=niter, T=T, minimizer_kwargs=minimizer_kwargs, 
            take_step=classicmytakestep,accept_test=mybounds,niter_success = same_place_iter)#, callback=callbackfunc)
            
            duration = time.time()-each_ctart

            total_evaluations += res3.nfev
            total_iterations += res3.nit

            if duration<=limit_time:
                
                
                #print(i)
                ft_final_trials_c = i
                i +=1
                ft_end_time_c.append(duration)
                ft_minimum_pos_c.append(res3.x)
                ft_minimum_vals_c.append(res3.fun)
                ft_minimums_dist_c.append(norm(res3.x - optimal))
                ft_function_evals_c.append(res3.nfev+nfval)
                ft_basin_iterations_c.append(res3.nit)
                ft_in_basin_c.append(basinQ(res3.x))
                final_time_c = duration
                
        total_trials_c[z,y] = ft_final_trials_c
        total_in_basin_c[z,y] = sum(ft_in_basin_c)
        total_evaluations_c[z,y] = sum(ft_function_evals_c)
        total_iterations_c[z,y] = sum(ft_basin_iterations_c)
        total_time_c[z,y] = final_time_c
    
        PIB_c[z,y] = 100*sum(ft_in_basin_c)/i
        EpS_c[z,y] = np.array(sum(ft_function_evals_c))/sum(ft_in_basin_c) #evaluations_per_cuccess_c
        IpS_c[z,y] = np.array(sum(ft_basin_iterations_c))/sum(ft_in_basin_c) #iterations_per_cuccess_c
        TpS_c[z,y] = np.array(final_time_c)/sum(ft_in_basin_c)               #time_per_cuccess_c

print('percent in basin classic:',PIB_c)
print('confidence interval PIB',CI_PIB_c)
print('confidence interval evaluations',CI_EpS_c)
print('confidence interval iterations',CI_IpS_c)
print('confidence interval time',CI_TpS_c)

 
#SKIPPING BASIN HOPPING

from numpy.linalg import norm 
import numpy.linalg as a 
import statistics 

T= 0

total_trials_s=np.ones([len_dim,len_var])
total_in_basin_s=np.ones([len_dim,len_var])
total_evaluations_s=np.ones([len_dim,len_var])
total_iterations_s=np.ones([len_dim,len_var])
total_time_s=np.ones([len_dim,len_var])

PIB_s=np.ones([len_dim,len_var])
EpS_s=np.ones([len_dim,len_var])
IpS_s=np.ones([len_dim,len_var])
TpS_s=np.ones([len_dim,len_var])
 
for z in range(len_dim):
    
    dimension = dimension_list[z]
    #print(dimension)
    x0 = x_initial[:,0:dimension]

    xmin = -boxsize*np.ones(dimension)#[-boxsize, -boxsize, -boxsize]
    xmax = boxsize*np.ones(dimension)#[boxsize, boxsize, boxsize]
    bounds = [(low, high) for low, high in zip(xmin, xmax)] #bounds for local minimizer (rewritten in the way required by L-BFGS-B, which can be used as the problem is smooth and bounded)
    minimizer_kwargs = dict(bounds=bounds, method=method)
    mybounds = MyBounds(np.array([boxsize]*dimension)) #bounds for accept_test
   # print(bounds)
    optimal = 420.9687*np.ones(dimension)
        
    for y in range(len_var):
               
        ft_minimum_pos_s = []
        ft_minimum_vals_s = []
        ft_minimums_dist_s = []
        ft_function_evals_s = []
        ft_basin_iterations_s = []
        ft_end_time_s = []
        ft_in_basin_s = []
        i= 1
        total_iterations = 0
        total_evaluations = 0
        iterations = 0
        duration = 0
        each_start = time.time()

        aa = variance[y]
               
        while duration<=limit_time:

            nfval = 0
            arraynfval = []
            xs = [x0[i].tolist()]
            proposedpoints = [x0[i].tolist()]
            startingpoints = [x0[i].tolist()]
            sss = 0
            emptymoves = 0
            #each_start = time.time()
            #print(x0[i])
            res3 = basinhopping(schwefel, x0[i], niter=niter, T=T, minimizer_kwargs=minimizer_kwargs, 
            take_step=skippingmytakestep,accept_test=mybounds,niter_success = same_place_iter)#, callback=callbackfunc)
            
            duration = time.time()-each_start

            total_evaluations += res3.nfev
            total_iterations += res3.nit

            if duration<=limit_time:
                
                
                #print(i)
                ft_final_trials_s = i
                i +=1
                ft_end_time_s.append(duration)
                ft_minimum_pos_s.append(res3.x)
                ft_minimum_vals_s.append(res3.fun)
                ft_minimums_dist_s.append(norm(res3.x - optimal))
                ft_function_evals_s.append(res3.nfev+nfval)
                ft_basin_iterations_s.append(res3.nit)
                ft_in_basin_s.append(basinQ(res3.x))
                final_time_s = duration
                
        total_trials_s[z,y] = ft_final_trials_s
        total_in_basin_s[z,y] = sum(ft_in_basin_s)
        total_evaluations_s[z,y] = sum(ft_function_evals_s)
        total_iterations_s[z,y] = sum(ft_basin_iterations_s)
        total_time_s[z,y] = final_time_s
    
        PIB_s[z,y] = 100*sum(ft_in_basin_s)/i
        EpS_s[z,y] = np.array(sum(ft_function_evals_s))/sum(ft_in_basin_s) #evaluations_per_success_s
        IpS_s[z,y] = np.array(sum(ft_basin_iterations_s))/sum(ft_in_basin_s) #iterations_per_success_s
        TpS_s[z,y] = np.array(final_time_s)/sum(ft_in_basin_s)               #time_per_success_s

print('percent in basin classic:',PIB_s)
print('confidence interval PIB',CI_PIB_s)
print('confidence interval evaluations',CI_EpS_s)
print('confidence interval iterations',CI_IpS_s)
print('confidence interval time',CI_TpS_s)

np.savez('BH_Var_PIB_EpS_IpS_TpS_all',pc=PIB_c,pm=PIB_m, ps=PIB_s,
        ec=EpS_c,em=EpS_m,es=EpS_s,
        ic=IpS_c,im=IpS_m,iss=IpS_s,
        tc=TpS_c,tm=TpS_m,ts=TpS_s)
 

np.savez('BH_Var_Totals_Trials_InBas_Eval_Iter_Time_all',ttc=total_trials_c,ttm=total_trials_m, tts=total_trials_s,
         tbc=total_in_basin_c,tbm=total_in_basin_m,tbs=total_in_basin_s,
         tec=total_evaluations_c,tem=total_evaluations_m,tes=total_evaluations_s,
         tic = total_iterations_c,tim = total_iterations_m,tis = total_iterations_s,
         tc = total_time_c,tm = total_time_m,ts = total_time_s)

