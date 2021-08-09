#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from numpy import *
from numpy.linalg import norm 
import numpy.linalg as a 

import statistics 


def basinQ(x):
    if norm(optimal-x)<=0.01:
        return 1
    else:
        return 0
    
def monotonicityconstraintQ(x, function, threshold):
    # checks threshold value constraints
    return function(x),function(x) < threshold

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

def check_smc_coordinates(x, boxsize):
    newx = x
    locate = abs(x)>boxsize
    sign = np.sign(x)
    newx[locate]= x[locate]-sign[locate]*2*boxsize
    return newx


# In[2]:


class ClassicalTakeStep(object):
    
    def __init__(self, boxsize, periodicboundary=False, proposeoutsidebounds=False):
        #self.stepsize = stepsize
        self.boxsize = boxsize
        self.periodicboundary = periodicboundary
        self.proposeoutsidebounds = proposeoutsidebounds
        
        
    def __call__(self, x):
        sigma = aa #self.stepsize
        boxsize = self.boxsize
        periodicboundary = self.periodicboundary
        proposeoutsidebounds = self.proposeoutsidebounds

        global iterations 
        
        distance = 0                
        iterations +=1
        d = x.shape[0]
        
        phi = np.random.normal(loc=0.0, scale=sigma, size=d)
        phi = phi/np.linalg.norm(phi)
        beta = math.sqrt(2*sigma)
        r = gengamma.rvs(a = d/2., c = 2., loc = 0., scale = beta, size = 1)
        xnew = x + phi*r[0]
        
        if proposeoutsidebounds or boundaryconstraintQ(xnew, boxsize):
            yy= xnew
        elif periodicboundary:
            xnew = checkcoordinates(xnew, boxsize)
            yy= xnew
        else:
            yy= x
            
        outside(x,yy,r,1)
        return yy
    
class callbackiguess(object):
    #This will stop the basin hopping routine when the returned test is true
    def __init__(self):
        self.time_taken = 0

    def __call__(self,x,f,accept):
        self.time_taken = (time.time() - start)>limit
        return self.time_taken
    
    
class MonotonicSkippingSMCTakeStep(object):
    
    def __init__(self, function, boxsize, maxjumps, NParticles,periodicboundary,proposeoutsidebounds):#,NParticles):
        self.function = function
        self.maxjumps = maxjumps
        self.NParticles = NParticles
        self.boxsize = boxsize
        self.periodicboundary = periodicboundary
        self.proposeoutsidebounds = proposeoutsidebounds
        
    def __call__(self, x):
        d= x.shape[0]
        sigma = (aa**2)*np.identity(d)
        maxjumps = self.maxjumps
        boxsize = self.boxsize
        periodicboundary = self.periodicboundary
        proposeoutsidebounds = self.proposeoutsidebounds
        function = self.function
        NParticles = self.NParticles
        
        global iterations,nfval,skips 
        distance = zeros([NParticles])
        displacement = 0
        iterations+=1
  
        fval = function(array(x))
        nfval+=1
        
        phi = np.random.multivariate_normal(mean=np.zeros(d), cov=sigma, size=NParticles)
        phi = phi/np.linalg.norm(phi)
        chisqr = np.random.chisquare(d,size=NParticles)
        r=np.sum((phi@np.linalg.inv(sigma))*phi,1)**(-0.5)*np.sqrt(chisqr)
        distance += r
        move =(r*phi.T).T
        new_xs = x + move
        particle_steps = np.ones(NParticles)
        
        
        if proposeoutsidebounds:
            new_xs = check_smc_coordinates(new_xs, boxsize)
            func_results,monotonicityflag = monotonicityconstraintQ(new_xs, function, fval)
            nfval += len(new_xs)
            active_particle_index = [True]
            
            while any(active_particle_index): 
                
                active_particle_index = (particle_steps.T<maxjumps)*(~monotonicityflag)
                r=np.sum((phi[active_particle_index,:]@np.linalg.inv(sigma))*
                         phi[active_particle_index,:],1)**(-0.5)*np.sqrt(chisqr[active_particle_index])
                move =(r*phi[active_particle_index,:].T).T
                new_xs[active_particle_index] += move
                distance[active_particle_index] += r
                particle_steps[active_particle_index] +=1
                func_results[active_particle_index],monotonicityflag[active_particle_index] = monotonicityconstraintQ(new_xs[active_particle_index], function, fval)
                nfval += len(new_xs[active_particle_index])
                
            if any(monotonicityflag):
               
                xnew = new_xs[func_results==min(func_results[monotonicityflag]),:]
                yy = xnew 
                ll=func_results[monotonicityflag]
                winning_chain = where(func_results==min(ll))
                skips = particle_steps[winning_chain]
                displacement = distance[winning_chain]

            else:
                yy = x
 
        elif periodicboundary:
            new_xs = check_smc_coordinates(new_xs, boxsize)
            func_results,monotonicityflag = monotonicityconstraintQ(new_xs, function, fval)
            nfval += len(new_xs)
            active_particle_index = [True]
            
            while any(active_particle_index): 
                active_particle_index = (particle_steps.T<maxjumps)*(~monotonicityflag)
                r=np.sum((phi[active_particle_index,:]@np.linalg.inv(sigma))*
                         phi[active_particle_index,:],1)**(-0.5)*np.sqrt(chisqr[active_particle_index])
                move =(r*phi[active_particle_index,:].T).T
                distance[active_particle_index]+=r
                new_xs[active_particle_index] += move
                new_xs[active_particle_index] = check_smc_coordinates(new_xs[active_particle_index], boxsize)
                particle_steps[active_particle_index] +=1
                func_results[active_particle_index],monotonicityflag[active_particle_index] = monotonicityconstraintQ(new_xs[active_particle_index], function, fval)

                nfval += len(func_results[active_particle_index])
 
            if any(monotonicityflag): #THESE ARE SUCCESSFUL JUMPS, BOTH SKIPS AND RANDOM WALKS
                
                ll=func_results[monotonicityflag]
                xnew =  new_xs[func_results==min(ll),:]
                winning_chain = where(func_results==min(ll))
                yy = xnew
                skips = particle_steps[winning_chain]
                displacement = distance[winning_chain]
            else:
                yy= x  
        outside(x,yy,displacement,particle_steps)
        return yy
     
class MyBounds(object):
    
    def __init__(self, box):
        self.xmax = np.array(box)
        self.xmin = np.array(-box)
        
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        return bool(np.all(x <= self.xmax)) and bool(np.all(x >= self.xmin))
    
def outside(x,y,r,all_skips):
    global key,avg_all_skips
    key = r
    avg_all_skips.append(mean(all_skips))
    #print('key',key,'|y-x|',norm(x-y),'avg_skips',avg_all_skips)
    
class MJSD(object):
    
    def __init__(self):
        self.a = 1
        
    def __call__(self,x,f,accept):
        global key,rw_dist,skips,skip_dist,moves,list_succ_skips,proposals
        proposals +=1
        #THIS CALCULATES THE NUMBER OF RW MOVES AND NUMBER OF SKIPPING (K>1 AND ACCEPT) MOVES
        if accept==True and key>0:
            moves+=1
            
            if skips==1:
                rw_dist=append(rw_dist,key)
            elif skips>1:
                skip_dist = append(skip_dist,key)
                list_succ_skips.append(skips)
                #print(list_succ_skips)


# In[3]:


def eggholder(x):
    if len(x.shape) ==1:
        return -(x[1] + 47.)* np.sin(np.sqrt(np.abs(x[1] + x[0]/2. + 47.))) - x[0]*np.sin(np.sqrt(np.abs(x[0] - x[1] - 47.)))
    else:
        return -(x[:,1] + 47.)* np.sin(np.sqrt(np.abs(x[:,1] + x[:,0]/2. + 47.))) - x[:,0]*np.sin(np.sqrt(np.abs(x[:,0] - x[:,1] - 47.)))  


# In[6]:


optimal = array([512,404.23180501])
boxsize = 512
seed = math.floor(time.time())
key = 0
moves = 0
rw_dist = []
skip_dist = []
list_succ_skips=[]
avg_all_skips = []
chosen_distance = 0
mjsd = MJSD()
dimension_list = np.array([2])
x_initial = np.random.uniform(low=-boxsize, high=boxsize, size=[20000,dimension_list[-1]])

jump_list = np.array([ 1,5,10,20,40,60,80,100,120,140,160,180,200])
variance = np.array([0.1,1,10,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300]) 
niter = 100000
len_dim = len(dimension_list)
len_var = len(variance)
len_jump = len(jump_list)
method ="L-BFGS-B"

limit_evals = 10000000 #maximum number of function evaluations
limit_time = 300
limit_iterations = 100000
same_place_iter = 50

classicmytakestep = ClassicalTakeStep(boxsize=boxsize, periodicboundary=False, 
                                      proposeoutsidebounds=True)


# In[7]:


#CONFIDENCE INTERVAL FUNCTIONS

def conf_int_PIB(pib,i):
    uCI = pib+1.96*np.sqrt(pib*(1-pib/100)/i)
    lCI = pib-1.96*np.sqrt(pib*(1-pib/100)/i)
    return ([lCI,uCI])

def t_con_int(mu,input_list,n):
    t_value = t.ppf(0.5+0.95/2, df=n-1, loc=0, scale=1)
    sam_sig = sample_error(n,input_list)
    CI = [mu-t_value*sam_sig/sqrt(n),mu+t_value*sam_sig/sqrt(n)]
    return CI

def sample_error(N_succ,input_list):
    return np.sqrt(np.sum((input_list-np.mean(input_list))**2)/(N_succ-1))


# In[8]:


#CLASSIC BASIN HOPPIN
#global dist,rw_dist,skips_succ_chains
T= 1

total_trials_c=np.ones([len_dim,len_var])
total_in_basin_c=np.ones([len_dim,len_var])
total_evaluations_c=np.ones([len_dim,len_var])
total_iterations_c=np.ones([len_dim,len_var])
total_time_c=np.ones([len_dim,len_var])

exp_EMJD_rw_c= np.ones([len_dim,len_var])
PSrw_c= np.ones([len_dim,len_var])
accept_rate_c= np.ones([len_dim,len_var])

PIB_c=np.ones([len_dim,len_var])
EpS_c=np.ones([len_dim,len_var])
IpS_c=np.ones([len_dim,len_var])
TpS_c=np.ones([len_dim,len_var])

CI_PIB_c=np.ones([len_dim,len_var,2])
CI_EpS_c=np.ones([len_dim,len_var,2]) 
CI_IpS_c=np.ones([len_dim,len_var,2]) 
CI_TpS_c=np.ones([len_dim,len_var,2]) 

for z in range(len_dim):
    
    dimension = dimension_list[z]
    x0 = x_initial[:,0:dimension]

    xmin = -boxsize*np.ones(dimension)#[-boxsize, -boxsize, -boxsize]
    xmax = boxsize*np.ones(dimension)#[boxsize, boxsize, boxsize]
    bounds = [(low, high) for low, high in zip(xmin, xmax)] #bounds for local minimizer (rewritten in the way required by L-BFGS-B, which can be used as the problem is smooth and bounded)
    minimizer_kwargs = dict(bounds=bounds, method=method,tol=1e-6)
    mybounds = MyBounds(np.array([boxsize]*dimension)) #bounds for accept_test
#    optimal = array([-0.9,-.95])#1*np.ones(dimension)
        
    for y in range(len_var):
               
        ft_cinimum_pos_c = []
        ft_cinimum_vals_c = []
        ft_cinimums_dist_c = []
        ft_function_evals_c = []
        ft_basin_iterations_c = []
        ft_end_time_c = []
        ft_in_basin_c = []
        list_succ_skips=[]
        avg_all_skips = []
        i= 1
        total_iterations = 0
        total_evaluations = 0
        iterations = 0
        duration = 0
        each_ctart = time.time()

        aa = variance[y]
        
        duration = 0
        evaluations = 0
        skips = 1 #set this equal to 1 to properly record RW               
        moves = 0
        rw_dist = []
        skip_dist = [] 
        proposals = 0
        while duration<=limit_time:

            nfval = 0           
            res3 = basinhopping(eggholder, x0[i], niter=niter, T=T, minimizer_kwargs=minimizer_kwargs, 
            take_step=classicmytakestep,accept_test=mybounds,niter_success = same_place_iter, callback=mjsd)
            
            duration = time.time()-each_ctart

            total_evaluations += res3.nfev
            total_iterations += res3.nit

            if duration<=limit_time:
                
                ft_final_trials_c = i
                i +=1
                ft_cinimum_pos_c.append(res3.x)
                ft_cinimum_vals_c.append(res3.fun)
                ft_cinimums_dist_c.append(norm(res3.x - optimal))
                ft_function_evals_c.append(res3.nfev+nfval)
                ft_basin_iterations_c.append(res3.nit)
                ft_in_basin_c.append(basinQ(res3.x))
                final_time_c = duration
                recorded_moves = moves; recorded_proposals = proposals

        exp_EMJD_rw_c[z,y] = nanmean(rw_dist)
        total_trials_c[z,y] = ft_final_trials_c
        total_in_basin_c[z,y] = sum(ft_in_basin_c)
        total_evaluations_c[z,y] = sum(ft_function_evals_c)
        total_iterations_c[z,y] = sum(ft_basin_iterations_c)
        total_time_c[z,y] = final_time_c
    
        accept_rate_c[z,y] = 100*recorded_moves/recorded_proposals #perecnt of successful proposals
        PSrw_c[z,y]= 100*len(rw_dist)/array(moves) #percent successful RW steps given that a move was accepted
        PIB_c[z,y] = 100*sum(ft_in_basin_c)/i
        EpS_c[z,y] = np.array(sum(ft_function_evals_c))/sum(ft_in_basin_c) #evaluations_per_cuccess_c
        IpS_c[z,y] = np.array(sum(ft_basin_iterations_c))/sum(ft_in_basin_c) #iterations_per_cuccess_c
        TpS_c[z,y] = np.array(final_time_c)/sum(ft_in_basin_c)               #time_per_cuccess_c

        CI_PIB_c[z,y,:] = conf_int_PIB(PIB_c[z,y],i)
        CI_EpS_c[z,y,:] = t_con_int(EpS_c[z,y],ft_function_evals_c,total_in_basin_c[z,y])
        CI_IpS_c[z,y,:] = t_con_int(IpS_c[z,y],ft_basin_iterations_c,total_in_basin_c[z,y])
        CI_TpS_c[z,y,:] = t_con_int(TpS_c[z,y],final_time_c,total_in_basin_c[z,y])

print('percent in basin classic:',PIB_c)
print('EpS:',EpS_c)
print('TpS:',TpS_c)
print('EMJD_RW',exp_EMJD_rw_c)
print('PSrw',PSrw_c)
print('Acceptance Rate',accept_rate_c)


# In[9]:





# In[4]:


#SKIPPING SAMPLER WITH N PARTICLES
#global dist, rw_dist, skip_dist, skips_succ_chains

NParticle_list = array([1])
len_part = len(NParticle_list)


exp_EMJD_rw_s   = np.ones([len_dim,len_part,len_var,len_jump])
exp_EMJD_skip_s = np.ones([len_dim,len_part,len_var,len_jump])

total_trials_s=  np.ones([len_dim,len_part,len_var,len_jump])
total_in_basin_s=np.ones([len_dim,len_part,len_var,len_jump])
total_evaluations_s=np.ones([len_dim,len_part,len_var,len_jump])
total_iterations_s= np.ones([len_dim,len_part,len_var,len_jump])
total_time_s= np.ones([len_dim,len_part,len_var,len_jump])

PSS_s =np.ones([len_dim,len_part,len_var,len_jump])
PSrw_s = np.ones([len_dim,len_part,len_var,len_jump])
accept_rate_s = np.ones([len_dim,len_part,len_var,len_jump])
mean_all_skips= np.ones([len_dim,len_part,len_var,len_jump])
mean_succ_skips = np.ones([len_dim,len_part,len_var,len_jump])
PIB_s=np.ones([len_dim,len_part,len_var,len_jump])
EpS_s=np.ones([len_dim,len_part,len_var,len_jump])
IpS_s=np.ones([len_dim,len_part,len_var,len_jump])
TpS_s=np.ones([len_dim,len_part,len_var,len_jump])

CI_PIB_s=np.ones([len_dim,len_part,len_var,len_jump,2])
CI_EpS_s=np.ones([len_dim,len_part,len_var,len_jump,2])
CI_IpS_s=np.ones([len_dim,len_part,len_var,len_jump,2])
CI_TpS_s=np.ones([len_dim,len_part,len_var,len_jump,2])

T= 0

for z in range(len_dim):

    dimension = dimension_list[z]
    x0 = x_initial[:,0:dimension]

    xmin = -boxsize*np.ones(dimension)
    xmax = boxsize*np.ones(dimension) 
    bounds = [(low, high) for low, high in zip(xmin, xmax)] #bounds for local minimizer (rewritten in the way required by L-BFGS-B, which can be used as the problem is smooth and bounded)
    minimizer_kwargs = dict(bounds=bounds, method=method,tol=1e-6)
    mybounds = MyBounds(np.array([boxsize]*dimension)) #bounds for accept_test
  #  optimal = array([-0.9,-.95])#1*np.ones(dimension)
    
    for y in range(len_part):
        
        NPart = NParticle_list[y]
        print(NPart)
        for xx in range(len_var):
            aa = variance[xx]
            
            for yy in range(len_jump):
            #initializations
      
                ft_cinimum_pos_s = []
                ft_cinimum_vals_s = []
                ft_cinimums_dist_s = []
                ft_function_evals_s = []
                ft_basin_iterations_s = []
                ft_end_time_s = []
                ft_in_basin_s = []
                final_time_s=0
                trial_cJD_rw =[]
                trial_cJD_skip = []
                list_succ_skips=[]
                avg_all_skips = []
                i= 0
                total_evaluations = 0
                iterations = 0
                start = time.time()

                max_jump=jump_list[yy]


                smcskippingmytakestep = MonotonicSkippingSMCTakeStep(function=eggholder, maxjumps=max_jump,
                                                                       boxsize=boxsize,NParticles=NPart, 
                                                                       periodicboundary=True,proposeoutsidebounds=False)
                duration = 0
                evaluations = 0
                skips = 0               
                moves = 0
                rw_dist = []
                skip_dist = [] 
                proposals = 0
    #THIS WHILE LOOP INITIATES THE EXPERIMENT

                while duration<=limit_time:

                    nfval = 0
                    each_start = time.time()

                    res3 = basinhopping(eggholder,x0[i], niter=niter, T=T, minimizer_kwargs=minimizer_kwargs, 
                                        take_step=smcskippingmytakestep,accept_test=mybounds,
                                        niter_success = same_place_iter, callback=mjsd)

                   # print('\n next thing \n')
                    evaluations = res3.nfev+nfval
                    duration = time.time() - start

                    #THIS RECORDS THE RESULT OF EACH TRIAL
                    if duration<=limit_time:

                        i +=1
                        ft_final_trials_s = i

                #MANY PROPOSALS AND ACCEPTANCES
    #TRIAL- SINGLE MINIMUM RESULT
    #N TRIALS
    #EXPERIMENT- N RESULTS

                        ft_end_time_s.append(duration)
                        ft_cinimum_pos_s.append(res3.x)
                        ft_cinimum_vals_s.append(res3.fun)
                        ft_cinimums_dist_s.append(norm(res3.x - optimal))
                        ft_function_evals_s.append(res3.nfev+nfval)
                        ft_basin_iterations_s.append(res3.nit)
                        ft_in_basin_s.append(basinQ(res3.x))
                        final_time_s = duration
                        recorded_moves = moves; recorded_proposals = proposals
                  #      print('BH iterations',res3.nit)
                #print('*************************num of rw',len(rw_dist))
                #print('*************************num of skip',len(skip_dist))
                #print('*************************moves',moves)
            
                exp_EMJD_rw_s[z,y,xx,yy] = nanmean(rw_dist)
                exp_EMJD_skip_s[z,y,xx,yy] = nanmean(skip_dist)

                total_trials_s[z,y,xx,yy] = i
                total_in_basin_s[z,y,xx,yy] = sum(ft_in_basin_s)
                total_evaluations_s[z,y,xx,yy] = sum(ft_function_evals_s)
                total_iterations_s[z,y,xx,yy] = sum(ft_basin_iterations_s)
                total_time_s[z,y,xx,yy] = final_time_s

                mean_all_skips[z,y,xx,yy] = mean(avg_all_skips)
                mean_succ_skips[z,y,xx,yy] = mean(list_succ_skips)
                accept_rate_s[z,y,xx,yy] = 100*recorded_moves/recorded_proposals #perecnt of successful proposals
                PSS_s[z,y,xx,yy] = 100*len(skip_dist)/array(moves)#percent successful skips given that a move was accepted
                PSrw_s[z,y,xx,yy]= 100*len(rw_dist)/array(moves) #percent successful RW steps given that a move was accepted
                PIB_s[z,y,xx,yy] = 100*sum(ft_in_basin_s)/i
                EpS_s[z,y,xx,yy] = sum(ft_function_evals_s)/sum(ft_in_basin_s) #evaluations_per_success_s
                IpS_s[z,y,xx,yy] = sum(ft_basin_iterations_s)/sum(ft_in_basin_s) #iterations_per_success_s
                TpS_s[z,y,xx,yy] = final_time_s/sum(ft_in_basin_s)               #time_per_success_s

                CI_PIB_s[z,y,xx,yy,:] = conf_int_PIB(PIB_s[z,y,xx,yy],i) 
                CI_EpS_s[z,y,xx,yy,:] = t_con_int(EpS_s[z,y,xx,yy],ft_function_evals_s,total_in_basin_s[z,y,xx,yy])
                CI_IpS_s[z,y,xx,yy,:] = t_con_int(IpS_s[z,y,xx,yy],ft_basin_iterations_s,total_in_basin_s[z,y,xx,yy])
                CI_TpS_s[z,y,xx,yy,:]  = t_con_int(TpS_s[z,y,xx,yy],final_time_s,total_in_basin_s[z,y,xx,yy])

print('percent in basin skipping:',PIB_s)
print('EpS:',EpS_s)
print('TpS:',TpS_s)
print('EMJD_rw',exp_EMJD_rw_s)
print('EMJD_skip',exp_EMJD_skip_s)
print('PSSkip',PSS_s)
print('PSrw',PSrw_s)
print('Acceptance Rate',accept_rate_s)
print('mean all skips',mean_all_skips)
print('mean successful skips',mean_succ_skips)


# In[11]:







# In[ ]:



np.savez('BH_Var_PIB_EpS_IpS_TpS_all',pc=PIB_c,ps=PIB_s,
        ec=EpS_c,es=EpS_s,
        ic=IpS_c,iss=IpS_s,
        tc=TpS_c,ts=TpS_s)
np.savez('BH_Var_CI_PIB_EpS_IpS_TpS_all',pc=CI_PIB_c,pm=CI_PIB_c, ps=CI_PIB_s,
        ec=CI_EpS_c,es=CI_EpS_s,
        ic=CI_IpS_c,iss=CI_IpS_s,
        tc=CI_TpS_c,ts=CI_TpS_s)

np.savez('BH_Var_Totals_Trials_InBas_Eval_Iter_Time_all',ttc=total_trials_c, tts=total_trials_s,
         tbc=total_in_basin_c,tbs=total_in_basin_s,
         tec=total_evaluations_c,tes=total_evaluations_s,
         tic = total_iterations_c,tis = total_iterations_s,
         tc = total_time_c,ts = total_time_s)

np.savez('Skipping_vs_RW_Accept_rate',rwdc=exp_EMJD_rw_c,rwds=exp_EMJD_rw_s,
         skps=exp_EMJD_skip_s ,pss=PSS_s,prw=PSrw_s,arc=accept_rate_c,ars=accept_rate_s,
         mas = mean_all_skips,mss = mean_succ_skips)


# In[ ]:


# In[4]:


PIB_s


# In[ ]:




