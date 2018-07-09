#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as alg

def TVGL(data, lengthOfSlice, lamb, beta, indexOfPenalty, useKernel = False, sigma = 1, width = 5, verbose = False, eps = 3e-3, epsAbs = 1e-3, epsRel = 1e-3): 
# Shuheng       
    if indexOfPenalty == 1:
        print 'Use l-1 penalty function'
        from inferGraphL1 import *
    elif indexOfPenalty == 2:
        print 'Use l-2 penalty function'
        from inferGraphL2 import *
    elif indexOfPenalty == 3:
        print 'Use laplacian penalty function'
        from inferGraphLaplacian import *
    elif indexOfPenalty == 4:
        print 'Use l-inf penalty function'
        from inferGraphLinf import *
    else:
        print 'Use perturbation node penalty function'
        from inferGraphPN import *

    numberOfTotalSamples = data.shape[0]
    timestamps = int(numberOfTotalSamples/lengthOfSlice)    
    size = data.shape[1]
    # Generate empirical covariance matrices
    sampleSet = []    # list of array
    k = 0
    for i in range(timestamps):
        # Generate the slice of samples for each timestamp from data
        k_next = min(k + lengthOfSlice, numberOfTotalSamples)
        samples = data[k : k_next, :]
        k = k_next
        sampleSet.append(samples)
    
    empCovSet = []    # list of array
    if useKernel != True:
        for i in range(timestamps):
            empCov = GenEmpCov(sampleSet[i].T)
            empCovSet.append(empCov)
    else:
        for i in range(timestamps):
            empCov = genEmpCov_kernel(i, sigma, width, sampleSet)
            empCovSet.append(empCov)
            
    # delete: for checking
#    print sampleSet.__len__() # 
#    print empCovSet
    print 'lambda = %s, beta = %s'%(lamb, beta)
    
    # Define a graph representation to solve
    gvx = TGraphVX()   
    for i in range(timestamps):
        n_id = i
        S = semidefinite(size, name='S')
        obj = -log_det(S) + trace(empCovSet[i] * S) #+ alpha*norm(S,1)
        gvx.AddNode(n_id, obj)
        
        if (i > 0): #Add edge to previous timestamp
            prev_Nid = n_id - 1
            currVar = gvx.GetNodeVariables(n_id)
            prevVar = gvx.GetNodeVariables(prev_Nid)            
            edge_obj = beta * norm(currVar['S'] - prevVar['S'], indexOfPenalty) 
            gvx.AddEdge(n_id, prev_Nid, Objective = edge_obj)
        
        #Add rake nodes, edges
        gvx.AddNode(n_id + timestamps)
        gvx.AddEdge(n_id, n_id + timestamps, Objective= lamb * norm(S,1))
            
    # need to write the parameters of ADMM
#    gvx.Solve()
    gvx.Solve(EpsAbs=epsAbs, EpsRel=epsRel, Verbose = verbose)
   # gvx.Solve(MaxIters = 700, Verbose = True, EpsAbs=eps_abs, EpsRel=eps_rel)
    # gvx.Solve( NumProcessors = 1, MaxIters = 3)
    
    # Extract the set of estimated theta 
    thetaSet = []
    for nodeID in range(timestamps):
        val = gvx.GetNodeValue(nodeID,'S')
        thetaEst = upper2FullTVGL(val, eps)
        thetaSet.append(thetaEst)
    return thetaSet

def myTVGL(mydata, lengthOfSlice, lamb, beta, indexOfPenalty, useKernel = False, sigma = 1, width = 5, verbose = False, eps = 3e-3, epsAbs = 1e-3, epsRel = 1e-3):        
# mydata: array of len_class by nt by p    
# sigma is h
    if indexOfPenalty == 1:
        print 'Use l-1 penalty function'
        from inferGraphL1 import *
    elif indexOfPenalty == 2:
        print 'Use l-2 penalty function'
        from inferGraphL2 import *
    elif indexOfPenalty == 3:
        print 'Use laplacian penalty function'
        from inferGraphLaplacian import *
    elif indexOfPenalty == 4:
        print 'Use l-inf penalty function'
        from inferGraphLinf import *
    else:
        print 'Use perturbation node penalty function'
        from inferGraphPN import *

    len_class = mydata.shape[0]
    numberOfTotalSamples = mydata.shape[1]
    timestamps = int(numberOfTotalSamples/lengthOfSlice)    
    size = mydata.shape[2]
    
    empCovSet_list = []
    for class_ix in range(len_class):
        data = mydata[class_ix]
        numberOfTotalSamples = data.shape[0]
        timestamps = int(numberOfTotalSamples/lengthOfSlice)    
        size = data.shape[1]
    # Generate empirical covariance matrices
        sampleSet = []    # list of array
        k = 0
        for i in range(timestamps):
        # Generate the slice of samples for each timestamp from data
            # k_next = np.divide((k+lengthOfSlice+numberOfTotalSamples)-abs(k+lengthOfSlice-numberOfTotalSamples), 2)
            k_next = np.min((k+lengthOfSlice, numberOfTotalSamples))   
            samples = data[k : k_next, :]
            k = k_next
            sampleSet.append(samples)
    
        empCovSet = []    # list of array
        if useKernel != True:
            for i in range(timestamps):
                empCov = GenEmpCov(sampleSet[i].T)
                empCovSet.append(empCov)
        else:
            for i in range(timestamps):
                empCov = genEmpCov_kernel(i, sigma, width, sampleSet)
                empCovSet.append(empCov)
        empCovSet_list.append(empCovSet)    
    
    
    # delete: for checking
#    print sampleSet.__len__() # 
#    print empCovSet
    print 'lambda = %s, beta = %s'%(lamb, beta)
    thetaSet_list = []
    for time_ix in range(timestamps):
        # Define a graph representation to solve
        gvx = TGraphVX()   
        for i in range(len_class):
            n_id = i
            S = semidefinite(size, name='S')
            obj = -log_det(S) + trace(empCovSet_list[i][time_ix] * S) #+ alpha*norm(S,1)
            gvx.AddNode(n_id, obj)
            
            if (i > 0): #Add edge to previous timestamp
                prev_Nid = n_id - 1
                currVar = gvx.GetNodeVariables(n_id)
                prevVar = gvx.GetNodeVariables(prev_Nid)            
                edge_obj = beta * norm(currVar['S'] - prevVar['S'], indexOfPenalty) 
                gvx.AddEdge(n_id, prev_Nid, Objective = edge_obj)
            
            #Add rake nodes, edges
            gvx.AddNode(n_id + len_class)
            gvx.AddEdge(n_id, n_id + len_class, Objective= lamb * norm(S,1))
                
        # need to write the parameters of ADMM
#   gvx.Solve()
        gvx.Solve(EpsAbs=epsAbs, EpsRel=epsRel, Verbose = verbose)
    # gvx.Solve(MaxIters = 700, Verbose = True, EpsAbs=eps_abs, EpsRel=eps_rel)
    # gvx.Solve( NumProcessors = 1, MaxIters = 3)
    
    # Extract the set of estimated theta 
        thetaSet = []
        for nodeID in range(len_class):
            val = gvx.GetNodeValue(nodeID,'S')
            thetaEst = upper2FullTVGL(val, eps)
            thetaSet.append(thetaEst)
        thetaSet_list.append(thetaSet)
    
    return thetaSet_list

def myTVGL0(mydata, lengthOfSlice, lamb, beta, indexOfPenalty, useKernel = False, sigma = 1, width = 5, verbose = False, eps = 3e-3, epsAbs = 1e-3, epsRel = 1e-3):           
# mydata ndarray t by n*len_class by p
    if indexOfPenalty == 1:
        print 'Use l-1 penalty function'
        from inferGraphL1 import *
    elif indexOfPenalty == 2:
        print 'Use l-2 penalty function'
        from inferGraphL2 import *
    elif indexOfPenalty == 3:
        print 'Use laplacian penalty function'
        from inferGraphLaplacian import *
    elif indexOfPenalty == 4:
        print 'Use l-inf penalty function'
        from inferGraphLinf import *
    else:
        print 'Use perturbation node penalty function'
        from inferGraphPN import *

    # len_t = len(mydata) # len_class to len_t
    len_t = mydata.shape[0]
    numberOfTotalSamples = mydata.shape[1] #  numberOfTotalSamples = n times len_class
    len_class = int(numberOfTotalSamples/lengthOfSlice) 
    # lengthOfSlice is the number of samples in each class
    # timestamps to len_class
    size = mydata.shape[2] # size = p

    mydata_array = np.array(mydata) # t by n*len_class by p
    mydata_array = np.reshape(mydata_array, (len_t, len_class, lengthOfSlice, size))
    mydata1_array = np.transpose(mydata_array, [1, 0, 2, 3]) # class by t by ni by p
    mydata1_array = np.reshape(mydata1_array, (len_class, lengthOfSlice*len_t, size))

    empCovSet_list = []
    for class_ix in range(len_class):
        data = mydata1_array[class_ix]
        numberOfTotalSamples = data.shape[0]
        timestamps = int(numberOfTotalSamples/lengthOfSlice)    
        size = data.shape[1]
    # Generate empirical covariance matrices
        sampleSet = []    # list of array
        k = 0
        for i in range(timestamps):
            # Generate the slice of samples for each timestamp from data
            k_next = min(k + lengthOfSlice, numberOfTotalSamples)
            samples = data[k : k_next, :]
            k = k_next
            sampleSet.append(samples)
    
        empCovSet = []    # list of array
        if useKernel != True:
            for i in range(timestamps):
                empCov = GenEmpCov(sampleSet[i].T)
                empCovSet.append(empCov)
        else:
            for i in range(timestamps):
                empCov = genEmpCov_kernel(i, sigma, width, sampleSet)
                empCovSet.append(empCov)
        empCovSet_list.append(empCovSet)   
        
        empCovSet_array = np.array(empCovSet_list) # len_class by timestamps by size by size
        empCovSet_array = np.transpose(empCovSet_array, [1, 0, 2, 3]) # t by class by size by size
        
    print 'lambda = %s, beta = %s'%(lamb, beta)
    thetaSet_list = []
    for time_ix in range(timestamps):
        # Define a graph representation to solve
        gvx = TGraphVX()   
        for i in range(len_class):
            n_id = i
            S = semidefinite(size, name='S')
            obj = -log_det(S) + trace(empCovSet_list[i][time_ix] * S) #+ alpha*norm(S,1)
            gvx.AddNode(n_id, obj)
            
            if (i > 0): #Add edge to previous timestamp
                prev_Nid = n_id - 1
                currVar = gvx.GetNodeVariables(n_id)
                prevVar = gvx.GetNodeVariables(prev_Nid)            
                edge_obj = beta * norm(currVar['S'] - prevVar['S'], indexOfPenalty) 
                gvx.AddEdge(n_id, prev_Nid, Objective = edge_obj)
            
            #Add rake nodes, edges
            gvx.AddNode(n_id + len_class)
            gvx.AddEdge(n_id, n_id + len_class, Objective= lamb * norm(S,1))
                
        # need to write the parameters of ADMM
#   gvx.Solve()
        gvx.Solve(EpsAbs=epsAbs, EpsRel=epsRel, Verbose = verbose)
   # gvx.Solve(MaxIters = 700, Verbose = True, EpsAbs=eps_abs, EpsRel=eps_rel)
    # gvx.Solve( NumProcessors = 1, MaxIters = 3)
    
    # Extract the set of estimated theta 
        thetaSet = []
        for nodeID in range(len_class):
            val = gvx.GetNodeValue(nodeID,'S')
            thetaEst = upper2FullTVGL(val, eps)
            thetaSet.append(thetaEst)
        thetaSet_list.append(thetaSet)
    return thetaSet_list    
        
#    print 'lambda = %s, beta = %s'%(lamb, beta)
#    thetaSet_list = []
#    for class_ix in range(len_class):
#        # Define a graph representation to solve
#        gvx = TGraphVX()   
#        for i in range(len_t):
#            n_id = i
#            S = semidefinite(size, name='S')
#            obj = -log_det(S) + trace(empCovSet_array[i][class_ix] * S) #+ alpha*norm(S,1)
#            gvx.AddNode(n_id, obj)
#            
#            if (i > 0): #Add edge to previous timestamp
#                prev_Nid = n_id - 1
#                currVar = gvx.GetNodeVariables(n_id)
#                prevVar = gvx.GetNodeVariables(prev_Nid)            
#                edge_obj = beta * norm(currVar['S'] - prevVar['S'], indexOfPenalty) 
#                gvx.AddEdge(n_id, prev_Nid, Objective = edge_obj)
#            
#            #Add rake nodes, edges
#            gvx.AddNode(n_id + len_class)
#            gvx.AddEdge(n_id, n_id + len_class, Objective= lamb * norm(S,1))
#                
#        # need to write the parameters of ADMM
##   gvx.Solve()
#        gvx.Solve(EpsAbs=epsAbs, EpsRel=epsRel, Verbose = verbose)
#   # gvx.Solve(MaxIters = 700, Verbose = True, EpsAbs=eps_abs, EpsRel=eps_rel)
#    # gvx.Solve( NumProcessors = 1, MaxIters = 3)
#    
#    # Extract the set of estimated theta 
#        thetaSet = []
#        for nodeID in range(len_t):
#            val = gvx.GetNodeValue(nodeID,'S')
#            thetaEst = upper2FullTVGL(val, eps)
#            thetaSet.append(thetaEst)
#        thetaSet_list.append(thetaSet)
#    return thetaSet_list

def GenEmpCov(samples, useKnownMean = False, m = 0):
    # samples should be array
    size, samplesPerStep = samples.shape
    if useKnownMean == False:
        m = np.mean(samples, axis = 1)
    empCov = 0
    for i in range(samplesPerStep):
        sample = samples[:,i]
        empCov = empCov + np.outer(sample - m, sample -m)
    empCov = empCov/samplesPerStep
    return empCov
    
def upper2FullTVGL(a, eps = 0):
    # a should be array
    ind = (a < eps) & (a > -eps)
    a[ind] = 0
    n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)  
    A = np.zeros([n,n])
    A[np.triu_indices(n)] = a 
    d = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(d))             
    return A   

# each element in sample_set is n by p len is total length of timestamp
# total width = 2*width + 1
def genEmpCov_kernel(t_query, sigma, width, sample_set, knownMean = False):
    timesteps = sample_set.__len__()
#    print timesteps
    K_sum = 0
    S = 0
#    print(range(int(max(0,t_query-width)), int(min(t_query+width+1, timesteps))))
    if knownMean != True:
        for j in range(int(max(0,t_query-width)), int(min(t_query+width+1, timesteps))):         
            K =  np.exp(-np.square(t_query - j)/sigma)
            samplesPerStep = sample_set[j].shape[0]
            mean = np.mean(sample_set[j], axis = 0) # p by 1
#            print mean
            mean_tile = np.tile(mean, (samplesPerStep,1))
#            print mean_tile.shape
            S = S + K*np.dot((sample_set[j]- mean_tile).T, sample_set[j] - mean_tile)/samplesPerStep
            K_sum = K_sum + K
    else:
        for j in range(int(max(0,t_query-width)), int(min(t_query+width+1, timesteps))):         
            K =  np.exp(-np.square(t_query - j)/sigma)
            samplesPerStep = sample_set[j].shape[0]
            S = S + K*np.dot((sample_set[j]).T, sample_set[j])/samplesPerStep
            K_sum = K_sum + K
    S = S/K_sum    
    return S

