#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing functions - General/Gaussian Linear Hidden Markov Model
@author: Diego Vidaurre 2023
"""

import math
import numpy as np
import warnings
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from scipy import signal
import scipy.io

from . import auxiliary
# import auxiliary

def apply_pca(X,d,whitening=False, exact=True):
    """Applies PCA to the input data X.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_parcels)
        The input data to be transformed.
    d : int or float
        If int, the number of components to keep.
        If float, the percentage of explained variance to keep.
        If array-like of shape (n_parcels, n_components), the transformation matrix.
    exact : bool, default=True
        Whether to use full SVD solver for PCA.

    Returns:
    --------
    X : array-like of shape (n_samples, n_components)
        The transformed data after applying PCA.
    pcamodel : sklearn estimator
        The estimated PCA model
    """
    if type(d) is np.ndarray:
        X -= np.mean(X,axis=0)
        X = X @ d
        whitening = True
        if whitening: X /= np.std(X,axis=0)
        return X, None

    svd_solver = 'full' if exact else 'auto'
    if d >= 1: 
        pcamodel = PCA(n_components=d,svd_solver=svd_solver)
        pcamodel.fit(X)
        X = pcamodel.transform(X)
    else: 
        pcamodel = PCA(svd_solver=svd_solver)
        pcamodel.fit(X)
        ncomp = np.where(np.cumsum(pcamodel.explained_variance_ratio_)>=d)[0][0] + 1
        X = pcamodel.transform(X)
        X = X[:,0:ncomp]
        d = ncomp

    # sign convention equal to Matlab's
    for j in range(d):
        jj = np.where(np.abs(X[:,j]) == np.abs(np.max(X[:,j])) )[0][0]
        if X[jj,j] < 0: X[:,j] *= -1

    return X, pcamodel

def apply_ica(X,d,algorithm='parallel'):
    """Applies ICA to the input data X.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_parcels)
        The input data to be transformed.
    d : int or float
        If int, the number of components to keep.
        If float, the percentage of explained variance to keep (according to a PCA decomposition)
    algorithm : {"parallel", "deflation"}, default="parallel"
        Specify which algorithm to use for FastICA.

    Returns:
    --------
    X : array-like of shape (n_samples, n_components)
        The transformed data after applying ICA.
    icamode : sklearn estimator
        The estimated ICA model
    """

    if d < 1:
        pcamodel = PCA()
        pcamodel.fit(X)
        ncomp = np.where(np.cumsum(pcamodel.explained_variance_ratio_)>=d)[0][0] + 1
    else: 
        ncomp = d

    icamodel = FastICA(n_components=ncomp,whiten='unit-variance',algorithm=algorithm)
    icamodel.fit(X)
    X = icamodel.transform(X)

    # sign convention equal to Matlab's
    for j in range(ncomp):
        jj = np.where(np.abs(X[:,j]) == np.abs(np.max(X[:,j])) )[0][0]
        if X[jj,j] < 0: X[:,j] *= -1

    return X, icamodel

def dampen_peaks(X,strength=5):
    """Applies dampening of extreme peaks to the input data X, at the group level.
    If the absolute value of X goes beyond 2 standard deviation of X, 
    it gets substituted by the logarithm of the absolute value of X.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_parcels)
        The input data to be transformed.
    strength : positive int
        The strength of dampening. This value refers to the base of the logarithm to use. 
        The bigger the base, the stronger the dampening.

    Returns:
    --------
    X_transformed : array-like of shape (n_samples, n_parcels)
        The transformed data after applying extreme peak dampening.
    """
    
    x_mask = np.abs(X)>2*np.std(X)
    X_transformed = X.copy()
    X_transformed[x_mask] = np.sign(X[x_mask])*(2*np.std(X) - np.log(2*np.std(X))/np.log(strength) + 
                                                np.log(np.abs(X[x_mask]))/np.log(strength))

    return X_transformed


def preprocess_data(data,indices,
        fs = 1, # frequency of the data
        dampen_extreme_peaks=None, # it can be None, True, or an int with the strength of dampening
        standardise=True, # True / False
        filter=None, # Tuple with low-pass high-pass thresholds, or None
        detrend=False, # True / False
        onpower=False, # True / False
        onphase=False, # True / False
        pca=None, # Number of principal components, % explained variance, or None
        exact_pca=True, # related to how to run PCA
        ica=None, # Number of independent components, % explained variance, or None (if specified, pca is not used)
        ica_algorithm='parallel', # related to how to run PCA
        post_standardise=None, # True / False, standardise the ICA/PCA components?
        downsample=None # new frequency, or None
        ):
    
    """Preprocess the input data.

    Parameters:
    -----------
    data : array-like of shape (n_samples, n_parcels)
        The input data to be preprocessed.

    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.

    fs : int or float, default=1
        The frequency of the input data.
        
    dampen_extreme_peaks : int, True or None, default=None
        determines whether to dampen extreme peaks in the data and the strength of the dampening. 
        If this is chosen, the data are centered first (per subject).
        If int, the strength of dampening
        If True, the dampening is done with default value 5.
        If None, no dampening will be applied.

    standardise : bool, default=True. 
        Whether to standardize the input data. 

    filter : tuple of length 2 or None, default=None
        The low-pass and high-pass thresholds to apply to the input data.
        If None, no filtering will be applied.
        If a tuple, the first element is the low-pass threshold and the second is the high-pass threshold.    

    detrend : bool, default=False
        Whether to detrend the input data.

    onpower : bool, default=False
        Whether to calculate the power of the input data using the Hilbert transform.

    onphase : bool, default=False
        Whether to calculate the phase of the input data using the Hilbert transform.
        If both onpower and onphase are set to True, power and phase will be included as separate columns in the output array

    pca : int or float or None, default=None
        If int, the number of components to keep after applying PCA.
        If float, the percentage of explained variance to keep after applying PCA (must be >=0 and >=1)
        If None, no PCA will be applied.

    exact_pca : bool, default=True
        Whether to use full SVD solver for PCA.
        
    ica : int or float or None, default=None
        determines whether to apply ICA (if pca is also specified, it is overridden, and only ica is used)
        If int, the number of independent components to estimate
        If float, the number of components is given by the percentage of explained variance from PCA.
        If None, no PCA will be applied.       

    ica_algorithm : {"parallel", "deflation"}, default="parallel"
        Specify which algorithm to use for ICA (based on FastICA).     

    post_standardise : bool, default=False if pca is used; =True if ica is used; 
        Whether to standardize after applying PCA/ICA (recommended when using the TDE-HMM)

    downsample : int or float or None, default=None
        The new frequency of the input data after downsampling.
        If None, no downsampling will be applied.

    Returns:
    --------
    data : array-like of shape (n_samples_processed, n_parcels)
        The preprocessed input data.

    indices_new : array-like of shape (n_sessions_processed, 2)
        The start and end indices of each trial/session in the preprocessed data.

    log : dict
        Dictionary logging which preprocessing has been applied, to be passed onto the
        glhmm object instance. This contains the variables passed to the function and (where relevant) 
        the estimators (PCA/ICA models)
    """
    p = data.shape[1]
    N = indices.shape[0]
    log = {**locals()}
    del(log["data"], log["indices"])

    data = np.copy(data)
    
    if dampen_extreme_peaks: 
        # center data first, per subject
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] -= np.mean(data[t,:],axis=0)
        # then dampen peaks at the group level    
        if isinstance(dampen_extreme_peaks,int):
            strength = dampen_extreme_peaks
        else:
            strength = 5
        data = dampen_peaks(data,strength)           
            
    if standardise:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] -= np.mean(data[t,:],axis=0)
            data[t,:] /= np.std(data[t,:],axis=0)   

    if filter != None:
        filterorder = 6
        if filter[0] == 0: # low-pass
            sos = signal.butter(filterorder, filter[1], 'lowpass', output='sos', fs = fs)
        elif filter[1] == None: # high-pass
            sos = signal.butter(filterorder, filter[0], 'highpass', output='sos', fs = fs)
        else:
            sos = signal.butter(filterorder, filter, 'bandpass', output='sos', fs = fs)
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1])
            data[t,:] = signal.sosfilt(sos, data[t,:], axis=0)
    
    if detrend:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] = signal.detrend(data[t,:], axis=0)       

    if onpower and not onphase:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] = np.abs(signal.hilbert(data[t,:], axis=0))

    if onphase and not onpower:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] = np.unwrap(np.angle(signal.hilbert(data[t,:], axis=0)))

    if onpower and onphase:
        data = np.concatenate((data,data),1)
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            analytical_signal = signal.hilbert(data[t,:p], axis=0)
            data[t,:p] = np.abs(analytical_signal)
            data[t,p:] = np.unwrap(np.angle(analytical_signal))
        p = data.shape[1]

    if (pca != None) and (ica is None):
        data, pcamodel = apply_pca(data,pca,exact_pca)
        p = data.shape[1]
        log["pcamodel"] = pcamodel

    if ica != None:
        data, icamodel = apply_ica(data,ica,ica_algorithm)
        p = data.shape[1]
        log["icamodel"] = icamodel       

    if post_standardise is None:
        if ica: post_standardise = True
        else: post_standardise = False

    if (pca or ica) and post_standardise:
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            data[t,:] -= np.mean(data[t,:],axis=0)
            data[t,:] /= np.std(data[t,:],axis=0)          
        
    if downsample != None:
        factor = downsample / fs
        Tnew = np.ceil(factor * (indices[:,1]-indices[:,0])).astype(int)
        indices_new = auxiliary.make_indices_from_T(Tnew)
        data_new = np.zeros((np.sum(Tnew),p))
        gcd = math.gcd(downsample,fs)
        for j in range(N):
            t = np.arange(indices[j,0],indices[j,1]) 
            tnew = np.arange(indices_new[j,0],indices_new[j,1]) 
            data_new[tnew,:] = signal.resample_poly(data[t,:], downsample/gcd, fs/gcd)
            # Tjnew = tnew.shape[0]
            # data_new[tnew,:] = signal.resample(data[t,:], Tjnew)     
        data = data_new
    else: indices_new = indices

    return data,indices_new,log


def build_data_autoregressive(data,indices,autoregressive_order=1,
        connectivity=None,center_data=True):
    """Builds X and Y for the autoregressive model, 
    as well as an adapted indices array and predefined connectivity 
    matrix in the right format. X and Y are centered by default.
    
    Parameters:
    -----------
    data : array-like of shape (n_samples,n_parcels)
        The data timeseries.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.
    autoregressive_order : int, optional, default=1
        The number of lags to include in the autoregressive model.
    connectivity : array-like of shape (n_parcels, n_parcels), optional, default=None
        The matrix indicating which regressors should be used for each variable.
    center_data : bool, optional, default=True
        If True, the data will be centered.

    Returns:
    --------
    X : array-like of shape (n_samples - n_sessions*autoregressive_order, n_parcels*autoregressive_order)
        The timeseries of set of variables 1 (i.e., the regressors).
    Y : array-like of shape (n_samples - n_sessions*autoregressive_order, n_parcels)
        The timeseries of set of variables 2 (i.e., variables to predict, targets).
    indices_new : array-like of shape (n_sessions, 2)
        The new array of start and end indices for each trial/session.
    connectivity_new : array-like of shape (n_parcels*autoregressive_order, n_parcels)
        The new connectivity matrix indicating which regressors should be used for each variable.

    """

    T,p = data.shape
    N = indices.shape[0]

    if autoregressive_order == 0:
        warnings.warn("autoregressive_order is 0 so nothing to be done")
        return np.empty(0),data,indices,connectivity
    
    X = np.zeros((T - N*autoregressive_order,p*autoregressive_order))
    Y = np.zeros((T - N*autoregressive_order,p))
    indices_new = np.zeros((N,2)).astype(int)

    for j in range(N):
        ind_1 = np.arange(indices[j,0]+autoregressive_order,indices[j,1],dtype=np.int64)
        ind_2 = np.arange(indices[j,0],indices[j,1]-autoregressive_order,dtype=np.int64) \
            - j * autoregressive_order
        Y[ind_2,:] = data[ind_1,:]
        for i in range(autoregressive_order):
            ind_3 = np.arange(indices[j,0]+autoregressive_order-(i+1),indices[j,1]-(i+1),dtype=np.int64)
            ind_ch = np.arange(p) + i * p
            X[ind_2,ind_ch[:,np.newaxis]] = data[ind_3,:].T
        indices_new[j,0] = ind_2[0]
        indices_new[j,1] = ind_2[-1] + 1

    # center
    if center_data:
        Y -= np.mean(Y,axis=0)
        X -= np.mean(X,axis=0)

    if connectivity is not None:
        # connectivity_new : (regressors by regressed) 
        connectivity_new = np.zeros((autoregressive_order*p,p))
        for i in range(autoregressive_order):
            ind_ch = np.arange(p) + i * p
            connectivity_new[ind_ch,:] = connectivity
        # regress out when asked
        for j in range(p):
            jj = np.where(connectivity_new[:,j]==0)[0]
            if len(jj)==0: continue
            b = np.linalg.inv(X[:,jj].T @ X[:,jj] + 0.001 * np.eye(len(jj))) \
                @ (X[:,jj].T @ Y[:,j])
            Y[:,j] -= X[:,jj] @ b
        # remove unused variables
        active_X = np.zeros(p,dtype=bool)
        active_Y = np.zeros(p,dtype=bool)
        for j in range(p):
            active_X[j] = np.sum(connectivity[j,:]==1) > 0
            active_Y[j] = np.sum(connectivity[:,j]==1) > 0
        active_X = np.tile(active_X,autoregressive_order)
        active_X = np.where(active_X)[0]
        active_Y = np.where(active_Y)[0]
        Y = Y[:,active_Y]
        X = X[:,active_X]
        connectivity_new = connectivity_new[active_X,active_Y[:,np.newaxis]].T
    else: connectivity_new = None

    return X,Y,indices_new,connectivity_new


def build_data_partial_connectivity(X,Y,connectivity=None,center_data=True):
    """Builds X and Y for the partial connectivity model, 
    essentially regressing out things when indicated in connectivity,
    and getting rid of regressors / regressed variables that are not used;
    it return connectivity with the right dimensions as well. 

    Parameters:
    -----------
    X : np.ndarray of shape (n_samples, n_parcels)
        The timeseries of set of variables 1 (i.e., the regressors).
    Y : np.ndarray of shape (n_samples, n_parcels)
        The timeseries of set of variables 2 (i.e., variables to predict, targets).
    connectivity : np.ndarray of shape (n_parcels, n_parcels), optional, default=None
        A binary matrix indicating which regressors affect which targets (i.e., variables to predict). 
    center_data : bool, default=True
        Center data to zero mean.

    Returns:
    --------
    X_new : np.ndarray of shape (n_samples, n_active_parcels)
        The timeseries of set of variables 1 (i.e., the regressors) after removing unused predictors and regressing out 
        the effects indicated in connectivity.
    Y_new : np.ndarray of shape (n_samples, n_active_parcels)
        The timeseries of set of variables 2 (i.e., variables to predict, targets) after removing unused targets and regressing out 
        the effects indicated in connectivity.
    connectivity_new : np.ndarray of shape (n_active_parcels, n_active_parcels), optional, default=None
        A binary matrix indicating which regressors affect which targets
        The matrix has the same structure as `connectivity` after removing unused predictors and targets.
    """

    X_new = np.copy(X)
    Y_new = np.copy(Y)

    if connectivity is not None:
        p = X.shape[1]
        q = Y.shape[1]
        # regress out when asked
        for j in range(q):
            jj = np.where(connectivity[:,j]==0)[0]
            if len(jj)==0: continue
            b = np.linalg.inv(X[:,jj].T @ X[:,jj] + 0.001 * np.eye(len(jj))) \
                @ (X[:,jj].T @ Y[:,j])
            Y_new[:,j] -= X[:,jj] @ b
        # remove unused variables
        active_X = np.zeros(p,dtype=bool)
        for j in range(p): active_X[j] = np.sum(connectivity[j,:]==1) > 0
        active_Y = np.zeros(q,dtype=bool)
        for j in range(q): active_Y[j] = np.sum(connectivity[:,j]==1) > 0
        active_X = np.where(active_X)[0]
        active_Y = np.where(active_Y)[0]
        Y = Y[:,active_Y]
        X = X[:,active_X]
        # copy of connectivity
        connectivity_new = np.copy(connectivity)
        connectivity_new = connectivity_new[active_X,active_Y[:,np.newaxis]].T
    else: connectivity_new = None

    # center
    if center_data:
        Y_new = Y_new - np.mean(Y_new,axis=0)
        X_new = X_new - np.mean(X_new,axis=0)

    return X_new,Y_new,connectivity_new


def build_data_tde(data,indices,lags,pca=None,standardise_pc=True):
    """Builds X for the temporal delay embedded HMM, as well as an adapted indices array.

    Parameters:
    -----------
    data : numpy array of shape (n_samples, n_parcels)
        The data matrix.
    indices : array-like of shape (n_sessions, 2)
        The start and end indices of each trial/session in the input data.
    lags : list or array-like
        The lags to use for the embedding.
    pca : None or int or float or numpy array, default=None
        The number of components for PCA, the explained variance for PCA, the precomputed PCA projection matrix, 
        or None to skip PCA.
    standardise_pc : bool, default=True
        Whether or not to standardise the principal components before returning.

    Returns:
    --------
    X : numpy array of shape (n_samples - n_sessions*rwindow, n_parcels*n_lags)
        The delay-embedded timeseries data.
    indices_new : numpy array of shape (n_sessions, 2)
        The adapted indices for each segment of delay-embedded data.
    pcamodel : sklearn estimator, optional
        If doing PCA, the estimated PCA model

    PCA can be run optionally: if pca >=1, that is the number of components;
    if pca < 1, that is explained variance;
    if pca is a numpy array, then it is a precomputed PCA projection matrix;
    if pca is None, then no PCA is run.
    """

    T,p = data.shape
    N = indices.shape[0]
    
    L = len(lags)
    minlag = np.min(lags)
    maxlag = np.max(lags)
    rwindow = maxlag-minlag

    X = np.zeros((T - N*rwindow,p*L))
    indices_new = np.zeros((N,2)).astype(int)

    # Embedding
    for j in range(N):
        ind_1 = np.arange(indices[j,0],indices[j,1],dtype=np.int64)
        ind_2 = np.arange(indices[j,0],indices[j,1]-rwindow,dtype=np.int64) - j * rwindow
        for i in range(L):
            l = lags[i]
            X_l = np.roll(data[ind_1,:],l,axis=0)
            X_l = X_l[-minlag:-maxlag,:]
            ind_ch = np.arange(i,L*p,L)
            X[ind_2,ind_ch[:,np.newaxis]] = X_l.T
        indices_new[j,0] = ind_2[0]
        indices_new[j,1] = ind_2[-1] + 1

    # Standardise (in Matlab's HMM-MAR we only centered pre-embedding)
    # note that this is done for the entire data set and not per sessions
    X -= np.mean(X,axis=0)
    X /= np.std(X,axis=0)

    # PCA and whitening 
    if pca is not None:
        X, pcamodel = apply_pca(X,pca,standardise_pc)

    if pca is not None:
        return X,indices_new,pcamodel
    else:
        return X,indices_new


def load_files(files,I=None,do_only_indices=False):        

    X = []
    Y = []
    indices = []
    sum_T = 0

    if I is None:
        I = np.arange(len(files))
    elif type(I) is int:
        I = np.array([I])

    for ij in range(I.shape[0]):

        j = I[ij]

        # if type(files[j]) is tuple:
        #     if len(files[j][0]) > 0: # X
        #         if files[j][0][-4:] == '.npy':
        #             X.append(np.load(files[j][0]))
        #         elif files[j][0][-4:] == '.txt':

        if files[j][-4:] == '.mat':
            dat = scipy.io.loadmat(files[j])

        elif files[j][-4:] == '.npz':
            dat = np.load(files[j])
            
        if not do_only_indices:
            if ('X' in dat) and (not 'Y' in dat): 
                Y.append(dat["X"])
            else:
                if 'X' in dat: X.append(dat["X"])
                Y.append(dat["Y"])
        if 'indices' in dat: 
            indices.append(dat['indices'])
        elif 'T' in dat:
            indices.append(auxiliary.make_indices_from_T(dat['T']) + sum_T)
        else:
            ind = np.zeros((1,2)).astype(int)
            ind[0,0] = 0
            ind[0,1] = Y[-1].shape[0]
            indices.append(ind + sum_T)

        sum_T += dat["Y"].shape[0]

    if not do_only_indices:
        if len(X) > 0: X = np.concatenate(X)
        Y = np.concatenate(Y)
    indices = np.concatenate(indices)
    if len(indices.shape) == 1: indices = np.expand_dims(indices,axis=0)
    if len(X) == 0: X = None

    return X,Y,indices