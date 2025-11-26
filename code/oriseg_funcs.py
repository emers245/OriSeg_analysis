#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:48:06 2023

@author: joe

OriSeg Functions: This code holds all the functions used for doing analysis 
and making plots related to the OriSeg dataset on context modulation.
"""

#import nibabel
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import collections.abc
import scipy.interpolate as sciInterp
from typing import Dict
from matplotlib.patches import Ellipse
from statsmodels.stats.multitest import multipletests
from collections.abc import Iterable

def makeProfile1D(depthData, nDepths, paramData, depthMin, depthMax, useLayNii):
    '''
    Makes depth profile with given depth boundaries.
    
    Inputs:
        depthData (array): depth values for each voxel in ROI
        nDepths (int): number of depth bins to use
        paramData (array): the measurement you want to return grouped by depth
        depthMax (float): the value for the maximum depth
        useLayNii (bool): use the laynii depths if True, otherwise use depthify
        
    Outputs:
        dataDict (dict): a dictionary of statistics computed for the profile 
            and over the whole ROI
        
    Dependencies:
        numpy
    '''
    # set up the depths
    binSize = (depthMax-depthMin)/nDepths
    print('depth bin size: %.2f' %binSize)
    if useLayNii:
        depthBoundaries = np.linspace(depthMin-binSize/2, depthMax, nDepths+1)
    else:
        depthBoundaries = 0.5 + np.arange(depthMin, depthMax + 1., binSize) 
    # mask the depth data using the ROI
    try:
        nParams = paramData.shape[1]
    except:
        nParams = 1
        paramData = np.reshape(paramData, (len(paramData), 1))
    dataDict = {'whole ROI': {}, 'profile': {}}
    dataDict['whole ROI']['nVox'] = len(depthData)
    dataDict['whole ROI']['avg'] = np.mean(paramData, axis=0)
    dataDict['profile']['depth'] = [0 for iD in range(len(depthBoundaries) - 1)]
    dataDict['profile']['nVox'] = [0 for iD in range(len(depthBoundaries) - 1)]
    dataDict['profile']['avg'] = [[0 for iD in range(len(depthBoundaries) - 1)] for iP in range(nParams)]
    dataDict['profile']['param'] = [[0 for iD in range(len(depthBoundaries) - 1)] for iP in range(nParams)]
    dataDict['profile']['std'] = [[0 for iD in range(len(depthBoundaries) - 1)] for iP in range(nParams)]
    for iP in range(nParams):
        for iD in range(len(depthBoundaries) - 1):
            mask = 1.*(depthData > depthBoundaries[iD])*(depthData <= depthBoundaries[iD + 1])
            dataDict['profile']['depth'][iD] = np.mean(depthData[mask > 0.])
            dataDict['profile']['nVox'][iD] = np.sum(mask)
            dataDict['profile']['param'][iP][iD] = paramData[:, iP][mask > 0.]
            dataDict['profile']['avg'][iP][iD] = np.mean(paramData[:, iP][mask > 0.])
            dataDict['profile']['std'][iP][iD] = np.std(paramData[:, iP][mask > 0.])
    return dataDict

def bin_depths(data, nDepths, depthMin, depthMax, useLayNii):
    
    # set up the depths
    binSize = (depthMax-depthMin)/nDepths
    print('depth bin size: %d' %binSize)
    if useLayNii:
        depthBoundaries = np.linspace(depthMin-binSize/2, depthMax, nDepths+1)
    else:
        depthBoundaries = 0.5 + np.arange(depthMin, depthMax + 1., binSize) 
    
    depth_digitized = np.digitize(data,bins=depthBoundaries[1:-1])
    
    return depth_digitized

def smooth_kernel(x, k, xloc):
    ''' copied from Joe's analysisROI_dev and changed name of last variable
    to help guide my own interpretation, and changed it so
    kernel normalization happens in here to shorten call in smoothen
    
    Inputs:
        x (array): coordinates
        k (float): smoothing factor (-1/k is exponent for kernel)
        xloc (array): coordinates to center kernel
    
    Outputs:
        kernel (array): normalized exponential kernel
        
    Dependencies:
        numpy
    '''
    kernel = np.exp(-(1/k)*np.abs(x-xloc))
    return kernel/np.sum(kernel)

def smoothen(data, x, radMax=4, nRadii=50, kernel=smooth_kernel, smooth_factor=.3):
    ''' copied from Joe's analysisROI_dev; reduced input args
    data: 1D vector of data. doesn't need to be sorted!
    smooth_factor is the exponential decay constant of the smoothing kernel
    
    Inputs:
        data (array): vector of data
        x (array): coordinate vector whose elements correspond to data
        radMax (float): max radius to smooth in the same units the data are in
            DEFAULT - 4
        nRadii (int): number of evenly spaced points to resample to
            DEFAULT - 50
        kernel (func): a kernel function for smoothing
            DEFUALT - smooth_kernel (exponential kernel)
        smooth_factor (float): a parameter used in the kernel that changes the 
            width of the kernel function
            DEFAULT - 0.3
    Outputs:
        data_smooth (array): smoothed data resampled to new size
        x_smooth (array): resampled coordinates corresponding to data_smooth
        
    Dependencies:
        numpy
    '''
    data_smooth = np.zeros((nRadii,))
    x_smooth = np.linspace(0, radMax, nRadii)
    for iR, rad in enumerate(x_smooth):
        data_smooth[iR] = np.sum(data*kernel(x, smooth_factor, rad)) 
    return data_smooth, x_smooth

def fix_axes(ax, lcolor='black', fcolor='white'):
    '''
    Formatting for plot axes used in depth profiles
    
    Inputs:
        ax (matplotlib.pyplot handle): axis handle
        lcolor (string): color of lines
            DEFAULT - 'black'
        fcolor (strong): color of background
            DEFAULT - 'white'
            
    Outputs:
        None
        
    Dependencies:
        matplotlib
    '''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('outward', 4)) #8))
    ax.spines['left'].set_color(lcolor)
    ax.spines['bottom'].set_position(('outward', 4)) #8))
    ax.spines['bottom'].set_color(lcolor)
    ax.tick_params(axis='x', colors=lcolor)
    ax.tick_params(axis='y', colors=lcolor)
    ax.patch.set_facecolor(fcolor)
    
def smooth_kernel2D(x,y,k,tx,ty):
    '''
    A 2D exponential kernel
    
    Inputs:
        x (array): x-coordinates
        y (array): y-coordinates
        k (float): smoothing factor (-1/k is exponent for kernel)
        tx (array): x-coordinates for kernel center
        ty (array): y-coordinates for kernel center
        
    Outputs:
        kernel (array): 2D normalized exponential kernel
        
    Dependencies:
        numpy
    '''
    return np.exp(-(1/k)*np.abs(np.sqrt((x-tx)**2+(y-ty)**2)))

def smoothen2D(data,xy,xy_resamp,kernel,smooth_factor):
    '''
    A 2D smoothing function
    
    Inputs:
        data (array): vector of data
        xy (array): vector of xy coordinates corresponding to each element in 
            data
        xy_resamp (array): vector of xy coordinates to resample to
        kernel (func): kernel function for smoothing
        smooth_factor (float): parameter for kernel function that controls 
            width of kernel 
        
    Outputs:
        data_smooth (array): smoothed data resampled to new coordinates

    '''
    data_smooth = np.zeros([np.shape(xy_resamp)[1],np.shape(xy_resamp)[1]])
    for xi in range(np.shape(xy_resamp)[1]):
        for yi in range(np.shape(xy_resamp)[1]):
            data_smooth[xi,yi] = np.sum(data*kernel(xy[0],xy[1],smooth_factor,xy_resamp[0][xi],xy_resamp[1][yi]))/np.sum(kernel(xy[0],xy[1],smooth_factor,xy_resamp[0][xi],xy_resamp[1][yi]))
    return data_smooth

def compute_all_depth_profiles(all_data,statDetails,profile_method,nDepths,mask,stat_mask=None,depthParam='z',radialParam='scale_xy_dist',spec_Drange='MinMax',smooth_factor=0.3,radMax=4,statTestType='t-test',num_permutations=10000):
    '''
    Computes depth profiles for each subject and returns data.
    Inputs:
      all_data (dict): all voxel beta weights for each individual
      statDetails (dict): a dictionary containing labels of the conditions and 
          colors
      profile_method (str): the profile method to use (bin or smooth)
      nDepths (int): number of depths to sample
      mask (dict): voxel mask; each item should contain a boolean array
      stat_mask (dict): an optional mask specific to individual statistics; 
          this could be a p-value threshold for instance
          Default - None
      depthParam (str): name of the variable to use as the depth parameter 
          (this is probably going to be z for depthify or d for LayNii)
          DEFAULT - 'z'
      radialParam (str): name of the varuable to use as the radial parameter 
          (this is probably going to be scale_xy_dist for depthify or scal_uv_dist for LayNii)
          DEFAULT - 'scale_xy_dist'
      spec_Drange (str or list/tuple/arraylike): if list/tuple/arraylike, uses the first value as the minimum depth 
          (WM side) and the second value as the maximum depth (pial side); if
          str and ='MinMax', use the maximum and minimum values specified in 
          the depth variable as the boundaries
          DEFAULT - 'MinMax'
      smooth_factor (float): smoothing factor if using smooth depth profiles
          DEFAULT - 0.3
      radMax (float): max radius (in SD) to use only if using smooth depth 
          profiles
          DEFAULT - 4
    
    Outputs:
      keep_rois (dict): a dictionary containing the depth profies for each 
          coniditon  provided in statDetails with the following information:
              1) average depth profiles (NROIs x Ndepths)
              2) standard deviations at each depth (NROIs x Ndepths)
              3) list of depths (NROIS x Ndepths)
              4) number of voxels at each depth (NROIs x Ndepths)
    
    Dependencies:
        numpy, collections, makeProfile1D, smoothen, smooth_kernel
        
    Notes:
        The current version does not return stdev for the smooth profiles.
        !!!WARNING!!! Permutation test does not work with current implementation of makeProfile1D (11/05/2024)
    '''
    
    # pick out ROIs where we're sure of localization
    keep_rois = {label: {'avg': [], 'stdev': [], 'N': [], 'depths': [], 'p-vals': []} for label in statDetails['labels']}
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        if np.sum(mask[label]) == 0:
            print(f"Skipping ROI {label} as it does not exist in the DataFrame")
            continue
        roi_idx = mask[label] == 1
        roi = df.loc[roi_idx]
        # roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
        if radialParam != None:
            roi = roi[roi[radialParam] >=0]
    
        for stat in statDetails['labels']:
            if profile_method == 'bin':
                if spec_Drange == 'MinMax':
                    if stat_mask == None:
                        dataDict = makeProfile1D(roi[depthParam].values,
                                                     nDepths,
                                                     roi[stat].values,
                                                     np.min(roi[depthParam].values),
                                                     np.max(roi[depthParam].values),
                                                     depthParam=='d') #assumes you want to use LayNii if the depthParam is d
                    else:
                        dataDict = makeProfile1D(roi[depthParam].values[stat_mask[label][stat+' F']],
                                                     nDepths,
                                                     roi[stat].values[stat_mask[label][stat+' F']],
                                                     np.min(roi[depthParam].values[stat_mask[label][stat+' F']]),
                                                     np.max(roi[depthParam].values[stat_mask[label][stat+' F']]),
                                                     depthParam=='d') #assumes you want to use LayNii if the depthParam is d
                elif isinstance(spec_Drange,(collections.abc.Sequence, np.ndarray)):
                    if stat_mask == None:
                        dataDict = makeProfile1D(roi[depthParam].values,
                                                     nDepths,
                                                     roi[stat].values,
                                                     spec_Drange[0],
                                                     spec_Drange[1],
                                                     depthParam=='d') #assumed you want to use LayNii if the depthParam is d
                    else:
                        dataDict = makeProfile1D(roi[depthParam].values[stat_mask[label][stat+' F']],
                                                     nDepths,
                                                     roi[stat].values[stat_mask[label][stat+' F']],
                                                     spec_Drange[0],
                                                     spec_Drange[1],
                                                     depthParam=='d') #assumed you want to use LayNii if the depthParam is d
                else:
                    print("compute_all_depth_profiles: Invalid variable type for spec_Drange!")
                    
                if statTestType == 't-test':
                    #one-sample two-sided t-test
                    #tests against the null hypothesis that there is no effect on beta weights from condition
                    tt = np.asarray(np.asarray(dataDict['profile']['avg'])/(np.asarray(dataDict['profile']['std'])/np.sqrt(dataDict['profile']['nVox']))) #T statistic
                    pvals = stats.t.sf(np.abs(tt), np.asarray(dataDict['profile']['nVox'])-1)*2 #2-sided t-test returns p-value
                elif statTestType == 'permutation':
                    ##!!!WARNING!!! Permutation test does not work with current implementation of makeProfile1D (11/05/2024)
                    averages = dataDict['profile']['avg']
                    pvals = permute_1samp(averages, np.mean, null_stat=0, n_permutations=num_permutations, test_type='two-sided', axis=0)
                else:
                    print("Invalid statTestType type!!! Using 't-test' as default.")
                    tt = np.asarray(np.asarray(dataDict['profile']['avg'])/(np.asarray(dataDict['profile']['std'])/np.sqrt(dataDict['profile']['nVox']))) #T statistic
                    pvals = stats.t.sf(np.abs(tt), np.asarray(dataDict['profile']['nVox'])-1)*2 #2-sided t-test returns p-value
                    
                keep_rois[stat]['depths'].append(np.squeeze(dataDict['profile']['depth']))
                keep_rois[stat]['avg'].append(np.squeeze(dataDict['profile']['avg']))
                keep_rois[stat]['stdev'].append(np.squeeze(dataDict['profile']['std']))
                keep_rois[stat]['N'].append(np.squeeze(dataDict['profile']['nVox']))
                keep_rois[stat]['p-vals'].append(np.squeeze(pvals))
                
            elif profile_method == 'smooth':
                profiles = []
                profile, depth = smoothen(roi[stat],
                                          roi[depthParam].values,
                                          radMax,
                                          nRadii=nDepths,
                                          kernel=smooth_kernel,
                                          smooth_factor=smooth_factor)
                keep_rois[stat]['depths'].append(np.squeeze(depth))
                keep_rois[stat]['avg'].append(np.squeeze(profiles))
        
    return(keep_rois)

def compute_diff_profiles(all_data,statDetails,diffDetails,profile_method,nDepths,useSI,mask,depthParam='z',radialParam='x',spec_Drange='MinMax',smooth_factor=0.3,radMax=4,statTestType='t-test',num_permutations=10000):
    '''
    Plots depth profiles for each subject and returns data.
    Inputs:
      all_data (dict): all voxel beta weights for each individual
      statDetails (dict): a dictionary containing labels of the conditions and 
          colors
      diffDetails (dict): a dictionary containing the labels of the difference 
          profiles and the corresponding the conditions to use in the 
          differences (e.g. {'diff1': ['cond1','cond2'], 'diff2': ['cond1','cond3']})
      profile_method (str): the profile method to use (bin or smooth)
      nDepths (int): number of depths to sample
      useSI (bool): tells the function wether to use suppression index (true) 
          rather than pure difference (false) for the difference profiles
      mask (dict): voxel mask; each item should contain a boolean array
      depthParam (str): name of the variable to use as the depth parameter 
          (this is probably going to be z for depthify or d for LayNii)
          DEFAULT - 'z'
      radialParam (str): name of the varuable to use as the radial parameter 
          (this is probably going to be scale_xy_dist for depthify or 
          scal_uv_dist for LayNii)
          DEFAULT - 'scale_xy_dist'
      spec_Drange (str or list/tuple/arraylike): if list/tuple/arraylike, uses 
          the first value as the minimum depth (WM side) and the second value 
          as the maximum depth (pial side); if str and ='MinMax', use the 
          maximum and minimum values specified in the depth variable as the 
          boundaries
          DEFAULT - 'MinMax'
      smooth_factor (float): smoothing factor if using smooth depth profiles
          DEFAULT - 0.3
      radMax (float): max radius (in SD) to use only if using smooth depth 
          profiles
          DEFAULT - 4
    
    Outputs:
      keep_diffs (dict): a dictionary containing difference profiles for each
          difference condition listed in diffDetails with the following 
          information:
              1) average difference between specified conditions at 
                 each depth (NROIs x Ndepths)
              2) standard deviations at each depth (NROIS x Ndepths)
              3) list of depths (NROIS x Ndepths)
              4) number of voxels at each depth (NROIs x Ndepths)
    
    Dependencies:
        numpy, matplotlib, makeProfile1D, smoothen, smooth_kernel
        
    Notes:
        !!!WARNING!!! Permutation test does not work with current implementation of makeProfile1D (11/05/2024)
    '''
    
    keep_diffs = {diff_label: {'avg': [], 'stdev': [], 'N': [], 'depths': [], 'p-vals': []} for diff_label in diffDetails.keys()}
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        if np.sum(mask[label]) == 0:
            print(f"Skipping ROI {label} as it does not exist in the DataFrame")
            continue
        roi_idx = mask[label] == 1
        roi = df.loc[roi_idx]
        # roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
        if radialParam != None:
            roi = roi[roi[radialParam] >=0]
    
        if profile_method == 'bin':
            if spec_Drange == 'MinMax':
                dataDict = makeProfile1D(roi[depthParam].values,
                                             nDepths,
                                             roi[statDetails['labels']].values,
                                             np.min(roi[depthParam].values),
                                             np.max(roi[depthParam].values),
                                             depthParam=='d') #assumes you want to use LayNii if the depth parameter is d
            elif isinstance(spec_Drange,(collections.abc.Sequence, np.ndarray)):
                dataDict = makeProfile1D(roi[depthParam].values,
                                             nDepths,
                                             roi[statDetails['labels']].values,
                                             spec_Drange[0],
                                             spec_Drange[1],
                                             depthParam=='d')
            else:
                print("compute_all_diff_profiles: Invalid variable type for spec_Drange!")
                
            for diff_label in diffDetails.keys():
                cond1 = diffDetails[diff_label][0]
                cond2 = diffDetails[diff_label][1]
                c1i = np.where(np.array(statDetails['labels']) == cond1)[0][0]
                c2i = np.where(np.array(statDetails['labels']) == cond2)[0][0]
                if c1i == None or c2i == None:
                    print("compute_diff_profiles: Invalid condition type!")
                if useSI:
                    cond_diff = np.asarray(dataDict['profile']['avg'][c1i]) - np.asarray(dataDict['profile']['avg'][c2i])
                    cond_sum = np.asarray(dataDict['profile']['avg'][c1i]) + np.asarray(dataDict['profile']['avg'][c2i])
                    diff_avg = cond_diff / cond_sum
                    stdev_diff = np.sqrt(np.asarray(dataDict['profile']['std'][c1i])**2 + np.asarray(dataDict['profile']['std'][c2i])**2) #propagate errors for subtraction or addition
                    stdev = np.sqrt((stdev_diff/cond_diff)**2 + (stdev_diff/cond_sum)**2) #propagate errors for division
                else:
                    diff_avg = np.asarray(dataDict['profile']['avg'][c1i]) - np.asarray(dataDict['profile']['avg'][c2i])
                    #stdev = np.sqrt(np.asarray(dataDict['profile']['std'][c1i])**2 + np.asarray(dataDict['profile']['std'][c2i])**2) #propagate errors for subtraction
                    #or since I have the indivdual voxels, just take the st. dev. of the contrasts
                    stdev = np.array([np.std(dataDict['profile']['param'][c1i][d_i] - dataDict['profile']['param'][c2i][d_i]) for d_i in range(nDepths)]) #I checked and there are minimal changes from doing it this way versus doing error prop
                    
                if statTestType == 't-test':
                    #one-sample two-sided t-test
                    #tests against the null hypothesis that there is no change between conditions
                    tt = np.asarray(diff_avg/(stdev/np.sqrt(dataDict['profile']['nVox']))) #T statistic
                    pvals = stats.t.sf(np.abs(tt), np.asarray(dataDict['profile']['nVox'])-1)*2 #2-sided t-test returns p-value
                elif statTestType == 'permutation':
                    ##!!!WARNING!!! Permutation test does not work with current implementation of makeProfile1D (11/05/2024)
                    averages = dataDict['profile']['avg']
                    pvals = permute_1samp(averages, np.mean, null_stat=0, n_permutations=num_permutations, test_type='two-sided', axis=0)
                else:
                    print("Invalid statTestType type!!! Using 't-test' as default.")
                    tt = np.asarray(np.asarray(dataDict['profile']['avg'])/(np.asarray(dataDict['profile']['std'])/np.sqrt(dataDict['profile']['nVox']))) #T statistic
                    pvals = stats.t.sf(np.abs(tt), np.asarray(dataDict['profile']['nVox'])-1)*2 #2-sided t-test returns p-value
                
                #save
                keep_diffs[diff_label]['depths'].append(dataDict['profile']['depth'])
                keep_diffs[diff_label]['N'].append(dataDict['profile']['nVox'])
                keep_diffs[diff_label]['avg'].append(diff_avg)
                keep_diffs[diff_label]['stdev'].append(stdev)
                keep_diffs[diff_label]['p-vals'].append(pvals)
                print(diff_label)
    
        elif profile_method == 'smooth':
            for diff_label in diffDetails.keys():
                cond1 = diffDetails[diff_label][0]
                cond2 = diffDetails[diff_label][1]
                c1i = np.where(np.array(statDetails['labels']) == cond1)[0][0]
                c2i = np.where(np.array(statDetails['labels']) == cond2)[0][0]
                
                profile1, depth1 = smoothen(roi[statDetails['labels']].values[:, c1i],
                                              roi[depthParam].values,
                                              radMax,
                                              nRadii=nDepths,
                                              kernel=smooth_kernel,
                                              smooth_factor=smooth_factor)
                profile2, depth2 = smoothen(roi[statDetails['labels']].values[:, c2i],
                                              roi[depthParam].values,
                                              radMax,
                                              nRadii=nDepths,
                                              kernel=smooth_kernel,
                                              smooth_factor=smooth_factor)
                diff_avg = profile1 - profile2
                
                #save
                keep_diffs[diff_label]['depths'].append(depth1)
                keep_diffs[diff_label]['avg'].append(diff_avg)
                print(diff_label)
                
    return(keep_diffs)
        

def depth_deconv(keep_rois,p2t_model,Nbins_model,Nbins,normalize_psf):
    '''
    Deconvolution across depth following the method in Markuerkiaga et al. (2021)
    
    Inputs:
        keep_rois (array-like): depth profiles for ROIs 
            (NROIs x Nconditions x Ndepths)
        p2t_model (float): peak-to-tail ratio used to set point spread 
            functions over depth
        Nbins_model (int): number of depth bins for the desing matrix
        Nbins (int): number of depth bins in data
        normalize_psf (bool): normalize the psf by the deepest layer if True
        
    Outputs:
        decon_rois (array-like): deconvolved depth profiles for ROIs
            (NROIs x Nconditions x Nbins_model)
            
    Dependencies:
        numpy
    '''
    
    p2t = p2t_model * Nbins/Nbins_model + (Nbins_model - Nbins)/(2*Nbins_model); #adjusted peak to tail ratio for number of bins
    psf = np.tril((1/p2t)*np.ones([Nbins,Nbins]))
    psf[np.diag(np.diag(np.ones([Nbins,Nbins]))) == 1] = 1 #unnormalized point spread functions

    
    decon_rois = []
    for iSubj in range(len(keep_rois)):
        decon_rois.append([])
        for iStim in range(len(keep_rois[iSubj])):
            decon_rois[iSubj].append([])
            coef = np.array(keep_rois[iSubj][iStim])
            if normalize_psf:
                norm_coef = coef[0]
                psf = np.tril((norm_coef/p2t)*np.ones([Nbins,Nbins]))
                psf[np.diag(np.diag(np.ones([Nbins,Nbins]))) == 1] = norm_coef #unnormalized point spread functions
            
            #solve GLM
            # I think grouping is wront in this one 
            #beta = np.dot(np.linalg.inv(np.dot(psf.T,psf)),np.dot(psf.T,coef_avg))
            
            # Trying it the way I figured out below for estimating modulation
            psf = np.matrix(psf)
            coef = np.matrix(coef).transpose()
            beta_deconvolved = psf.I*coef
            beta_deconvolved = [np.array(beta_deconvolved)[iV, 0] for iV in range(beta_deconvolved.shape[0])]
            decon_rois[iSubj][iStim] = beta_deconvolved
            
    return decon_rois


def compute_avg_depth_profile(rois,statDetails,diffDetails,plotStats,plotDiffs,use_decon,prop_err,useSI,statTestType='t-test',num_permutations=10000):
    '''
    Calculate average depth profiles across ROIs
    
    Inputs:
        rois (dict): roi profiles with the following information
            1) average depth profiles for each condition (NROIs x Ndepths)
            2) standard deviation of depth profiles for each condition (NROIs x Ndepths)
            3) depths of each bin (NROIs x Ndepths)
        statDetails (dict): dictionary of condition details including labels 
            and colors for each condition
        diffDetails (dict): dictionary of the difference profiles and 
            corresponding stats to use to calculate them
        plotStats (list): list of labels for stats to plot
        plotDiffs (list): list of labels for diffs to plot
        use_decon (bool): use deconvolved profiles if True
        prop_err (bool): propagate errors from individual profiles if true
        useSI (bool): use suppression index rather than difference profile
        
    Outputs:
        stat_avg (dict): average depth values for each condition which 
            contains the following information:
            1) average profile over all ROIs (Ndepths)
            2) standard deviation over all ROIs (Ndepths)
            3) normalized depths (Ndepths)
        diff_avg (dict): average depth difference profile which contains the 
            following information:
            1) average different profile over all ROIs (Ndepths)
            2) standard deviation over all ROIs (Ndepths)
            3) normalized depths (Ndepths)
            4) results of a single-sample two-sided t-test computed at each 
                depth against the null of diff = 0 (Ndepths)
            
        Dependencies:
            numpy, matplotlib, scipy
        
    '''
    
    stat_avg = {stat: {'avg': [], 'stdev': [], 'norm_depths': []} for stat in plotStats}
    diff_avg = {diff: {'avg': [], 'stdev': [], 'norm_depths': [], 'p-vals': []} for diff in plotDiffs}
    
    for iStat, stat in enumerate(plotStats):
        
        #average profile
        if use_decon:
            stat_avg[stat]['avg'] = np.nanmean(np.asarray(rois[stat]['avg_decon']), axis=0)
        else:
            stat_avg[stat]['avg'] = np.nanmean(np.asarray(rois[stat]['avg']), axis=0)
        
        #normalized depths
        depth_avg = np.nanmean(np.asarray(rois[stat]['depths']), axis=0)
        if not isinstance(depth_avg, Iterable) or isinstance(depth_avg, (str, bytes)):
            depth_avg = np.array([depth_avg]) #turn into iterable
        stat_avg[stat]['norm_depths'] = np.linspace(0,1,len(depth_avg)) #(depth_avg - np.min(depth_avg))/(np.max(depth_avg) - np.min(depth_avg)) #normalized depth
       
        #standard deviation
        if prop_err:
            stat_avg[stat]['stdev'] = np.sqrt(np.nansum(np.asarray(rois[stat]['stdev'])**2,axis=0))/np.shape(rois[stat]['stdev'])[0]
        else:
            stat_avg[stat]['stdev'] = np.nanstd(np.asarray(rois[stat]['avg']), axis=0)
            
        #number of samples
        stat_avg[stat]['Nsamp'] = np.shape(rois[stat]['avg'])[0]
        
        #one-sample two-sided t-test
        if statTestType == 't-test':
            stat_avg[stat]['p-vals'] = stats.ttest_1samp(rois[stat]['avg'],0,axis=0)
        elif statTestType == 'permutation':
            averages = np.vstack(rois[stat]['avg'])
            stat_avg[stat]['p-vals'] = permute_1samp(averages, np.mean, null_stat=0, n_permutations=num_permutations, test_type='two-sided', axis=0)
        else:
            print("Invalid statTestType type!!! Using 't-test' as default.")
            stat_avg[stat]['p-vals'] = stats.ttest_1samp(rois[stat]['avg'],0,axis=0)
            
    for iDiff, diff in enumerate(plotDiffs):
        
        #get conditions to subtract
        cond1 = diffDetails[diff][0]
        cond2 = diffDetails[diff][1]
        
        #calculate diffs
        if use_decon:
            diff_all = np.asarray(rois[cond1]['avg_decon'])-np.asarray(rois[cond2]['avg_decon'])
        else:
            diff_all = np.asarray(rois[cond1]['avg'])-np.asarray(rois[cond2]['avg'])
        diff_avg[diff]['avg'] = np.nanmean(diff_all,axis=0)
        
        #normalized depths
        depth_avg = np.nanmean(np.asarray(rois[cond1]['depths']), axis=0)
        if not isinstance(depth_avg, Iterable) or isinstance(depth_avg, (str, bytes)):
            depth_avg = np.array([depth_avg]) #turn into iterable
        diff_avg[diff]['norm_depths'] = np.linspace(0,1,len(depth_avg)) #(depth_avg - np.min(depth_avg))/(np.max(depth_avg) - np.min(depth_avg)) #normalized depth
        
        #standard deviation
        if prop_err:
            diff_avg[diff]['stdev'] = np.sqrt(np.nansum(np.vstack([stat_avg[cond1]['stdev']**2,stat_avg[cond2]['stdev']**2]),axis=0))
        else:
            diff_avg[diff]['stdev'] = np.std(diff_all,axis=0)
            
        #number of samples
        diff_avg[diff]['Nsamp'] = np.shape(rois[cond1]['avg'])[0]
            
        #one-sample two-sided t-test
        if statTestType == 't-test':
            diff_avg[diff]['p-vals'] = stats.ttest_1samp(diff_all,0,axis=0)
        elif statTestType == 'permutation':
            averages = np.vstack(diff_all)
            diff_avg[diff]['p-vals'] = permute_1samp(averages, np.mean, null_stat=0, n_permutations=num_permutations, test_type='two-sided', axis=0)
        else:
            print("Invalid statTestType type!!! Using 't-test' as default.")
            diff_avg[diff]['p-vals'] = stats.ttest_1samp(diff_all,0,axis=0)
        
    
    return(stat_avg,diff_avg)

def plot_avg_depth_profile(p1,avgProfiles,Stats,Colors,ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=None,use_decon=True,showSig=False,pthresh=0.05,statCorrType='none',xticks=None,yticks=None):
    '''
    Plot average depth profile for a given set of conditions
    
    Inputs:
        p1 (matplotlib handle): figure handle
        avgProfiles (dict): a dictionary of average profiles for each condition
            should contain:
            1) averages
            2) stdevs
            3) normalized depths
        Stats (array-like): list or array of strings indicating stats to plot
        Colors (array-like): list or array of colors for each stat
        ylim (array-like): list or array of y limits (min and max)
        xlim (array-like): list or array of x limits (min and max)
        dx (float): x location of legend
        dy (float): y location of legend
        Ntext (array-like): list or array with coordinates for location of N 
            text
        lcolor (str or tuple): color of lines
        fsize (float): title fontsize (all other fontsizes are based off of this)
        plot_indiv (dict): overlay plots of individual profiles; should contain
            entries for each stat entered
            DEFAULT = None
        use_decon (bool): If true use deconvolved profiles (only if plotting 
            individual profiles)
            DEFAULT = True
        xticks (array-like): Manually specified xticks
        yticks (array-like): Manually specified yticks
        
    Outputs:
        (None)
        
    Dependencies:
        numpy, matplotlib

    '''
    
    for iStat, stat in enumerate(Stats):
        p1.plot(avgProfiles[stat]['avg'], avgProfiles[stat]['norm_depths'], color=Colors[iStat])
        p1.fill_betweenx(avgProfiles[stat]['norm_depths'],
                        avgProfiles[stat]['avg'] - avgProfiles[stat]['stdev']/np.sqrt(avgProfiles[stat]['Nsamp']),
                        avgProfiles[stat]['avg'] + avgProfiles[stat]['stdev']/np.sqrt(avgProfiles[stat]['Nsamp']),
                        linewidth=0.,
                        alpha=0.4,
                        color=Colors[iStat])
        
        if showSig:
            if isinstance(statCorrType,Iterable) and not isinstance(statCorrType,str): #custom corrected p-values
                top = avgProfiles[stat]['avg'] + avgProfiles[stat]['stdev']/np.sqrt(np.shape(avgProfiles[stat]['avg'])[0])
                corrected_pvalues = statCorrType[stat]
                p1.plot(top[corrected_pvalues <= pthresh] + 0.1,avgProfiles[stat]['norm_depths'][corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None',markersize=fsize-4)
            elif isinstance(statCorrType,str):
                if statCorrType == 'none':
                    top = avgProfiles[stat]['avg'] + avgProfiles[stat]['stdev']/np.sqrt(np.shape(avgProfiles[stat]['avg'])[0])
                    p1.plot(top[avgProfiles[stat]['p-vals'].pvalue <= pthresh] + 0.1,avgProfiles[stat]['norm_depths'][avgProfiles[stat]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None',markersize=fsize-4)
                else:
                    top = avgProfiles[stat]['avg'] + avgProfiles[stat]['stdev']/np.sqrt(np.shape(avgProfiles[stat]['avg'])[0])
                    corrected_pvalues = multipletests(avgProfiles[stat]['p-vals'].pvalue,method=statCorrType)[1]
                    p1.plot(top[corrected_pvalues <= pthresh] + 0.1,avgProfiles[stat]['norm_depths'][corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None',markersize=fsize-4)
                    
            else:
                print("plot_avg_depth_profiles: Improper Formatting of statCorrType Argument")
        
        
        p1.text(dx, dy + iStat*.07, stat,
                color=Colors[iStat],
                fontsize=fsize-1)
        p1.set_ylim(ylim)
        p1.set_xlim(xlim)
        p1.set_xlabel('BOLD % change', fontsize=fsize, color=lcolor)
        p1.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
        p1.text(Ntext[0], Ntext[1], 'n=%d hemis' %avgProfiles[stat]['Nsamp'], color=lcolor, fontsize=fsize-1, fontstyle='italic')
        p1.tick_params(axis='both', which='major', labelsize=fsize-1)
        if xticks:
            p1.set_xticks(ticks=xticks)
        if yticks:
            p1.set_yticks(ticks=yticks)
        
        if isinstance(plot_indiv,dict):
            print("plotting individual profiles")
            if use_decon:
                p1.plot(plot_indiv[stat]['avg_decon'].T, np.tile(avgProfiles[stat]['norm_depths'],[avgProfiles[stat]['Nsamp'],1]).T, color=Colors[iStat],alpha=0.3)
            else:
                p1.plot(plot_indiv[stat]['avg'].T, np.tile(avgProfiles[stat]['norm_depths'],[avgProfiles[stat]['Nsamp'],1]).T, color=Colors[iStat],alpha=0.3)

    return()

def plot_avg_diff_profile(p2,avgDiffs,Diffs,Colors,ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=None,use_decon=True,showSig=False,pthresh=0.05,statCorrType='none'):
    '''
    Plot average difference profiles across depth

    Inputs:
        p2 (matplotlib handle): figure handle
        avgDiffs (dict): a dictionary of average profiles for each condition
            should contain:
            1) averages
            2) stdevs
            3) normalized depths
            4) p-values
        Stats (array-like): list or array of strings indicating stats to plot
        Colors (array-like): list or array of colors for each stat
        ylim (array-like): list or array of y limits (min and max)
        xlim (array-like): list or array of x limits (min and max)
        dx (float): x location of legend
        dy (float): y location of legend
        Ntext (array-like): list or array with coordinates for location of N 
            text
        lcolor (str or tuple): color of lines
        fsize (float): title fontsize (all other fontsizes are based off of this)
        useSI (bool): use suppression index in difference
        plot_indiv (dict): overlay plots of individual profiles; should contain
            entries for each stat entered
            DEFAULT = None
        use_decon (bool): If true use deconvolved profiles (only if plotting 
            individual profiles)
            DEFAULT = True
        showSig (bool): indicate significance if true
            DEFAULT - false
        pthresh (float): p-value threshold if showSig is true
            DEFAULT - 0.05
        statTestType (str): the type of statistical correction for multiple 
            comparisons for the diff profile agaisnt the profile to the null (0)
            DEFAULT - 'none'
        
    Outputs:
        (None)
        
    Dependencies:
        numpy, matplotlib

    '''
    
    p2.plot([0, 0], [0, 1], '--', color='gray')

    for iDiff, diff in enumerate(Diffs):
        p2.plot(avgDiffs[diff]['avg'], avgDiffs[diff]['norm_depths'], color=Colors[iDiff])
        p2.fill_betweenx(avgDiffs[diff]['norm_depths'],
                        avgDiffs[diff]['avg'] - avgDiffs[diff]['stdev']/np.sqrt(avgDiffs[diff]['Nsamp']),
                        avgDiffs[diff]['avg'] + avgDiffs[diff]['stdev']/np.sqrt(avgDiffs[diff]['Nsamp']),
                        linewidth=0.,
                        alpha=0.4,
                        color=Colors[iDiff])
        
        if showSig:
            if isinstance(statCorrType,Iterable) and not isinstance(statCorrType,str): #custom corrected p-values
                top = avgDiffs[diff]['avg'] + avgDiffs[diff]['stdev']/np.sqrt(np.shape(avgDiffs[diff]['avg'])[0])
                corrected_pvalues = statCorrType
                p2.plot(top[corrected_pvalues <= pthresh] + 0.1,avgDiffs[diff]['norm_depths'][corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None',markersize=fsize-4)
            elif isinstance(statCorrType,str):
                if statCorrType == 'none':
                    top = avgDiffs[diff]['avg'] + avgDiffs[diff]['stdev']/np.sqrt(np.shape(avgDiffs[diff]['avg'])[0])
                    p2.plot(top[avgDiffs[diff]['p-vals'].pvalue <= pthresh] + 0.1,avgDiffs[diff]['norm_depths'][avgDiffs[diff]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None',markersize=fsize-4)
                else:
                    top = avgDiffs[diff]['avg'] + avgDiffs[diff]['stdev']/np.sqrt(np.shape(avgDiffs[diff]['avg'])[0])
                    corrected_pvalues = multipletests(avgDiffs[diff]['p-vals'].pvalue,method=statCorrType)[1]
                    p2.plot(top[corrected_pvalues <= pthresh] + 0.1,avgDiffs[diff]['norm_depths'][corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None',markersize=fsize-4)
            else:
                print("plot_avg_diff_profiles: Improper Formatting of statCorrType Argument")
            
        p2.set_ylim(ylim)
        p2.set_xlim(xlim)
        p2.text(1.15, .05+iDiff*0.1, diff, color=Colors[iDiff], fontsize=fsize-1)
        p2.set_yticklabels([])
        p2.tick_params(axis='both', which='major', labelsize=fsize-1)
        if useSI:
            p2.set_xlabel(r'$\Delta$ SI', fontsize=fsize, color=lcolor)
        else:
            p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)
            
        if isinstance(plot_indiv,dict):
            print("plotting individual profiles")
            if use_decon:
                p2.plot(np.array(plot_indiv[diff]['avg_decon']).T, np.tile(avgDiffs[diff]['norm_depths'],[avgDiffs[diff]['Nsamp'],1]).T, color=Colors[iDiff],alpha=0.3)
            else:
                p2.plot(np.array(plot_indiv[diff]['avg']).T, np.tile(avgDiffs[diff]['norm_depths'],[avgDiffs[diff]['Nsamp'],1]).T, color=Colors[iDiff],alpha=0.3)
            
def get_lmnv(df,key='stdev_xerrts'):
    """
    Returns the log-mean-normalized-variance (log(MNV)) of a dataframe.
    """
    mnv = df[key].values ** 2
    lmnv = np.log(mnv)
    return lmnv

def get_deep_layer_dist(df, depth_var, deep_pct):
    """
    Returns the deep layer distribution of a dataframe based on a given depth variable
    and percentile.
    """
    z = df[depth_var]
    deep = z <= np.percentile(z, deep_pct)
    lmnv = get_lmnv(df[deep])
    lmnv = lmnv[(lmnv != np.inf) & (lmnv != -np.inf)] #sometimes there are infinities from division by zero. I'll ignore these.
    deep_mean = np.mean(lmnv)
    deep_std = np.std(lmnv)
    return deep_mean, deep_std, deep

def get_mnv_mask(df, depth_var, deep_pct, sd_thresh):
    """
    Returns a mask for a given dataframe based on a deep layer distribution and
    standard deviation threshold.
    """
    lmnv = get_lmnv(df)
    deep_mean, deep_std, deep = get_deep_layer_dist(df, depth_var, deep_pct)
    lmnv_thresh = deep_mean + sd_thresh * deep_std
    return lmnv < lmnv_thresh, lmnv_thresh

def plot_mnv_histograms(lmnv, deep_lmnv, mnv_mask, deep_pct, key, k_i, NROIs, fsize, pad=0.1, figsize=(6.5,6.5), fname="thresh"):
    """
    Plots histograms for a given dataframe.
    """
    fthresh = plt.figure(fname,figsize=figsize)
    
    # Plot on even-numbered rows
    rows = 2 * int(np.ceil(np.sqrt(NROIs)))  # Double the number of rows
    cols = int(np.ceil(np.sqrt(NROIs)))
    
    # Adjust k_i to plot on even rows (0, 2, 4,...)
    index = 2 * cols * (k_i // cols) + (k_i % cols) + 1
    plt.subplot(cols,rows,index)
    #plt.subplot(2, NROIs, (k_i + 1))
    plt.hist(lmnv, bins=np.linspace(0, 10, 200), density=True, alpha=0.5)
    plt.hist(deep_lmnv, bins=np.linspace(0, 10, 200), density=True, alpha=0.5)
    plt.xlabel("log(MNV)", fontsize=0.7 * fsize)
    plt.ylabel("Density (voxels/bin len)", fontsize=0.7 * fsize)
    plt.legend(['full', 'deepest %d%%' % (deep_pct)], fontsize=0.7 * fsize)
    plt.xticks(fontsize=0.5 * fsize)
    plt.yticks(fontsize=0.5 * fsize)
    plt.title(key, fontsize=fsize)
    
    # Plot on odd-numbered rows (1, 3, 5, ...)
    index += cols
    plt.subplot(cols,rows,index)
    #plt.subplot(2, NROIs, (k_i + 1) + NROIs)
    plt.hist(lmnv[mnv_mask], bins=np.linspace(0, 10, 200), density=True, alpha=0.5)
    plt.xlabel("log(MNV)", fontsize=0.7 * fsize)
    plt.ylabel("Density (voxels/bin len)", fontsize=0.7 * fsize)
    plt.legend(['masked'], fontsize=0.7 * fsize)
    plt.xticks(fontsize=0.5 * fsize)
    plt.yticks(fontsize=0.5 * fsize)
    plt.title(key, fontsize=fsize)
    
    #fthresh.tight_layout(pad=pad)
    return(fthresh)


def plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, vlim, fsize, fname='dmap', mask=None, pad=0.1, figsize=(6.5,6.5)):
    """
    Plots depth maps for a given dataframe.
    """
    if isinstance(mask, np.ndarray) or isinstance(mask, list):
        mnv_mask = mask
    else:
        mnv_mask = np.ones(np.shape(mnv),dtype=bool)
    x = df[x_var]
    y = df[y_var]
    Ngroups = len(depth_groups.keys())
    fdepth = plt.figure(fname, figsize=figsize)
    for d_i, depth_label in enumerate(depth_labels):
        level_mask = (df[depth_var] >= depth_groups[depth_label][0]) & (df[depth_var] <= depth_groups[depth_label][1])
        plt.subplot(Ngroups*2, NROIs, (k_i+1)+2*d_i*NROIs)
        plt.scatter(x[level_mask*mnv_mask], y[level_mask*mnv_mask], s=0.5, c=np.log(mnv[level_mask*mnv_mask]), cmap='Reds', vmin=vlim[0], vmax=vlim[1])
        if k_i == NROIs-1:
            plt.colorbar(label='log(MNV)')
        plt.title(f"{depth_label}", fontsize=fsize)
        plt.xlabel('U', fontsize=0.7*fsize)
        plt.ylabel('V', fontsize=0.7*fsize)
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,        # ticks along left side are off
            right=False,     #ticks along right side are off
            labelbottom=False,      # tick labels along the bottom edge are off
            labeltop=False,         # tick labels along the top edge are off
            labelleft=False,        # tick labels along left side are off
            labelright=False)       #tick labels along right side are off
        # plt.xticks(fontsize=0.5*fsize)
        # plt.yticks(fontsize=0.5*fsize)
        plt.gca().set_aspect('equal')
        plt.show()
        
        #plot hist
        plt.subplot(Ngroups*4,NROIs,(k_i+1)+(2*NROIs)+(4*d_i*NROIs))
        plt.hist(np.log(mnv[level_mask*mnv_mask]),bins=np.linspace(2,5,100))
        plt.xticks(ticks = None, labels = None, fontsize=0.5*fsize)
        plt.yticks(ticks = None, labels = None, fontsize=0.5*fsize)
        plt.xlabel("log(MNV)",fontsize=0.7*fsize)
    
    #fdepth.tight_layout(pad=pad)
    return(fdepth)

def plot_depth_voxel_loss(z, mnv_mask, nDepths, NROIs, key, k_i, fsize, pad=0.1, figsize=(15,3), fname = "depth"):
    """
    Plots voxel loss at each depth after masking.
    
    Parameters:
    z (numpy.ndarray): array of depths.
    mnv_mask (numpy.ndarray): binary mask indicating voxels to include.
    nDepths (int): number of depth bins.
    key (str): title for the plot.
    k_i (int): subplot index.
    fsize (int): font size for plot labels.
    """
    
    frac_included = np.zeros((int(nDepths),))
    depthBoundaries = np.linspace(np.min(z),np.max(z),nDepths)

    dmasked = z[mnv_mask]
    bin_count = np.bincount(np.digitize(z,depthBoundaries,right=True))
    bin_count_masked = np.bincount(np.digitize(dmasked,depthBoundaries,right=True))
    frac_included = bin_count_masked/bin_count
    
    fdepth_hist = plt.figure(fname,figsize=figsize)
    
    # Plot on even-numbered rows
    rows = 2 * int(np.ceil(np.sqrt(NROIs)))  # Double the number of rows
    cols = int(np.ceil(np.sqrt(NROIs)))
    
    # Adjust k_i to plot on even rows (0, 2, 4,...)
    index = 2 * cols * (k_i // cols) + (k_i % cols) + 1
    plt.subplot(cols,rows,index)
    plt.hist(z,bins=np.linspace(np.min(z),np.max(z),nDepths+1))
    plt.title(key+"\n Depth Hist",fontsize=fsize)
    plt.xticks(ticks = None, labels = None, fontsize=0.5*fsize)
    plt.xlim([0,1])
    plt.ylim([0,1.3*np.max(bin_count)])
    plt.yticks(ticks = None, labels = None, fontsize=0.5*fsize)
    plt.hist(z[mnv_mask],bins=np.linspace(np.min(z),np.max(z),nDepths+1),alpha=0.7)
    plt.legend(["unmasked","masked"],fontsize=0.7*fsize)
    
    # Plot on odd-numbered rows (1, 3, 5, ...)
    index += cols
    plt.subplot(cols,rows,index)
    plt.bar(np.linspace(np.min(z)+np.max(z)/(2*nDepths),np.max(z)-np.max(z)/(2*nDepths),nDepths),frac_included,width=np.max(z)/nDepths,color='tomato')
    plt.xticks(ticks = None, labels = None, fontsize=0.5*fsize)
    plt.yticks(ticks = None, labels = None, fontsize=0.5*fsize)
    plt.xlabel("Depth (WM --> Pia)",fontsize=0.7*fsize)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("Frac Included",fontsize=0.7*fsize)
    
    fdepth_hist.tight_layout(pad=pad)
    return(fdepth_hist)

def plot_centroids(all_data, mask_dict, statDetails, roiRad=2, nDepths=10, pad=0.0, figsize=(6.5,6.5), xlim=[-5,30], radParam = "scale_xy_dist"):
    for iStat in range(len(statDetails['labels'])):
        plt.figure(statDetails['labels'][iStat],figsize=figsize)
        for iR, label in enumerate(all_data.keys()):
            df = all_data[label]
            if np.sum(mask_dict[label]) == 0:
                print(f"Skipping ROI {label} as it does not exist in the DataFrame")
                continue
            roi_idx = mask_dict[label] == 1
            roi = df.loc[roi_idx]
            if radParam != None:
                roi = df[df[radParam] < roiRad] # only very center, to be sure!
                roi = roi[roi[radParam] >= 0] #don't allow negative radii   
            maxDepth = np.max(roi['d'])
            minDepth = np.min(roi['d'])
            normDepths = (roi['d']-minDepth)/(maxDepth-minDepth)
            dataDict = makeProfile1D(roi['d'].values,
                                 nDepths, #number of depths
                                 roi[statDetails['labels']].values,
                                 np.min(roi['d'].values), #min depth value
                                 np.max(roi['d'].values), #max depth value
                                 True) #Use LayNii values
            avg_normDepths = (dataDict['profile']['depth'] - np.nanmin(dataDict['profile']['depth']))/(np.nanmax(dataDict['profile']['depth']) - np.nanmin(dataDict['profile']['depth']))
        
            #plot
            plt.subplot(2,int(np.ceil(len(all_data.keys())/2.)), 1 + iR)
            plt.scatter(roi[statDetails['labels'][iStat]],normDepths,s=0.5,c=statDetails['colors'][iStat])
            plt.plot(dataDict['profile']['avg'][iStat],avg_normDepths,color=statDetails['colors'][iStat])
            plt.ylim([-0.02, 1.02])
            plt.xlim(xlim)
            plt.title(label,fontsize = 8)
            plt.tick_params(labelsize = 6)
        plt.show()
        plt.tight_layout(pad=pad)

def plot_centroids_diff(all_data, mask_dict, statDetails, diffDetails, roiRad=2, nDepths=10, pad=0.0, figsize=(6.5,6.5), radParam = 'scale_xy_dist'):
    for iDiff, Diff in enumerate(diffDetails['statIDs'].keys()):
        plt.figure(Diff,figsize=figsize)
        for iR, label in enumerate(all_data.keys()):
            df = all_data[label]
            if np.sum(mask_dict[label]) == 0:
                print(f"Skipping ROI {label} as it does not exist in the DataFrame")
                continue
            roi_idx = mask_dict[label] == 1
            roi = df.loc[roi_idx]
            if radParam != None:
                roi = df[df[radParam] < roiRad] # only very center, to be sure!
                roi = roi[roi[radParam] >= 0] #don't allow negative radii    
            maxDepth = np.max(roi['d'])
            minDepth = np.min(roi['d'])
            normDepths = (roi['d']-minDepth)/(maxDepth-minDepth)
            dataDict = makeProfile1D(roi['d'].values,
                                 nDepths, #number of depths
                                 roi[statDetails['labels']].values,
                                 np.min(roi['d'].values), #min depth value
                                 np.max(roi['d'].values), #max depth value
                                 True) #Use LayNii values
            avg_normDepths = (dataDict['profile']['depth'] - np.nanmin(dataDict['profile']['depth']))/(np.nanmax(dataDict['profile']['depth']) - np.nanmin(dataDict['profile']['depth']))

            #compute differences
            statIDs = diffDetails['statIDs'][Diff]
            cond1 = statIDs[0]
            cond2 = statIDs[1]
            c1i = np.where(np.array(statDetails['labels']) == cond1)[0][0]
            c2i = np.where(np.array(statDetails['labels']) == cond2)[0][0]
            diff = roi[statDetails['labels'][c1i]] - roi[statDetails['labels'][c2i]]
            diff_avg = np.array(dataDict['profile']['avg'][c1i]) - np.array(dataDict['profile']['avg'][c2i])

            #plot
            plt.subplot(2,int(np.ceil(len(all_data.keys())/2.)), 1 + iR)
            plt.scatter(diff,normDepths,s=0.5,c=diffDetails['colors'][iDiff])
            plt.plot(diff_avg,avg_normDepths,color=diffDetails['colors'][iDiff])
            plt.ylim([-0.02, 1.02])
            plt.xlim([-5,10])
            plt.title(label,fontsize=8)
            plt.tick_params(labelsize=6)
        plt.tight_layout(pad=pad)
    plt.show()

def plot_flat_patches(all_data, cond, pthresh, cmax, red, fcolor, fontsize, lcolor, xlims=None, ylims=None, xvar = 'x', yvar = 'y', pval_var = 'task p-val'):
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        if pval_var in df.columns:
            df[pval_var] = df[pval_var]
            all_data[label] = df

        # Plot iso
        if not isinstance(xlims,np.ndarray) or not isinstance(xlims,list):
            xlims = [np.min(df[xvar].values), np.max(df[xvar].values)]
        if not isinstance(ylims,np.ndarray) or not isinstance(ylims,list):
            ylims = [np.min(df[yvar].values), np.max(df[yvar].values)]
        aw = .8/len(all_data)
        ax = plt.axes([.07 + iR*1.1*aw, .5,  aw, .26])
        for iV, beta in enumerate(df[cond]):
            if df[pval_var][iV] < pthresh:
                weight = np.min((np.max((beta, -cmax)), cmax))/cmax
                weight = (weight + 1)/2; #make value between 0 and 1
                if weight > 0:
                    color = weight*red
                else:
                    color = weight*red
                ax.plot(df[xvar][iV], df[yvar][iV], '.', color=color)
        ax.set_xlim([xlims[0], xlims[1]])
        ax.set_ylim([ylims[0], ylims[1]])
        ax.patch.set_facecolor(fcolor)
        plt.axis('off')
        ax.set_title(label, fontsize=fontsize, color=lcolor)

def plot_smooth_patches(all_data, cond, pthresh, cmax, smooth_kernel2D, smooth_factor, resamp_factor, xlims=None, ylims=None, xvar='x', yvar='y', pval_var = 'task p-val'):
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]

        # resample and smooth beta values
        if not isinstance(xlims,np.ndarray) or not isinstance(xlims,list):
            xlims = [np.min(df[xvar].values), np.max(df[xvar].values)]
        if not isinstance(ylims,np.ndarray) or not isinstance(ylims,list):
            ylims = [np.min(df[yvar].values), np.max(df[yvar].values)]
        x_interp = np.linspace(xlims[0], xlims[1], resamp_factor)
        y_interp = np.linspace(ylims[0], ylims[1], resamp_factor)
        #beta_interp = sciInterp.interp2d(df[xvar].values, df[yvar].values, df[cond].values, kind='linear')
        betas = df[cond].values[df[pval_var] < pthresh]
        betas_x = df[xvar].values[df[pval_var] < pthresh]
        betas_y = df[yvar].values[df[pval_var] < pthresh]
        beta_resamp = smoothen2D(betas,np.array([betas_x,betas_y]),np.array([x_interp,y_interp]),smooth_kernel2D,smooth_factor)

        # plot
        plt.subplot(1, len(all_data), iR+1)
        plt.imshow(beta_resamp, cmap='Reds')
        plt.colorbar()
        plt.title(label+': '+cond)

    plt.show()
    
def plot_profile_comparisons(p2,avgs,profiles,Diffs,Colors,LineStyles,ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=False,pthresh=0.05,statCorrType='none',show_comp=[0,1]):
    '''
    Compare depth profiles

    Inputs:
        p2 (matplotlib handle): figure handle
        avgs (dict): precomputed averages which carries forward any 
            parameters from compute_all_depth_profiles (could be implemented 
            here, but it would make things more complicated)
        profiles (dict): a dictionary containing the depth profiles for 
            each ROI (only used for computing t-stat)
        Stats (array-like): list or array of strings indicating stats to plot
        Colors (array-like): list or array of colors for each stat
        LineStyles (array-like): linestyles of each for each stat
        ylim (array-like): list or array of y limits (min and max)
        xlim (array-like): list or array of x limits (min and max)
        dx (float): x location of legend
        dy (float): y location of legend
        Ntext (array-like): list or array with coordinates for location of N 
            text
        lcolor (str or tuple): color of lines
        fsize (float): title fontsize (all other fontsizes are based off of this)
        useSI (bool): use suppression index in difference
        showSig (bool): indicate significance if true
            DEFAULT - false
        pthresh (float): p-value threshold if showSig is true
            DEFAULT - 0.05
        statTestType (str): the type of statistical correction for multiple 
            comparisons for the diff profile agaisnt the profile to the null (0)
            DEFAULT - 'none'
        show_comp (array-like): a list of indicies for the conditions to 
            compare. You can only plot a miximum of comparisons between two 
            conditions.
            DEFAULT - [0,1]
        
    Outputs:
        (None)
        
    Dependencies:
        numpy, matplotlib, stats, statsmodels

    '''
    
    p2.plot([0, 0], [0, 1], '--', color='gray')
    
    # compute 2-sample t-test
    T, D = np.shape(profiles[Diffs[0]]['avg'])
    profiles_arr = np.zeros([len(Diffs),D,T])
    for iDiff, diff in enumerate(Diffs):
        profiles_arr[iDiff,:,:] = np.transpose(profiles[diff]['avg'])
    t_statistics, p_values = compare_conditions(profiles_arr,Diffs)
            
    top_arr = np.zeros([len(Diffs),D]) #set up array to hold the max values for each depth
    for iDiff, diff in enumerate(Diffs):
        p2.plot(avgs[diff]['avg'], avgs[diff]['norm_depths'], color=Colors[iDiff], linestyle=LineStyles[iDiff], label=diff)
        p2.fill_betweenx(avgs[diff]['norm_depths'],
                        avgs[diff]['avg'] - avgs[diff]['stdev']/np.sqrt(avgs[diff]['Nsamp']),
                        avgs[diff]['avg'] + avgs[diff]['stdev']/np.sqrt(avgs[diff]['Nsamp']),
                        linewidth=0.,
                        alpha=0.4,
                        color=Colors[iDiff],
                        label=None)
        
        p2.set_ylim(ylim)
        p2.set_xlim(xlim)
        #p2.text(1.15, .05+iDiff*0.1, diff, color=Colors[iDiff], fontsize=fsize-2)
        p2.set_yticklabels([])
        if useSI:
            p2.set_xlabel(r'$\Delta$ SI', fontsize=fsize, color=lcolor)
        else:
            p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)
            
        if showSig:
            top_arr[iDiff, :] = avgs[diff]['avg'] + avgs[diff]['stdev']/np.sqrt(avgs[diff]['Nsamp'])
            
    p2.legend(loc = 'upper right',fontsize=0.6*fsize)
    
    top = np.max(top_arr,axis=0) #get the top values for plotting significance indicator
    if showSig:
        if isinstance(statCorrType,Iterable) and not isinstance(statCorrType,str): #custom corrected p-values
            corrected_pvalues = statCorrType
            p2.plot(top[corrected_pvalues <= pthresh] + 0.1,avgs[diff]['norm_depths'][corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None',label=None)
        elif isinstance(statCorrType,str):
            if statCorrType == 'none':
                p2.plot(top[p_values[Diffs[show_comp[0]]][Diffs[show_comp[1]]] <= pthresh] + 0.1,avgs[diff]['norm_depths'][p_values[Diffs[show_comp[0]]][Diffs[show_comp[1]]] <= pthresh],color='k',marker='$*$',linestyle='None',label=None)
            else:
                corrected_pvalues = multipletests(p_values[Diffs[show_comp[0]]][Diffs[show_comp[1]]],method=statCorrType)[1]
                p2.plot(top[corrected_pvalues <= pthresh] + 0.1,avgs[diff]['norm_depths'][corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None',label=None)
        else:
            print("plot_profile_comparison: Improper formating of statCorrType argument")
            
def compare_conditions(profiles,conditions,statTestType='t-test', npermSamples=10000):
    '''
    Compare Conditions
    
    For N conditions and M measurments, returns the p-value for the two-sample
    t-test between each of the N conditions of the corresponding measurements.
    
    Inputs:
        profiles (array-like): an NxMxT array containing the values for each 
            measurement m_i in a given condition n_i for a given sample t_i
        conditions (array-like): names of the conditions in the order they 
            appear in the profiles array
            
    Outputs:
        t_statistics (dict): an dictionary containing the t-stats for 
            each of the comparisons between conditions (n_i) for each 
            measurment (m_i)
        p_values (dict): n dictionary containing the p-vals for 
            each of the comparisons between conditions (n_i) for each 
            measurment (m_i)
            
    Dependencies:
        numpy, scipy.stats
    '''
    
    N, M, T = profiles.shape

    # Initialize arrays to store t-statistics and p-values
    t_statistics = {c:{} for c in conditions}
    p_values = {c:{} for c in conditions}

    # Perform t-test for each pair of conditions
    for i in range(N):
        for j in range(N):
            if i != j:
                # Select the corresponding measurements
                condition_1 = np.squeeze(profiles[i,:,:])
                condition_2 = np.squeeze(profiles[j,:,:])
                
                if statTestType == 't-test':
                    # Perform two-sample t-test
                    t_stat, p_value = stats.ttest_ind(condition_1, condition_2, axis = 1)
    
                    # Store the t-statistic and p-value in the respective arrays
                    t_statistics[conditions[i]][conditions[j]] = t_stat
                    p_values[conditions[i]][conditions[j]] = p_value
                    
                elif statTestType == 'permutation':
                    # Perform two-sample permutation test
                    p_value = stats.permutation_test((condition_1, condition_2), np.mean, 
                                                             permutation_type='independent', 
                                                             n_resamples=npermSamples, 
                                                             alternative='two-sided', 
                                                             axis = 1).pvalue
    
                    # Store the t-statistic and p-value in the respective arrays
                    p_values[conditions[i]][conditions[j]] = p_value
                    t_statistics[conditions[i]][conditions[j]] = np.nan(np.shape(p_value))
                
                else:
                    print("Invalid statTestType type!!! Using 't-test' as default.")
                    # Perform two-sample t-test
                    t_stat, p_value = stats.ttest_ind(condition_1, condition_2, axis = 1)
    
                    # Store the t-statistic and p-value in the respective arrays
                    t_statistics[conditions[i]][conditions[j]] = t_stat
                    p_values[conditions[i]][conditions[j]] = p_value

    return t_statistics, p_values

def makeRadProfile1D(radData, nRad, paramData, radMin, radMax, useLayNii):
    '''
    Makes radial profile with given surface boundaries.
    
    Inputs:
        radData (array): radius values for each voxel in ROI
        nRad (int): number of radial bins to use
        paramData (array): the measurement you want to return grouped by depth
        radMax (float): the value for the maximum depth
        useLayNii (bool): use the laynii depths if True, otherwise use depthify
        
    Outputs:
        dataDict (dict): a dictionary of statistics computed for the profile 
            and over the whole ROI
        
    Dependencies:
        numpy
    '''
    # set up the depths
    binSize = (radMax-radMin)/nRad
    print('depth bin size: %d' %binSize)
    if useLayNii:
        radBoundaries = np.linspace(radMin-binSize/2, radMax, nRad+1)
    else:
        radBoundaries = 0.5 + np.arange(radMin, radMax + 1., binSize) 
    # mask the depth data using the ROI
    try:
        nParams = paramData.shape[1]
    except:
        nParams = 1
        paramData = np.reshape(paramData, (len(paramData), 1))
    dataDict = {'whole ROI': {}, 'profile': {}}
    dataDict['whole ROI']['nVox'] = len(radData)
    dataDict['whole ROI']['avg'] = np.mean(paramData, axis=0)
    dataDict['profile']['rad'] = [0 for iD in range(len(radBoundaries) - 1)]
    dataDict['profile']['nVox'] = [0 for iD in range(len(radBoundaries) - 1)]
    dataDict['profile']['avg'] = [[0 for iD in range(len(radBoundaries) - 1)] for iP in range(nParams)]
    dataDict['profile']['param'] = [[0 for iD in range(len(radBoundaries) - 1)] for iP in range(nParams)]
    dataDict['profile']['std'] = [[0 for iD in range(len(radBoundaries) - 1)] for iP in range(nParams)]
    for iP in range(nParams):
        for iD in range(len(radBoundaries) - 1):
            mask = 1.*(radData > radBoundaries[iD])*(radData <= radBoundaries[iD + 1])
            dataDict['profile']['rad'][iD] = np.mean(radData[mask > 0.])
            dataDict['profile']['nVox'][iD] = np.sum(mask)
            dataDict['profile']['param'][iP][iD] = paramData[:, iP][mask > 0.]
            dataDict['profile']['avg'][iP][iD] = np.mean(paramData[:, iP][mask > 0.])
            dataDict['profile']['std'][iP][iD] = np.std(paramData[:, iP][mask > 0.])
    return dataDict

def compute_all_rad_profiles(all_data,statDetails,profile_method,nRad,mask,radParam='scale_xy_dist',spec_Drange='MinMax',smooth_factor=0.3,radMax=4):
    '''
    Computes radial profiles for each subject and returns data.
    Inputs:
      all_data (dict): all voxel beta weights for each individual
      statDetails (dict): a dictionary containing labels of the conditions and 
          colors
      profile_method (str): the profile method to use (bin or smooth)
      nRad (int): number of radii to sample
      mask (dict): voxel mask; each item should contain a boolean array
      radParam (str): name of the variable to use as the depth parameter 
          (this is probably going to be z for depthify or d for LayNii)
          DEFAULT - 'z'
      spec_Drange (str or list/tuple/arraylike): if list/tuple/arraylike, uses the first value as the minimum depth 
          (WM side) and the second value as the maximum depth (pial side); if
          str and ='MinMax', use the maximum and minimum values specified in 
          the depth variable as the boundaries
          DEFAULT - 'MinMax'
      smooth_factor (float): smoothing factor if using smooth depth profiles
          DEFAULT - 0.3
      radMax (float): max radius (in SD) to use only if using smooth depth 
          profiles
          DEFAULT - 4
    
    Outputs:
      keep_rois (dict): a dictionary containing the depth profies for each 
          coniditon  provided in statDetails with the following information:
              1) average depth profiles (NROIs x Ndepths)
              2) standard deviations at each depth (NROIs x Ndepths)
              3) list of depths (NROIS x Ndepths)
              4) number of voxels at each depth (NROIs x Ndepths)
    
    Dependencies:
        numpy, collections, makeProfile1D, smoothen, smooth_kernel
        
    Notes:
        The current version does not return stdev for the smooth profiles.
    '''
    
    # pick out ROIs where we're sure of localization
    keep_rois = {label: {'avg': [], 'stdev': [], 'N': [], 'rad': []} for label in statDetails['labels']}
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = mask[label]
        roi = df[roi_idx]
        # roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
        roi = roi[roi[radParam] >=0]
    
        for stat in statDetails['labels']:
            if profile_method == 'bin':
                if spec_Drange == 'MinMax':
                    dataDict = makeRadProfile1D(roi[radParam].values,
                                                 nRad,
                                                 roi[stat].values,
                                                 np.min(roi[radParam].values),
                                                 np.max(roi[radParam].values),
                                                 radParam==radParam) #assumes you want to use LayNii if the depthParam is d
                elif isinstance(spec_Drange,(collections.abc.Sequence, np.ndarray)):
                    dataDict = makeRadProfile1D(roi[radParam].values,
                                                 nRad,
                                                 roi[stat].values,
                                                 spec_Drange[0],
                                                 spec_Drange[1],
                                                 radParam==radParam) #assumed you want to use LayNii if the depthParam is d
                else:
                    print("compute_all_rad_profiles: Invalid variable type for spec_Drange!")
                    
                keep_rois[stat]['rad'].append(np.squeeze(dataDict['profile']['rad']))
                keep_rois[stat]['avg'].append(np.squeeze(dataDict['profile']['avg']))
                keep_rois[stat]['stdev'].append(np.squeeze(dataDict['profile']['std']))
                keep_rois[stat]['N'].append(np.squeeze(dataDict['profile']['nVox']))
                
            elif profile_method == 'smooth':
                profiles = []
                profile, radius = smoothen(roi[stat],
                                          roi[radParam].values,
                                          radMax,
                                          nRadii=nRad,
                                          kernel=smooth_kernel,
                                          smooth_factor=smooth_factor)
                keep_rois[stat]['rad'].append(np.squeeze(radius))
                keep_rois[stat]['avg'].append(np.squeeze(profiles))
        
    return(keep_rois)

def compute_rad_diff_profiles(all_data,statDetails,diffDetails,profile_method,nRad,useSI,mask,radParam='scale_xy_dist',spec_Drange='MinMax',smooth_factor=0.3,radMax=4):
    '''
    Plots radial profiles for each subject and returns data.
    Inputs:
      all_data (dict): all voxel beta weights for each individual
      statDetails (dict): a dictionary containing labels of the conditions and 
          colors
      diffDetails (dict): a dictionary containing the labels of the difference 
          profiles and the corresponding the conditions to use in the 
          differences (e.g. {'diff1': ['cond1','cond2'], 'diff2': ['cond1','cond3']})
      profile_method (str): the profile method to use (bin or smooth)
      nRad (int): number of radii to sample
      useSI (bool): tells the function wether to use suppression index (true) 
          rather than pure difference (false) for the difference profiles
      mask (dict): voxel mask; each item should contain a boolean array
      radParam (str): name of the variable to use as the rad parameter 
          DEFAULT - 'scale_xy_dist'
      spec_Drange (str or list/tuple/arraylike): if list/tuple/arraylike, uses 
          the first value as the minimum depth (WM side) and the second value 
          as the maximum depth (pial side); if str and ='MinMax', use the 
          maximum and minimum values specified in the depth variable as the 
          boundaries
          DEFAULT - 'MinMax'
      smooth_factor (float): smoothing factor if using smooth depth profiles
          DEFAULT - 0.3
      radMax (float): max radius (in SD) to use only if using smooth depth 
          profiles
          DEFAULT - 4
    
    Outputs:
      keep_diffs (dict): a dictionary containing difference profiles for each
          difference condition listed in diffDetails with the following 
          information:
              1) average difference between specified conditions at 
                 each depth (NROIs x Ndepths)
              2) standard deviations at each depth (NROIS x Ndepths)
              3) list of depths (NROIS x Ndepths)
              4) number of voxels at each depth (NROIs x Ndepths)
    
    Dependencies:
        numpy, matplotlib, makeProfile1D, smoothen, smooth_kernel
    '''
    
    keep_diffs = {diff_label: {'avg': [], 'stdev': [], 'N': [], 'rad': []} for diff_label in diffDetails.keys()}
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = mask[label]
        roi = df[roi_idx]
        # roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
        roi = roi[roi[radParam] >=0]
    
        if profile_method == 'bin':
            if spec_Drange == 'MinMax':
                dataDict = makeRadProfile1D(roi[radParam].values,
                                             nRad,
                                             roi[statDetails['labels']].values,
                                             np.min(roi[radParam].values),
                                             np.max(roi[radParam].values),
                                             radParam==radParam) #assumes you want to use LayNii if the depth parameter is d
            elif isinstance(spec_Drange,(collections.abc.Sequence, np.ndarray)):
                dataDict = makeRadProfile1D(roi[radParam].values,
                                             nRad,
                                             roi[statDetails['labels']].values,
                                             spec_Drange[0],
                                             spec_Drange[1],
                                             radParam==radParam)
            else:
                print("compute_all_rad_profiles: Invalid variable type for spec_Drange!")
                
            for diff_label in diffDetails.keys():
                cond1 = diffDetails[diff_label][0]
                cond2 = diffDetails[diff_label][1]
                c1i = np.where(np.array(statDetails['labels']) == cond1)[0][0]
                c2i = np.where(np.array(statDetails['labels']) == cond2)[0][0]
                if c1i == None or c2i == None:
                    print("compute_rad_profiles: Invalid condition type!")
                if useSI:
                    cond_diff = np.asarray(dataDict['profile']['avg'][c1i]) - np.asarray(dataDict['profile']['avg'][c2i])
                    cond_sum = np.asarray(dataDict['profile']['avg'][c1i]) + np.asarray(dataDict['profile']['avg'][c2i])
                    diff_avg = cond_diff / cond_sum
                    stdev_diff = np.sqrt(np.asarray(dataDict['profile']['std'][c1i])**2 + np.asarray(dataDict['profile']['std'][c2i])**2) #propagate errors for subtraction or addition
                    stdev = np.sqrt((stdev_diff/cond_diff)**2 + (stdev_diff/cond_sum)**2) #propagate errors for division
                else:
                    diff_avg = np.asarray(dataDict['profile']['avg'][c1i]) - np.asarray(dataDict['profile']['avg'][c2i])
                    stdev = np.sqrt(np.asarray(dataDict['profile']['std'][c1i])**2 + np.asarray(dataDict['profile']['std'][c2i])**2) #propagate errors for subtraction
            
                #save
                keep_diffs[diff_label]['rad'].append(dataDict['profile']['rad'])
                keep_diffs[diff_label]['N'].append(dataDict['profile']['nVox'])
                keep_diffs[diff_label]['avg'].append(diff_avg)
                keep_diffs[diff_label]['stdev'].append(stdev)
                print(diff_label)
    
        elif profile_method == 'smooth':
            for diff_label in diffDetails.keys():
                cond1 = diffDetails[diff_label][0]
                cond2 = diffDetails[diff_label][1]
                c1i = np.where(np.array(statDetails['labels']) == cond1)[0][0]
                c2i = np.where(np.array(statDetails['labels']) == cond2)[0][0]
                
                profile1, rad1 = smoothen(roi[statDetails['labels']].values[:, c1i],
                                              roi[radParam].values,
                                              radMax,
                                              nRadii=nRad,
                                              kernel=smooth_kernel,
                                              smooth_factor=smooth_factor)
                profile2, rad2 = smoothen(roi[statDetails['labels']].values[:, c2i],
                                              roi[radParam].values,
                                              radMax,
                                              nRadii=nRad,
                                              kernel=smooth_kernel,
                                              smooth_factor=smooth_factor)
                diff_avg = profile1 - profile2
                
                #save
                keep_diffs[diff_label]['rad'].append(rad1)
                keep_diffs[diff_label]['avg'].append(diff_avg)
                print(diff_label)
                
    return(keep_diffs)

def compute_avg_rad_profile(rois,statDetails,diffDetails,plotStats,plotDiffs,prop_err,useSI,statTestType = 't-test', npermSamples=10000):
    '''
    Calculate average radial profiles across ROIs
    
    Inputs:
        rois (dict): roi profiles with the following information
            1) average depth profiles for each condition (NROIs x Ndepths)
            2) standard deviation of depth profiles for each condition (NROIs x Ndepths)
            3) depths of each bin (NROIs x Ndepths)
        statDetails (dict): dictionary of condition details including labels 
            and colors for each condition
        diffDetails (dict): dictionary of the difference profiles and 
            corresponding stats to use to calculate them
        plotStats (list): list of labels for stats to plot
        plotDiffs (list): list of labels for diffs to plot
        prop_err (bool): propagate errors from individual profiles if true
        useSI (bool): use suppression index rather than difference profile
        
    Outputs:
        stat_avg (dict): average depth values for each condition which 
            contains the following information:
            1) average profile over all ROIs (Ndepths)
            2) standard deviation over all ROIs (Ndepths)
            3) normalized depths (Ndepths)
        diff_avg (dict): average depth difference profile which contains the 
            following information:
            1) average different profile over all ROIs (Ndepths)
            2) standard deviation over all ROIs (Ndepths)
            3) normalized depths (Ndepths)
            4) results of a single-sample two-sided t-test computed at each 
                depth against the null of diff = 0 (Ndepths)
            
        Dependencies:
            numpy, matplotlib, scipy
        
    '''
    
    stat_avg = {stat: {'avg': [], 'stdev': [], 'norm_rad': []} for stat in plotStats}
    diff_avg = {diff: {'avg': [], 'stdev': [], 'norm_rad': [], 'p-vals': []} for diff in plotDiffs}
    
    for iStat, stat in enumerate(plotStats):
        
        #average profile
        stat_avg[stat]['avg'] = np.nanmean(np.asarray(rois[stat]['avg']), axis=0)
        
        #normalized depths
        stat_all = np.asarray(rois[stat]['avg'])
        rad_avg = np.nanmean(np.asarray(rois[stat]['rad']), axis=0)
        stat_avg[stat]['norm_rad'] = (rad_avg - np.min(rad_avg))/(np.max(rad_avg) - np.min(rad_avg)) #normalized depth
       
        #standard deviation
        if prop_err:
            stat_avg[stat]['stdev'] = np.sqrt(np.nansum(np.asarray(rois[stat]['stdev'])**2,axis=0))/np.shape(rois[stat]['stdev'])[0]
        else:
            stat_avg[stat]['stdev'] = np.nanstd(np.asarray(rois[stat]['avg']), axis=0)
            
        #number of samples
        stat_avg[stat]['Nsamp'] = np.shape(rois[stat]['avg'])[0]
        
        #one-sample two-sided t-test
        if statTestType == 't-test':
            stat_avg[stat]['p-vals'] = stats.ttest_1samp(stat_all,0,axis=0)
        elif statTestType == 'permutation':
            stat_avg[stat]['p-vals'] = permute_1samp(stat_all,np.mean,null_stat=0,n_permutations=npermSamples,test_type='two-sided',axis=0)
        else:
            print("Unknown stat test type. Defaulting to t-test.")
            stat_avg[stat]['p-vals'] = stats.ttest_1samp(stat_all,0,axis=0)
            
    for iDiff, diff in enumerate(plotDiffs):
        
        #get conditions to subtract
        cond1 = diffDetails[diff][0]
        cond2 = diffDetails[diff][1]
        
        #calculate diffs
        diff_all = np.asarray(rois[cond1]['avg'])-np.asarray(rois[cond2]['avg'])
        diff_avg[diff]['avg'] = np.nanmean(diff_all,axis=0)
        
        #normalized depths
        rad_avg = np.nanmean(np.asarray(rois[cond1]['rad']), axis=0)
        diff_avg[diff]['norm_rad'] = (rad_avg - np.min(rad_avg))/(np.max(rad_avg) - np.min(rad_avg)) #normalized depth
        
        #standard deviation
        if prop_err:
            diff_avg[diff]['stdev'] = np.sqrt(np.nansum(np.vstack([stat_avg[cond1]['stdev']**2,stat_avg[cond2]['stdev']**2]),axis=0))
        else:
            diff_avg[diff]['stdev'] = np.nanstd(diff_all,axis=0)
            
        #number of samples
        diff_avg[diff]['Nsamp'] = np.shape(rois[cond1]['avg'])[0]
            
        #one-sample two-sided t-test
        if statTestType == 't-test':
            diff_avg[diff]['p-vals'] = stats.ttest_1samp(diff_all,0,axis=0)
        elif statTestType == 'permutation':
            diff_avg[diff]['p-vals'] = permute_1samp(diff_all,np.mean,null_stat=0,n_permutations=npermSamples,test_type='two-sided',axis=0)
        else:
            print("Unknown stat test type. Defaulting to t-test.")
            diff_avg[diff]['p-vals'] = stats.ttest_1samp(diff_all,0,axis=0)
    
    return(stat_avg,diff_avg)

def permute_1samp(data, test_stat, null_stat=0, n_permutations=1000, test_type='two-sided', axis=0):
    # Compute the actual test statistic for the given data
    actual_stat = test_stat(data, axis=axis)
    
    # Determine the maximum number of unique permutations
    max_permutations = 2 ** data.shape[axis]
    if n_permutations > max_permutations:
        n_permutations = max_permutations
    
    # Initialize a set to store unique sign flip patterns and an array to store the permutation test statistics
    unique_sign_patterns = set()
    perm_stats = np.zeros((n_permutations,) + actual_stat.shape)
    
    # Generate unique permutations by randomly flipping the signs of the observations
    i = 0
    while i < n_permutations:
        sign_flips = tuple(np.random.choice([-1, 1], size=data.shape[axis], replace=True))
        if sign_flips not in unique_sign_patterns:
            unique_sign_patterns.add(sign_flips)
            expanded_sign_flips = np.expand_dims(np.array(sign_flips), axis=tuple(range(1, data.ndim)))
            permuted_data = data * expanded_sign_flips
            perm_stats[i] = test_stat(permuted_data, axis=axis)
            i += 1
    
    # Compute p-values based on the null distribution
    if test_type == 'two-sided':
        p_values = np.mean(np.abs(perm_stats - null_stat) >= np.abs(actual_stat - null_stat), axis=0)
    elif test_type == 'one-sided-positive':
        p_values = np.mean(perm_stats >= actual_stat, axis=0)
    elif test_type == 'one-sided-negative':
        p_values = np.mean(perm_stats <= actual_stat, axis=0)
    else:
        raise ValueError("Invalid test_type. Choose from 'two-sided', 'one-sided-positive', or 'one-sided-negative'.")
    
    return p_values
