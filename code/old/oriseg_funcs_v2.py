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
    print('depth bin size: %d' %binSize)
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
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['left'].set_color(lcolor)
    ax.spines['bottom'].set_position(('outward', 8))
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

def compute_all_depth_profiles(all_data,roi_dict,statDetails,profile_method,nDepths,mask,depthParam='z',radialParam='scale_xy_dist',spec_Drange='MinMax',smooth_factor=0.3,radMax=4):
    '''
    Computes depth profiles for each subject and returns data.
    Inputs:
      all_data (dict): all voxel beta weights for each individual
      roi_dict (dict): all rois as binary arrays organized in a dictionary
      statDetails (dict): a dictionary containing labels of the conditions and 
          colors
      profile_method (str): the profile method to use (bin or smooth)
      nDepths (int): number of depths to sample
      mask (dict): voxel mask; each item should contain a boolean array
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
    '''
    
    # pick out ROIs where we're sure of localization
    keep_rois = {label: {'avg': [], 'stdev': [], 'N': [], 'depths': []} for label in statDetails['labels']}
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = roi_dict[label] & mask[label]
        roi = df[roi_idx]
        # roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
        roi = roi[roi[radialParam] >=0]
    
        for stat in statDetails['labels']:
            if profile_method == 'bin':
                if spec_Drange == 'MinMax':
                    dataDict = makeProfile1D(roi[depthParam].values,
                                                 nDepths,
                                                 roi[stat].values,
                                                 np.min(roi[depthParam].values),
                                                 np.max(roi[depthParam].values),
                                                 depthParam=='d') #assumes you want to use LayNii if the depthParam is d
                elif isinstance(spec_Drange,(collections.abc.Sequence, np.ndarray)):
                    dataDict = makeProfile1D(roi[depthParam].values,
                                                 nDepths,
                                                 roi[stat].values,
                                                 spec_Drange[0],
                                                 spec_Drange[1],
                                                 depthParam=='d') #assumed you want to use LayNii if the depthParam is d
                else:
                    print("compute_all_depth_profiles: Invalid variable type for spec_Drange!")
                    
                keep_rois[stat]['depths'].append(np.squeeze(dataDict['profile']['depth']))
                keep_rois[stat]['avg'].append(np.squeeze(dataDict['profile']['avg']))
                keep_rois[stat]['stdev'].append(np.squeeze(dataDict['profile']['std']))
                keep_rois[stat]['N'].append(np.squeeze(dataDict['profile']['nVox']))
                
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

def compute_diff_profiles(all_data,roi_dict,statDetails,diffDetails,profile_method,nDepths,useSI,mask,depthParam='z',radialParam='x',spec_Drange='MinMax',smooth_factor=0.3,radMax=4):
    '''
    Plots depth profiles for each subject and returns data.
    Inputs:
      all_data (dict): all voxel beta weights for each individual
      roi_dict (dict): all rois as binary arrays organized in a dictionary
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
    '''
    
    keep_diffs = {diff_label: {'avg': [], 'stdev': [], 'N': [], 'depths': []} for diff_label in diffDetails.keys()}
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = roi_dict[label] & mask[label]
        roi = df[roi_idx]
        # roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
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
                    stdev = np.sqrt(np.asarray(dataDict['profile']['std'][c1i])**2 + np.asarray(dataDict['profile']['std'][c2i])**2) #propagate errors for subtraction
            
                #save
                keep_diffs[diff_label]['depths'].append(dataDict['profile']['depth'])
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
        keep_rois (list): depth profiles for ROIs 
            (NROIs x Nconditions x Ndepths)
        p2t_model (float): peak-to-tail ratio used to set point spread 
            functions over depth
        Nbins_model (int): number of depth bins for the desing matrix
        Nbins (int): number of depth bins in data
        normalize_psf (bool): normalize the psf by the deepest layer if True
        
    Outputs:
        decon_rois (list): deconvolved depth profiles for ROIs
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


def compute_avg_depth_profile(rois,statDetails,diffDetails,plotStats,plotDiffs,use_decon,prop_err,useSI):
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
        stat_avg[stat]['norm_depths'] = (depth_avg - np.min(depth_avg))/(np.max(depth_avg) - np.min(depth_avg)) #normalized depth
       
        #standard deviation
        if prop_err:
            stat_avg[stat]['stdev'] = np.sqrt(np.nansum(np.asarray(rois[stat]['stdev'])**2,axis=0))/np.shape(rois[stat]['stdev'])[0]
        else:
            stat_avg[stat]['stdev'] = np.nanstd(np.asarray(rois[stat]['avg']), axis=0)
            
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
        diff_avg[diff]['norm_depths'] = (depth_avg - np.min(depth_avg))/(np.max(depth_avg) - np.min(depth_avg)) #normalized depth
        
        #standard deviation
        if prop_err:
            diff_avg[diff]['stdev'] = np.sqrt(np.nansum(np.vstack([stat_avg[cond1]['stdev']**2,stat_avg[cond2]['stdev']**2]),axis=0))
        else:
            diff_avg[diff]['stdev'] = np.std(diff_all,axis=0)
            
        #one-sample two-sided t-test
        diff_avg[diff]['p-vals'] = stats.ttest_1samp(diff_all,0,axis=0)
    
    return(stat_avg,diff_avg)

def plot_avg_depth_profile(p1,avgProfiles,Stats,Colors,ylim,xlim,dx,dy,Ntext,lcolor,fsize):
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
        
    Outputs:
        (None)
        
    Dependencies:
        numpy, matplotlib

    '''
    
    for iStat, stat in enumerate(Stats):
        p1.plot(avgProfiles[stat]['avg'], avgProfiles[stat]['norm_depths'], color=Colors[iStat])
        p1.fill_betweenx(avgProfiles[stat]['norm_depths'],
                        avgProfiles[stat]['avg'] - avgProfiles[stat]['stdev']/np.sqrt(np.shape(avgProfiles[stat]['stdev'])[0]),
                        avgProfiles[stat]['avg'] + avgProfiles[stat]['stdev']/np.sqrt(np.shape(avgProfiles[stat]['stdev'])[0]),
                        linewidth=0.,
                        alpha=0.4,
                        color=Colors[iStat])
        p1.text(dx, dy + iStat*.07, stat,
                color=Colors[iStat],
                fontsize=fsize-2)
        p1.set_ylim(ylim)
        p1.set_xlim(xlim)
        p1.set_xlabel('BOLD % change', fontsize=fsize, color=lcolor)
        p1.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
        p1.text(Ntext[0], Ntext[1], 'n=%d hemis' %np.shape(avgProfiles[stat]['avg'])[0], color=lcolor, fontsize=fsize*.5, fontstyle='italic')

    return()

def plot_avg_diff_profile(p2,avgDiffs,Diffs,Colors,ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=False,pthresh=0.05):
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
        showSig (bool): indicate significance if true
            DEFAULT - false
        pthresh (float): p-value threshold if showSig is true
            DEFAULT - 0.05
        
    Outputs:
        (None)
        
    Dependencies:
        numpy, matplotlib

    '''
    
    p2.plot([0, 0], [0, 1], '--', color='gray')

    for iDiff, diff in enumerate(Diffs):
        p2.plot(avgDiffs[diff]['avg'], avgDiffs[diff]['norm_depths'], color=Colors[iDiff])
        p2.fill_betweenx(avgDiffs[diff]['norm_depths'],
                        avgDiffs[diff]['avg'] - avgDiffs[diff]['stdev']/np.sqrt(np.shape(avgDiffs[diff]['stdev'])),
                        avgDiffs[diff]['avg'] + avgDiffs[diff]['stdev']/np.sqrt(np.shape(avgDiffs[diff]['stdev'])),
                        linewidth=0.,
                        alpha=0.4,
                        color=Colors[iDiff])
        
        if showSig:
            top = avgDiffs[diff]['avg'] + avgDiffs[diff]['stdev']/np.sqrt(np.shape(avgDiffs[diff]['avg'])[0])
            p2.plot(top[avgDiffs[diff]['p-vals'].pvalue <= pthresh] + 0.1,avgDiffs[diff]['norm_depths'][avgDiffs[diff]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
        
        p2.set_ylim(ylim)
        p2.set_xlim(xlim)
        p2.text(1.15, .05+iDiff*0.1, diff, color=Colors[iDiff], fontsize=fsize-2)
        p2.set_yticklabels([])
        if useSI:
            p2.set_xlabel(r'$\Delta$ SI', fontsize=fsize, color=lcolor)
        else:
            p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)
