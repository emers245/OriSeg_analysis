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

def plot_depth_profiles(all_data,roi_dict,statDetails,profile_method,nDepths,figNames,useSI,mask,useLayNii):
    '''
    Plots depth profiles for each subject and returns data.
    Inputs:
      all_data (dict): all voxel beta weights for each individual
      roi_dict (dict): all rois as binary arrays organized in a dictionary
      statDetails (dict): a dictionary containing labels of the conditions and colors
      profile_method (str): the profile method to use (bin or smooth)
      nDepths (int): number of depths to sample
      figNames (list): a list of 2 strings for the figure names
      useSI (bool): tells the function wether to use suppression index (true) rather than pure difference (false) for the difference profiles
      mask (dict): voxel mask; each item should contain a boolean array
    
    Outputs:
      hprofiles (matplotlib.pyplot handle): handle for profiles figure
      hdiff (matplotlib.pyplot handle): handle for difference profiles figure
      keep_rois (list): list of depth profiles for each condition and ROI 
          (NROIS x Nconditions x Ndepths)
      keep_std (list): standard deviations at each depth 
          (NROIS x Nconditions x Ndepths)
      keep_depths (list): list of depths (NROIS x Ndepths)
      fgm (list): figure-ground modulations profiles (NROIs x Ndepths)
      odss (list): orientation-dependent surround suppression profiles (NROIs x Ndepths)
      dSI (list): difference in suppression index profiles orth-iso (NROIs x Ndepths)
    
    Dependencies:
        numpy, matplotlib
    '''
    
    hprofiles = plt.figure(figNames[0])
    hdiff = plt.figure(figNames[1])
    
    # pick out ROIs where we're sure of localization
    keep_depths = []
    keep_rois = []
    keep_std = []
    fgm = []
    odss = []
    dSI = []
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = roi_dict[label] & mask[label]
        roi = df[roi_idx]
        # roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
        if useLayNii:
            roi = roi[roi['scale_uv_dist'] >=0]
        else:
             roi = roi[roi['scale_xy_dist'] >= 0] #don't allow negative radii
        
        plt.figure(figNames[0])
        plt.subplot(2, int(np.ceil(len(all_data.keys())/2.)), 1 + iR)    
        plt.figure(figNames[1])
        plt.subplot(2, int(np.ceil(len(all_data.keys())/2.)), 1 + iR)  
    
        if profile_method == 'bin':
            if useLayNii:
                dataDict = makeProfile1D(roi['d'].values,
                                         nDepths,
                                         roi[statDetails['labels']].values,
                                         np.min(roi['d'].values),
                                         np.max(roi['d'].values),
                                         useLayNii)
            else:
                dataDict = makeProfile1D(roi['z'].values,
                                         nDepths,
                                         roi[statDetails['labels']].values,
                                         np.min(roi['z'].values), #0
                                         np.max(roi['z'].values), #21
                                         useLayNii)
            for iStat in range(len(statDetails['labels'])):
                plt.figure(figNames[0])
                plt.plot(dataDict['profile']['avg'][iStat],
                         dataDict['profile']['depth'],
                         color=statDetails['colors'][iStat])
                plt.fill_betweenx(dataDict['profile']['depth'],
                        dataDict['profile']['avg'][iStat] - dataDict['profile']['std'][iStat]/np.sqrt(dataDict['profile']['nVox'][iR]),
                        dataDict['profile']['avg'][iStat] + dataDict['profile']['std'][iStat]/np.sqrt(dataDict['profile']['nVox'][iR]),
                        linewidth=0.,
                        alpha=0.4,
                        color=statDetails['colors'][iStat])
                
            plt.figure(figNames[1])
            if useSI:
                dSI_diff = np.asarray(dataDict['profile']['avg'][3]) - np.asarray(dataDict['profile']['avg'][1])
                dSI_sum = np.asarray(dataDict['profile']['avg'][3]) + np.asarray(dataDict['profile']['avg'][1])
                avg_dSI = dSI_diff / dSI_sum
                stddev_diff = np.sqrt(np.asarray(dataDict['profile']['std'][3])**2 + np.asarray(dataDict['profile']['std'][1])**2) #propagate errors for subtraction or addition
                stddev_dSI = np.sqrt((stddev_diff/dSI_diff)**2 + (stddev_diff/dSI_sum)**2) #propagate errors for division
            else:
                avg_dSI = np.asarray(dataDict['profile']['avg'][3]) - np.asarray(dataDict['profile']['avg'][1])
                stddev_dSI = np.sqrt(np.asarray(dataDict['profile']['std'][3])**2 + np.asarray(dataDict['profile']['std'][1])**2) #propagate errors for subtraction
            plt.plot(avg_dSI,
                         dataDict['profile']['depth'],
                         color='magenta')
            plt.fill_betweenx(dataDict['profile']['depth'],
                        avg_dSI - stddev_dSI/np.sqrt(dataDict['profile']['nVox'][iR]),
                        avg_dSI + stddev_dSI/np.sqrt(dataDict['profile']['nVox'][iR]),
                        linewidth=0.,
                        alpha=0.4,
                        color='magenta')
            if useSI:
                fgm_diff = np.asarray(dataDict['profile']['avg'][2]) - np.asarray(dataDict['profile']['avg'][1])
                fgm_sum = np.asarray(dataDict['profile']['avg'][2]) + np.asarray(dataDict['profile']['avg'][1])
                avg_fgm = fgm_diff / fgm_sum
                stddev_diff = np.sqrt(np.asarray(dataDict['profile']['std'][2])**2 + np.asarray(dataDict['profile']['std'][1])**2) #propagate errors for subtraction or addition
                stddev_fgm = np.sqrt((stddev_diff/fgm_diff)**2 + (stddev_diff/fgm_sum)**2) #propagate errors for division
            else:
                avg_fgm = np.asarray(dataDict['profile']['avg'][2]) - np.asarray(dataDict['profile']['avg'][1])
                stddev_fgm = np.sqrt(np.asarray(dataDict['profile']['std'][2])**2 + np.asarray(dataDict['profile']['std'][1])**2) #propagate errors for subtraction
            plt.plot(avg_fgm,
                     dataDict['profile']['depth'],
                         color='cyan')
            plt.fill_betweenx(dataDict['profile']['depth'],
                        avg_fgm - stddev_fgm/np.sqrt(dataDict['profile']['nVox'][iR]),
                        avg_fgm + stddev_fgm/np.sqrt(dataDict['profile']['nVox'][iR]),
                        linewidth=0.,
                        alpha=0.4,
                        color='cyan')
            if useSI:
                odss_diff = np.asarray(dataDict['profile']['avg'][3]) - np.asarray(dataDict['profile']['avg'][2])
                odss_sum = np.asarray(dataDict['profile']['avg'][3]) + np.asarray(dataDict['profile']['avg'][2])
                avg_odss = odss_diff / odss_sum
                stddev_diff = np.sqrt(np.asarray(dataDict['profile']['std'][3])**2 + np.asarray(dataDict['profile']['std'][2])**2) #propagate errors for subtraction or addition
                stddev_odss = np.sqrt((stddev_diff/odss_diff)**2 + (stddev_diff/odss_sum)**2) #propagate errors for division
            else:
                avg_odss = np.asarray(dataDict['profile']['avg'][3]) - np.asarray(dataDict['profile']['avg'][2])
                stddev_odss = np.sqrt(np.asarray(dataDict['profile']['std'][3])**2 + np.asarray(dataDict['profile']['std'][2])**2) #propagate errors for subtraction
            plt.plot(avg_odss,
                     dataDict['profile']['depth'],
                         color='tab:green')
            plt.fill_betweenx(dataDict['profile']['depth'],
                        avg_odss - stddev_odss/np.sqrt(dataDict['profile']['nVox'][iR]),
                        avg_odss + stddev_odss/np.sqrt(dataDict['profile']['nVox'][iR]),
                        linewidth=0.,
                        alpha=0.4,
                        color='tab:green')
            
            print(label)
            keep_depths.append(dataDict['profile']['depth'])
            keep_rois.append(dataDict['profile']['avg'])
            keep_std.append(dataDict['profile']['std'])
        elif profile_method == 'smooth':
            profiles = []
            for iStat in range(len(statDetails['labels'])):
                profile, depth = smoothen(roi[statDetails['labels']].values[:, iStat],
                                          roi['z'].values,
                                          radMax=10,
                                          nRadii=nDepths,
                                          kernel=smooth_kernel,
                                          smooth_factor=.3)
                plt.figure(figNames[0])
                plt.plot(depth, profile, color=statDetails['colors'][iStat])
                profiles.append(profile)
            keep_depths.append(depth)
            keep_rois.append(profiles)
    
        # now compute FGM and ODSS
        idx_orth = statDetails['labels'].index('orth')
        idx_iso90 = statDetails['labels'].index('iso90')
        idx_iso = statDetails['labels'].index('iso0')
        if useSI:
            odss_sub = np.asarray(dataDict['profile']['avg'][idx_orth]) - np.asarray(dataDict['profile']['avg'][idx_iso90])
            odss_sum = np.asarray(dataDict['profile']['avg'][idx_orth]) + np.asarray(dataDict['profile']['avg'][idx_iso90])
            odss.append(odss_sub/odss_sum)
            fgm_sub = np.asarray(dataDict['profile']['avg'][idx_iso90]) - np.asarray(dataDict['profile']['avg'][idx_iso])
            fgm_sum = np.asarray(dataDict['profile']['avg'][idx_iso90]) + np.asarray(dataDict['profile']['avg'][idx_iso])
            fgm.append(fgm_sub/fgm_sum)
            dSI_sub = np.asarray(dataDict['profile']['avg'][idx_orth]) - np.asarray(dataDict['profile']['avg'][idx_iso])
            dSI_sum = np.asarray(dataDict['profile']['avg'][idx_orth]) + np.asarray(dataDict['profile']['avg'][idx_iso])
            dSI.append(dSI_sub/dSI_sum)
            print("Using Suppression Index for Difference Profiles")
        else:
            odss.append(np.asarray(dataDict['profile']['avg'][idx_orth]) - np.asarray(dataDict['profile']['avg'][idx_iso90]))
            fgm.append(np.asarray(dataDict['profile']['avg'][idx_iso90]) - np.asarray(dataDict['profile']['avg'][idx_iso]))
            dSI.append(np.asarray(dataDict['profile']['avg'][idx_orth]) - np.asarray(dataDict['profile']['avg'][idx_iso]))
            print("Using deltaBOLD for Difference Profiles")
        
        plt.figure(figNames[0])
        plt.title('%s (%d vox)' %(label, len(roi)), fontsize=8)
        plt.figure(figNames[1])
        plt.title('%s (%d vox)' %(label, len(roi)), fontsize=8)
        
    return(hprofiles, hdiff, keep_depths, keep_rois, keep_std, fgm, odss, dSI)

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


def plot_avg_depth_profile(rois,keep_depths,keep_std,statDetails,use_decon,prop_err,useSI,lcolor,fcolor):
    '''
    Plots average depth profiles across ROIs
    
    Inputs:
        rois (list): ROI depth profiles (NROIs x Nconditions x Ndepths)
        keep_depths (list): depths for each roi (NROIs x Ndepths)
        keep_std (list): standard deviations for depth profiles
            (NROIs x Nconditions x Ndepths)
        statDetails (dict): dictionary of condition details including labels 
            and colors for each condition
        use_decon (bool): use deconvolved profiles if True
        prop_err (bool): propagate errors from individual profiles if true
        useSI (bool): use suppression index rather than difference profile
        lcolor (string): color of lines in plots
        fcolor (string): color of background in plots
        
    Outputs:
        fig (matplotlib.pyplot handle): handle for figure
        stat_avg (array): average depth values for each condition 
            (Nconditions x Ndepths)
        stat_std (array): standard deviations for each condition
            (Nconditions x Ndepths)
        depth_avg_norm (array): normalized depths (Ndepths)
        dx (float): x coordinate used for legend
        dy (float): y coordinate used for legend
        odss_avg (array): orientation-dependent surround suppression average
            depth profiles (Ndepths)
        odss_std (array): orientation-dependent surround suppression standard 
            deviaitions (Ndepths)
        fgm_avg (array): figure-ground modulation average depth profiles
            (Ndepths)
        fgm_std (array): figure-ground modulation standard deviations 
            (Ndepths)
        dSI_avg (array): change in suppression index orth-iso average depth
            profiles (Ndepths)
        dSI_std (array): change in suppression index orth-iso standard 
            deviations (Ndepths)
        Ttest_odss (stats object): contains results of single-sample two-sided 
            t-test computed by scipy.stats.Ttest_1samp on ODSS depth profiles
        Ttest_fgm (stats object): contains results of single-sample two-sided
            t-test computed by scipy.stats.Ttest_1samp on FGM depth profiles
        Ttest_dSI (stats object): contains results of single-sample two-sided
            t-test computed by scipy.stats.Ttest_1samp on dSI depth profiles
            
        Dependencies:
            numpy, matplotlib, scipy
        
    '''
    
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)
        
    fig.clf()
    fsize = 14
        
    p1 = fig.add_axes([.15, .2, .3, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
    # since we're switching back and forth between decon and not decon, we'll 
    # have to re-calculate odss and fgm ...
    idx_orth = statDetails['labels'].index('orth')
    idx_iso90 = statDetails['labels'].index('iso90')
    idx_iso = statDetails['labels'].index('iso0')
    
    depth_avg = np.nanmean(np.asarray(keep_depths), axis=0) - .5
    if use_decon:
        dx = 4.
        dy = .7
        data_list = rois.copy()
    else:
        dx = 1.
        dy = .7
        data_list = rois.copy()
        
    stat_avg = np.nanmean(np.asarray(data_list), axis=0)
    if prop_err:
        stat_std = np.sqrt(np.nansum(np.asarray(keep_std)**2,axis=0))/np.shape(data_list)[0]
    else:
        stat_std = np.nanstd(np.asarray(data_list), axis=0)
    # now compute FGM and ODSS from deconvolved profiles
    odss = []
    fgm = []
    dSI = []
    for iSubj in range(len(data_list)):
        if useSI:
            odss.append((np.asarray(data_list[iSubj][idx_orth]) - np.asarray(data_list[iSubj][idx_iso90]))/(np.asarray(data_list[iSubj][idx_orth]) + np.asarray(data_list[iSubj][idx_iso90])))
            fgm.append((np.asarray(data_list[iSubj][idx_iso90]) - np.asarray(data_list[iSubj][idx_iso]))/(np.asarray(data_list[iSubj][idx_iso90]) + np.asarray(data_list[iSubj][idx_iso])))
            dSI.append((np.asarray(data_list[iSubj][idx_orth]) - np.asarray(data_list[iSubj][idx_iso]))/(np.asarray(data_list[iSubj][idx_orth]) + np.asarray(data_list[iSubj][idx_iso])))
        else:
            odss.append(np.asarray(data_list[iSubj][idx_orth]) - np.asarray(data_list[iSubj][idx_iso90]))
            fgm.append(np.asarray(data_list[iSubj][idx_iso90]) - np.asarray(data_list[iSubj][idx_iso]))
            dSI.append(np.asarray(data_list[iSubj][idx_orth]) - np.asarray(data_list[iSubj][idx_iso]))
    odss_avg = np.nanmean(np.asarray(odss), axis=0)
    fgm_avg = np.nanmean(np.asarray(fgm), axis=0)
    dSI_avg = np.nanmean(np.asarray(dSI), axis=0)
    if prop_err:
        odss_std = np.sqrt(stat_std[3,:]**2 + stat_std[2,:]**2)
        fgm_std = np.sqrt(stat_std[2,:]**2 + stat_std[1,:]**2)
        dSI_std = np.sqrt(stat_std[3,:]**2 + stat_std[1,:]**2)
    else:
        odss_std = np.nanstd(np.asarray(odss), axis=0)
        fgm_std = np.nanstd(np.asarray(fgm), axis=0)
        dSI_std = np.nanstd(np.asarray(dSI), axis=0)
    
    depth_avg_norm = (depth_avg - np.min(depth_avg))/(np.max(depth_avg) - np.min(depth_avg)) #normalized depth
    for iStat in range(len(statDetails['labels'])):
    #for iStat in [0, 1]:
        p1.plot(stat_avg[iStat, :], depth_avg_norm, color=statDetails['colors'][iStat])
        p1.fill_betweenx(depth_avg_norm,
                        stat_avg[iStat, :] - stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                        stat_avg[iStat, :] + stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                        linewidth=0.,
                        alpha=0.4,
                        color=statDetails['colors'][iStat])
        p1.text(dx, dy + iStat*.07, statDetails['labels'][iStat],
                color=statDetails['colors'][iStat],
                fontsize=fsize-2)
    p1.set_ylim([-0.02, 1.02])
    p1.set_xlim([0, 7])
    p1.set_xlabel('BOLD % change', fontsize=fsize, color=lcolor)
    p1.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
    p1.text(4, .05, 'n=%d hemis' %np.shape(rois)[0], color=lcolor, fontsize=fsize*.5, fontstyle='italic')
    
    p2 = fig.add_axes([.7, .2, .25, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
    #Calculate stats on context modulation
    Ttest_odss = stats.ttest_1samp(odss,0,axis=0)
    Ttest_fgm = stats.ttest_1samp(fgm,0,axis=0)
    Ttest_dSI = stats.ttest_1samp(dSI,0,axis=0)
    
    p2.plot([0, 0], [0, 1], '--', color='gray')
    p2.plot(odss_avg, depth_avg_norm, color='tab:green')
    p2.fill_betweenx(depth_avg_norm,
                    odss_avg - odss_std/np.sqrt(len(keep_depths)),
                    odss_avg + odss_std/np.sqrt(len(keep_depths)),
                    linewidth=0.,
                    alpha=0.4,
                    color='tab:green')
    p2.plot(fgm_avg, depth_avg_norm, color='cyan')
    p2.fill_betweenx(depth_avg_norm,
                    fgm_avg - fgm_std/np.sqrt(len(keep_depths)),
                    fgm_avg + fgm_std/np.sqrt(len(keep_depths)),
                    linewidth=0.,
                    alpha=0.2,
                    color='cyan')
    p2.plot(dSI_avg, depth_avg_norm, color='magenta')
    p2.fill_betweenx(depth_avg_norm,
                    dSI_avg - dSI_std/np.sqrt(len(keep_depths)),
                    dSI_avg + dSI_std/np.sqrt(len(keep_depths)),
                    linewidth=0.,
                    alpha=0.2,
                    color='magenta')
    p2.set_ylim([-0.02, 1.02])
    p2.set_xlim([-.3, 2.2])
    p2.text(1.15, .4-.05, 'OTSS', color='tab:green', fontsize=fsize-2)
    p2.text(1.15, .4-.12, 'FGM', color='cyan', fontsize=fsize-2)
    p2.text(1.15, .4-.19, '$\Delta$SI', color='magenta', fontsize=fsize-2)
    p2.set_yticklabels([])
    if useSI:
        p2.set_xlabel(r'$\Delta$ SI', fontsize=fsize, color=lcolor)
    else:
        p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)
    
    return(fig,stat_avg,stat_std,depth_avg_norm,dx,dy,odss_avg,odss_std,fgm_avg,fgm_std,dSI_avg,dSI_std,Ttest_odss,Ttest_fgm,Ttest_dSI)
