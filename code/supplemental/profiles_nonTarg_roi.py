#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cheryl

Note to self 31 May 2022: need to go back and correct dataTypes/GLMs so
  the same analysis can be applied to Summer 2021 data as Summer 2022.
"""
#import nibabel
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
plt.close('all')
def makeProfile1D(depthData, nDepths, paramData):
    '''
    '''
    # set up the depths
    depthMax = 21.       # we might not want to take the pial surface
    binSize = depthMax/nDepths
    print('depth bin size: %d' %binSize)
    depthBoundaries = 0.5 + np.arange(0., depthMax + 1., binSize) 
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
    '''
    kernel = np.exp(-(1/k)*np.abs(x-xloc))
    return kernel/np.sum(kernel)

def smoothen(data, x, radMax=4, nRadii=50, kernel=smooth_kernel, smooth_factor=.3):
    ''' copied from Joe's analysisROI_dev; reduced input args
    data: 1D vector of data. doesn't need to be sorted!
    smooth_factor is the exponential decay constant of the smoothing kernel
    '''
    data_smooth = np.zeros((nRadii,))
    x_smooth = np.linspace(0, radMax, nRadii)
    for iR, rad in enumerate(x_smooth):
        data_smooth[iR] = np.sum(data*kernel(x, smooth_factor, rad)) 
    return data_smooth, x_smooth

def fix_axes(ax, lcolor='black', fcolor='white'):
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
    return np.exp(-(1/k)*np.abs(np.sqrt((x-tx)**2+(y-ty)**2)))

def smoothen2D(data,xy,xy_resamp,kernel,smooth_factor):
    data_smooth = np.zeros([np.shape(xy_resamp)[1],np.shape(xy_resamp)[1]])
    for xi in range(np.shape(xy_resamp)[1]):
        for yi in range(np.shape(xy_resamp)[1]):
            data_smooth[xi,yi] = np.sum(data*kernel(xy[0],xy[1],smooth_factor,xy_resamp[0][xi],xy_resamp[1][yi]))/np.sum(kernel(xy[0],xy[1],smooth_factor,xy_resamp[0][xi],xy_resamp[1][yi]))
    return data_smooth
    
    
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code'
# datasets = glob.glob(os.path.join(mainDir, 'roi_data', 'pnr???_??_??_???????_?????.csv'))
#for now exclude pnr756 until non-target ROIs are corrected
datasets = ['/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_lh_nonTarg_rad10.csv',
 '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_rh_nonTarg_rad10.csv',
 '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V1_lh_nonTarg_rad10.csv',
 '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V1_rh_nonTarg_rad10.csv',
 '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_lh_nonTarg_rad10.csv',
 '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_rh_nonTarg_rad10.csv']
datasets.sort()
# to get pnr739_lh on right to use as demo subject
DoF_file = os.path.join(mainDir,'DoF_V1.txt') #degrees of freedom file for F-stat calculation
#import degrees of freedom
with open(DoF_file,'r') as infile:
        DoF_old = json.load(infile)
DoF = {}
roiKeys = DoF_old.keys()
for key in roiKeys: #clean up key names
    subjKeys = DoF_old[key].keys()
    DoF[key] = {}
    for subjDir in subjKeys:
        newlabel = subjDir[:6]
        DoF[key][newlabel] = DoF_old[key][subjDir]

roiRad = 1.
import pandas as pd
all_data = {}
for dataset in datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    all_data[f] = pd.read_csv(dataset, sep=',', index_col=False)
    
#%% Check stria

# check and see what the Stria profile looks like in each ROI
fig = plt.figure(num=1)
fig.clf()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    #Calculate p-vals from F-stats
    if 'nonTarg' in label:
        if 'lh' in label:
            pvals = 1-stats.f.cdf(df['ctr-sur F'],DoF['V1_nonTarg_lh'][label[:6]]['numerator'],DoF['V1_nonTarg_lh'][label[:6]]['denominator'])
        else:
            pvals = 1-stats.f.cdf(df['ctr-sur F'],DoF['V1_nonTarg_rh'][label[:6]]['numerator'],DoF['V1_nonTarg_rh'][label[:6]]['denominator'])
    else:
        if 'lh' in label:
            pvals = 1-stats.f.cdf(df['ctr-sur F'],DoF['V1_tgt_lh'][label[:6]]['numerator'],DoF['V1_tgt_lh'][label[:6]]['denominator'])
        else:
            pvals = 1-stats.f.cdf(df['ctr-sur F'],DoF['V1_tgt_rh'][label[:6]]['numerator'],DoF['V1_tgt_rh'][label[:6]]['denominator'])
    df['ctr-sur pvals'] = pvals #add pvals to dataframe  
    
    # calculate scaled distance from center of ROI (note: since we aren't fitting an ellipse, this will be different that the scale_xy for the target ROI)
    com = np.array([np.mean(df['x']),np.mean(df['y'])]) #center of mass for ROI
    [x_scl,y_scl] = [df['x'] - com[0], df['y'] - com[1]] #position relative to center
    rad_dist = np.sqrt(x_scl**2 + y_scl**2) #add scaled radial distance to the dataframe
    df['scale_xy_dist'] = rad_dist/np.max(rad_dist) #normalize the radius because each ROI will be a different size
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = roi[roi['scale_xy_dist'] > 0]
    dataDict = makeProfile1D(roi['z'].values,
                             10, 
                             roi['t1'].values)
    plt.subplot(int(np.ceil(len(all_data.keys())/2.)), 2, 1 + iR)
    plt.plot(dataDict['profile']['depth'],
             dataDict['profile']['avg'][0])
    plt.title('%s (%d vox)' %(label, len(roi)), fontsize=8)
    
#%%         
plt.figure(1)
plt.figure(2)
statDetails = {'labels': ['sur', 'iso0', 'iso90', 'orth'],
               'colors': [[.7, .7, .7], 'red', 'darkviolet', 'orange']}

profile_method = 'bin' # bin or smooth

# pick out ROIs where we're sure of localization
keep_depths = []
keep_rois = []
keep_std = []
fgm = []
odss = []
dSI = []
nDepths = 10
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
    roi = roi[roi['scale_xy_dist'] >= 0]
    
    plt.figure(1)
    plt.subplot(2, int(np.ceil(len(all_data.keys())/2.)), 1 + iR)    
    plt.figure(2)
    plt.subplot(2, int(np.ceil(len(all_data.keys())/2.)), 1 + iR)  

    if profile_method == 'bin':
        dataDict = makeProfile1D(roi['z'].values,
                                 nDepths,
                                 roi[statDetails['labels']].values)
        for iStat in range(len(statDetails['labels'])):
            plt.figure(1)
            plt.plot(dataDict['profile']['avg'][iStat],
                     dataDict['profile']['depth'],
                     color=statDetails['colors'][iStat])
            plt.fill_betweenx(dataDict['profile']['depth'],
                    dataDict['profile']['avg'][iStat] - dataDict['profile']['std'][iStat]/np.sqrt(dataDict['profile']['nVox'][iR]),
                    dataDict['profile']['avg'][iStat] + dataDict['profile']['std'][iStat]/np.sqrt(dataDict['profile']['nVox'][iR]),
                    linewidth=0.,
                    alpha=0.4,
                    color=statDetails['colors'][iStat])
            
        plt.figure(2)
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
            plt.figure(1)
            plt.plot(depth, profile, color=statDetails['colors'][iStat])
            profiles.append(profile)
        keep_depths.append(depth)
        keep_rois.append(profiles)

    # now compute FGM and ODSS
    idx_orth = statDetails['labels'].index('orth')
    idx_iso90 = statDetails['labels'].index('iso90')
    idx_iso = statDetails['labels'].index('iso0')
    odss.append(np.asarray(dataDict['profile']['avg'][idx_orth]) - np.asarray(dataDict['profile']['avg'][idx_iso90]))
    fgm.append(np.asarray(dataDict['profile']['avg'][idx_iso90]) - np.asarray(dataDict['profile']['avg'][idx_iso]))
    dSI.append(np.asarray(dataDict['profile']['avg'][idx_orth]) - np.asarray(dataDict['profile']['avg'][idx_iso]))
    
    plt.figure(1)
    plt.title('%s (%d vox)' %(label, len(roi)), fontsize=8)
    plt.figure(2)
    plt.title('%s (%d vox)' %(label, len(roi)), fontsize=8)
                
#%% Deconvolution

# measured_betas = mixing_matrix * true_betas
#
# mixing_matrix: nDepths x nDepths
# true_betas: nDepths x 1

#define point spread funct"ion
p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
Nbins = 10 #number of bins to use in this analysis
p2t = p2t_model * Nbins/Nbins_model + (Nbins_model - Nbins)/(2*Nbins_model); #adjusted peak to tail ratio for number of bins
psf = np.tril((1/p2t)*np.ones([Nbins,Nbins]))
psf[np.diag(np.diag(np.ones([Nbins,Nbins]))) == 1] = 1 #unnormalized point spread functions

normalize_psf = False #True if you want to normalize the psf by the deepest layer  

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

#%% now make some average plots

prop_err = True # do error propagation?

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
use_decon = True

# since we're switching back and forth between decon and not decon, we'll 
# have to re-calculate odss and fgm ...
idx_orth = statDetails['labels'].index('orth')
idx_iso90 = statDetails['labels'].index('iso90')
idx_iso = statDetails['labels'].index('iso0')

depth_avg = np.nanmean(np.asarray(keep_depths), axis=0) - .5
if use_decon:
    dx = 4.
    dy = .7
    data_list = decon_rois.copy()
else:
    dx = 1.
    dy = .7
    data_list = keep_rois.copy()
    
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
p1.text(4, .05, 'n=%d hemis' %len(all_data), color=lcolor, fontsize=fsize*.5, fontstyle='italic')

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
p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)
if use_decon:
    fig.savefig('task_profiles_decon.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig.savefig('task_profiles.png', facecolor=fig.get_facecolor(), edgecolor='none')

#%% Plot each context modulation effect separately

pthresh = 0.05 #pvalue significance threshold

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

for iStat in [0,1]:
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
p1.text(4, .05, 'n=%d hemis' %len(all_data), color=lcolor, fontsize=fsize*.5, fontstyle='italic')

if use_decon:
    fig.savefig('task_baselines_decon.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig.savefig('task_baselines.png', facecolor=fig.get_facecolor(), edgecolor='none')
   
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

for iStat in [1,3]:
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
p1.text(4, .05, 'n=%d hemis' %len(all_data), color=lcolor, fontsize=fsize*.5, fontstyle='italic')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
p2.plot([0, 0], [0, 1], '--', color='gray')
p2.plot(dSI_avg, depth_avg_norm, color='magenta')
p2.fill_betweenx(depth_avg_norm,
                dSI_avg - dSI_std/np.sqrt(len(keep_depths)),
                dSI_avg + dSI_std/np.sqrt(len(keep_depths)),
                linewidth=0.,
                alpha=0.2,
                color='magenta')
top = dSI_avg + dSI_std/np.sqrt(len(keep_depths))
p2.plot(top[Ttest_dSI.pvalue <= pthresh] + 0.1,depth_avg_norm[Ttest_dSI.pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
p2.set_ylim([-0.02, 1.02])
p2.set_xlim([-.3, 2.2])
p2.text(1.15, .4-.19, '$\Delta$SI', color='magenta', fontsize=fsize-2)
p2.set_yticklabels([])
p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)

if use_decon:
    fig.savefig('task_dSI_decon.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig.savefig('task_dSI.png', facecolor=fig.get_facecolor(), edgecolor='none')
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

for iStat in [1,2]:
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
p1.text(4, .05, 'n=%d hemis' %len(all_data), color=lcolor, fontsize=fsize*.5, fontstyle='italic')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
p2.plot([0, 0], [0, 1], '--', color='gray')
p2.plot(fgm_avg, depth_avg_norm, color='cyan')
p2.fill_betweenx(depth_avg_norm,
                fgm_avg - fgm_std/np.sqrt(len(keep_depths)),
                fgm_avg + fgm_std/np.sqrt(len(keep_depths)),
                linewidth=0.,
                alpha=0.2,
                color='cyan')
top = fgm_avg + fgm_std/np.sqrt(len(keep_depths))
p2.plot(top[Ttest_fgm.pvalue <= pthresh] + 0.1,depth_avg_norm[Ttest_fgm.pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
p2.set_ylim([-0.02, 1.02])
p2.set_xlim([-.3, 2.2])
p2.text(1.15, .4-.19, 'FGM', color='cyan', fontsize=fsize-2)
p2.set_yticklabels([])
p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)

if use_decon:
    fig.savefig('task_FGM_decon.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig.savefig('task_FGM.png', facecolor=fig.get_facecolor(), edgecolor='none')
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
for iStat in [2,3]:
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
p1.text(4, .05, 'n=%d hemis' %len(all_data), color=lcolor, fontsize=fsize*.5, fontstyle='italic')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
p2.plot([0, 0], [0, 1], '--', color='gray')
p2.plot(odss_avg, depth_avg_norm, color='tab:green')
p2.fill_betweenx(depth_avg_norm,
                odss_avg - odss_std/np.sqrt(len(keep_depths)),
                odss_avg + odss_std/np.sqrt(len(keep_depths)),
                linewidth=0.,
                alpha=0.2,
                color='tab:green')
top = odss_avg + odss_std/np.sqrt(len(keep_depths))
p2.plot(top[Ttest_odss.pvalue <= pthresh] + 0.1,depth_avg_norm[Ttest_odss.pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
p2.set_ylim([-0.02, 1.02])
p2.set_xlim([-.3, 2.2])
p2.text(1.15, .4-.19, 'ODSS', color='tab:green', fontsize=fsize-2)
p2.set_yticklabels([])
p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)

if use_decon:
    fig.savefig('task_ODSS_decon.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig.savefig('task_ODSS.png', facecolor=fig.get_facecolor(), edgecolor='none')
    
#%% Stats: Comparisons across depths

dSI = np.array(dSI).T
deep = np.mean([dSI[0,:],dSI[1,:],dSI[2,:]],axis=0)
middle = np.mean([dSI[3,:],dSI[4,:],dSI[5,:]],axis=0)
superficial = np.mean([dSI[6,:],dSI[7,:],dSI[8,:]],axis=0)
#[Fanova,panova] = stats.f_oneway(dSI[0,:],dSI[1,:],dSI[2,:],dSI[3,:],dSI[4,:],dSI[5,:],dSI[6,:],dSI[7,:],dSI[8,:],dSI[9,:])
[Fanova,panova] = stats.f_oneway(deep,middle,superficial)
#p_ttest = np.zeros([len(all_data.keys()),len(all_data.keys())])
p_ttest = np.zeros([3,3])
#for iR in range(len(all_data.keys())):
    #for jR in range(len(all_data.keys())):
dms = np.array([deep,middle,superficial])
for iR in range(3):
    for jR in range(3):
        p_ttest[iR,jR] = stats.ttest_ind(dms[iR,:],dms[jR,:])[1]
        
plt.imshow(p_ttest < 0.05)

#%% Estimating FF, FB, and HC contributions at each depth for each condition

# do this separately for each subject

# A = design matrix (nStim x nFactors), 
# x = concatenated responses (nFactors x nLayers), 
# y = data (nStim x nLayers)
#
# A * x = y
#  so
# x = A.I * x


ff = {'sur': 0, 'iso0': 1, 'iso90': 1, 'orth': 1}
hc = {'sur': 0, 'iso0': 1, 'iso90': 0, 'orth': 0}
fb = {'sur': 1, 'iso0': 0, 'iso90': 1, 'orth': 1}
nMod = 3
nStim = 4

# Same design for everyone
A = np.matrix(np.zeros((4, 3)))
B = np.zeros((4, 3))
for iStat, stat in enumerate(['sur', 'iso0', 'iso90', 'orth']):
    for iM, mod in enumerate([ff, hc, fb]):
        A[iStat, iM] = mod[stat]

# cherry-pick 3 depths to represent deep, middle, and superficial
if len(keep_rois[0][0]) == 10:
    depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #[1, 4, 7]
elif len(keep_rois[0][0]) == 5:
    depths = [0, 2, 4]
nDepths = len(depths)

# a matrix to catch the answers
nSubj = len(keep_rois)
answers = np.zeros((nMod, nDepths, nSubj))

# we'll see if our work is plausible by reconstructing
inputs = np.zeros((nStim, nDepths, nSubj))
reconstructions = np.zeros((nStim, nDepths, nSubj))
input_variance = np.zeros((nSubj,))
residual_variance = np.zeros((nSubj,))

if use_decon: # set in the average profile cell
    data_list = decon_rois.copy()
else:
    data_list = keep_rois.copy()

# cruise through and do the regressions
for iSubj in range(nSubj):
    y = np.matrix(np.zeros((nStim, nDepths)))
    for iStat, stat in enumerate(statDetails['labels']):
        for iD, depth in enumerate(depths):
            y[iStat, iD] = data_list[iSubj][iStat][depth]
    input_variance[iSubj] = np.var(y)
    x = A.I*y
    answers[:, :, iSubj] = x
    # reconstruct profiles from answers
    yest = A*x
    residual_variance[iSubj] = np.var(yest - y)
    reconstructions[:, :, iSubj] = yest
    inputs[:, :, iSubj] = y
    # plt.subplot(nSubj, 2, iSubj*2 + 1)
    # plt.plot(y.transpose())
    # plt.subplot(nSubj, 2, iSubj*2 + 2)
    # plt.plot(yest.transpose())
    
#see what we got for x!
modColors = ['lightgray', 'lightgreen', 'cyan']
# for iSubj in range(nSubj):
#     plt.subplot(2, 7, iSubj+1)
#     plt.imshow(answers[:, :, iSubj].transpose(), vmin=-3, vmax=3, cmap='plasma')
#     plt.title('%2.2f / %2.2f' %(input_variance[iSubj], residual_variance[iSubj]), fontsize=8)
#     plt.axis('off')

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6, 4))
fig.patch.set_facecolor(fcolor)

ax = plt.subplot(1, 2, 1)
ax.imshow(A.T, cmap='gray', vmin=-.5, vmax=2)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['ff', 'hc', 'fb'], color=lcolor)
ax.set_xticks(range(4))
ax.set_xticklabels(statDetails['labels'], color=lcolor)
ax.set_title('avg %d%% var explained' %(int(100*(1-np.mean(np.array(residual_variance)/np.array(input_variance))))),
             color=lcolor)

ax = plt.axes([.7, .2, .25, .7])
ax.plot([0, 0], [0, 1], '--', color='gray', linewidth=.5)
for iM, mod in enumerate(['ff', 'hc', 'fb']):
    ax.plot(np.mean(answers[iM, :, :], axis=1).transpose(),
             np.linspace(0, 1, nDepths),
             color=modColors[iM])
    ax.text(2.2, .5-.1*iM, mod, color=modColors[iM])
ax.set_xlim([-.5, 2.5])
ax.set_xticks(np.arange(0, 2.5, 1.))
ax.set_xticklabels(np.arange(0, 2.5, 1.), color=lcolor)
ax.set_ylabel('Relative depth', color=lcolor)
ax.set_xlabel('% change', color=lcolor)
fix_axes(ax, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    title = 'model on deconvolved data'
else:
    title = 'model on orig data'
ax.set_title('Estimated weights', color=lcolor)

fig.savefig('model_fits.png', facecolor=fig.get_facecolor(), edgecolor='none')
#%% An iteration to see what kind of A for middle layers does best prediction
plt.figure()
ff = {'sur': 0, 'iso0': 1, 'iso90': 1, 'orth': 1}
hc = {'sur': 0, 'iso0': 1, 'iso90': 0, 'orth': 0}
fb = {'sur': 1, 'iso0': 0, 'iso90': 1, 'orth': 1}
nMod = 3
nStim = 4
if use_decon: # set in the avg profile cell
    data_list = decon_rois.copy()
else:
    data_list = keep_rois.copy()

# Same design for everyone
Abase = np.zeros((4, 3))
for iStat, stat in enumerate(statDetails['labels']):
    for iM, mod in enumerate([ff, hc, fb]):
        Abase[iStat, iM] = mod[stat]
residual_variance = np.zeros((16, nSubj))
iA = -1
for iSur in range(2):
    for iIso in range(2):
        for iIso90 in range(2):
            for iOrth in range(2):
                A = Abase.copy()
                A[:, 1] = [iSur, iIso, iIso90, iOrth]
                A = np.matrix(A)
                iA += 1
                # cruise through and do the regressions
                answers = np.zeros((nMod, nDepths, nSubj))
                for iSubj in range(nSubj):
                    y = np.matrix(np.zeros((nStim, nDepths)))
                    for iStat, stat in enumerate(statDetails['labels']):
                        for iD, depth in enumerate(depths):
                            y[iStat, iD] = data_list[iSubj][iStat][depth]
                    if iA == 0:
                        input_variance[iSubj] = np.var(y)
                    x = A.I*y
                    answers[:, :, iSubj] = x
                    # reconstruct profiles from answers
                    yest = A*x
                    residual_variance[iA, iSubj] = np.var(yest - y)
                ax = plt.subplot(4, 8, 2*iA + 1)
                ax.imshow(A)
                ax.set_title('%.2f' %np.mean(residual_variance[iA, :]))
                if iA == 0:
                    ax.set_yticks(range(4))
                    ax.set_yticklabels(hc.keys())
                ax = plt.subplot(4, 8, 2*iA + 2)
                for iM, mod in enumerate(['ff', 'hc', 'fb']):
                    ax.plot(np.mean(answers[iM, :, :], axis=1).transpose(),
                             np.linspace(0, 1, nDepths),
                             color=modColors[iM])
                ax.set_xlim([-3, 3])
    

#%% Show patches
# In this version, I will not be fitting an ellipse since there is no center-surround representation in the ROI

purple = np.array([.5, 0., 0.5])
yellow = np.array([1., 1., 0.])
middle = (yellow + purple)/2
# had a hard time deciding what color to use for radial distance
white = np.array([1., 1., 1.])
red = np.array([1., 0., 0.])
blue = np.array([0., 0., 1.])
gray = np.array([.5, .5, .5])
cmax = 5.
fig = plt.figure(figsize=(8,8))
fig.set_size_inches((8, 4))
fontsize = 8
fig.patch.set_facecolor(fcolor)

patchwidth = .13
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    # show localizer data
    minx = np.min(df['x'].values)
    miny = np.min(df['y'].values)
    aw = .8/len(all_data)
    ax = plt.axes([.07 + iR*1.1*aw, .5,  aw, .26])
    for iV, beta in enumerate(df['ctr-sur']):
        if df['ctr-sur F'][iV] > 6.66:
            weight = np.min((np.max((beta, -cmax)), cmax))/cmax
            if weight > 0:
                color = middle + weight*(yellow - middle)
            else:
                color = middle - weight*(purple - middle)
            ax.plot(df['x'][iV], df['y'][iV], '.', color=color)
    ax.set_xlim([minx, minx+18])
    ax.set_ylim([miny, miny+18])
    ax.patch.set_facecolor(fcolor)
    plt.axis('off')
    ax.set_title(label, fontsize=fontsize, color=lcolor)
    
fig.savefig('patches.png', facecolor=fig.get_facecolor(), edgecolor='none')

#%% Visualize Betas across Surface
import scipy.interpolate as sciInterp

red = np.array([1,0,0]); #color
cmax = 10.; #max beta weight
patchwidth = .13
cond = 'iso0'
pthresh = 0.01
fig = plt.figure(figsize=(8,8))
fig.set_size_inches((8, 4))
fontsize = 8
fig.patch.set_facecolor(fcolor)

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    # Plot iso
    minx = np.min(df['x'].values)
    miny = np.min(df['y'].values)
    aw = .8/len(all_data)
    ax = plt.axes([.07 + iR*1.1*aw, .5,  aw, .26])
    for iV, beta in enumerate(df[cond]):
        if df['ctr-sur pvals'][iV] < pthresh:
            weight = np.min((np.max((beta, -cmax)), cmax))/cmax
            weight = (weight + 1)/2; #make value between 0 and 1
            if weight > 0:
                color = weight*red
            else:
                color = weight*red
            ax.plot(df['x'][iV], df['y'][iV], '.', color=color)
    ax.set_xlim([minx, minx+18])
    ax.set_ylim([miny, miny+18])
    ax.patch.set_facecolor(fcolor)
    plt.axis('off')
    ax.set_title(label, fontsize=fontsize, color=lcolor)
    
#Smooth 2D space with kernel
fig2 = plt.figure()
smooth_factor = 0.5
resamp_factor = 100
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    # Plot iso
    minx = np.min(df['x'].values)
    maxx = np.max(df['x'].values)
    miny = np.min(df['y'].values)
    maxy = np.max(df['y'].values)
    x_interp = np.linspace(minx,maxx,resamp_factor)
    y_interp = np.linspace(miny,maxy,resamp_factor)
    # beta_interp = sciInterp.interp2d(df['x'].values,df['y'].values,df[cond].values,kind='linear')
    # beta_resamp = beta_interp(x_interp,y_interp)
    # beta_resamp[beta_resamp > cmax] = cmax #truncate
    # beta_resamp[beta_resamp < -cmax] = -cmax
    
    # 2D blurring
    betas = df[cond].values[df['ctr-sur pvals'] < pthresh]
    betas_x = df['x'].values[df['ctr-sur pvals'] < pthresh]
    betas_y = df['y'].values[df['ctr-sur pvals'] < pthresh]
    beta_resamp = smoothen2D(betas,np.array([betas_x,betas_y]),np.array([x_interp,y_interp]),smooth_kernel2D,smooth_factor)

    # plot
    plt.subplot(1,len(all_data),iR+1)
    plt.imshow(beta_resamp,cmap='Reds')
    plt.colorbar()
    plt.title(label+': '+cond)

#%% Across surface!
# smoothing according to line 441 in analysisROI_dev
#with smoothing
plot_orig_data = True #plot original data?
plot_Fstat = False #True #plot fstat?

fig1 = plt.figure(figsize=(8.75, 4))
fig1.set_size_inches((8, 4))
fontsize = 8
fig1.patch.set_facecolor(fcolor)


radMax = 1
nRadii = 20
ymax = 6
highlight = ['sur', 'iso0'] #['orth', 'iso90']#'orth'
depth_labels = ['deep', 'middle', 'superficial']
all_profiles = {depth_label: {label: [] for label in statDetails['labels']} for depth_label in depth_labels}

for iR, label in enumerate(all_data.keys()):
    for iD in range(3):
        df = all_data[label]
        if iD == 0:
            df = df[(df['z'] >=1) & (df['z'] <3)]
        elif iD == 1:
            df = df[(df['z'] >=4) & (df['z'] <6)]
        elif iD == 2:
            df = df[(df['z'] >=6) & (df['z'] <=9)]
        aw = .8/len(all_data)
        plt.figure(1)
        ax = plt.axes([.07 + iR*1.1*aw, .14 + iD*.3,  aw, .2])
        ax2 = ax.twinx()
        ax2.yaxis.set_visible(False)
        for iStat in range(len(statDetails['labels'])):
            alpha = 1.
            if highlight:
                if statDetails['labels'][iStat] in highlight:
                    alpha = 1.
                else:
                    alpha = 0.2
            coef = df[statDetails['labels'][iStat]].values
            x = df['scale_xy_dist'].values
            #smooth data
            coef_smooth, x_smooth = smoothen(coef, x)
            if iR < 5:
                all_profiles[depth_labels[iD]][statDetails['labels'][iStat]].append(coef_smooth)
            # dataDict = makeProfile1D(x, nRadii, coef)
            # plt.plot(dataDict['profile']['depth'],
            #          dataDict['profile']['avg'][0],
            #          '--',
            #          color=statDetails['colors'][iStat])

            ax.plot(x_smooth, coef_smooth, color=statDetails['colors'][iStat], alpha=alpha)
            ax.set_ylim([0, ymax])
            ax.set_xlim([0, radMax])
            # ax.set_xticks([0, 1, 2, 3])
            if iD == 2:
                ax.set_title(label, fontsize=8, color=lcolor) 
            ax.plot([2, 2], [0, ymax], '--', color='gray', alpha=0.15)
            fix_axes(ax, lcolor=lcolor, fcolor=fcolor)
            if iR > 0:
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
    
            else:
                plt.ylabel(r'$\beta$, %s' %depth_labels[iD], fontsize = 8, color=lcolor)
                if iD == 2:
                    ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
                    ax.set_yticklabels(['0', '2', '4', '6', '8', '10', '12'], fontsize=8, color=lcolor)
                else:
                    ax.set_yticks([0, 2, 4, 6])
                    ax.set_yticklabels(['0', '2', '4', '6'], fontsize=8, color=lcolor)
            # ax.set_xticklabels(['0', '', '2', ''], fontsize=8, color=lcolor)
            if iR == 0:
                if iD == 0:
                    ax.set_xlabel("norm dist from ROI center", fontsize = 8, color=lcolor)
            else:
                ax.set_xticklabels([])
                
            # Now plot ctr - sur Fstat across surface
            if plot_Fstat:
                Fstat = df['ctr-sur F'].values
                x = df['scale_xy_dist'].values
                #smooth data
                Fstat_smooth, x_smooth = smoothen(Fstat, x)
    
                ax2.plot(x_smooth, Fstat_smooth, color='black', alpha=alpha)
                ax2.set_ylim([0, 40])
                # ax2.set_xticks([0, 1, 2, 3])
                if iD == 3:
                    ax2.set_title(label, fontsize=8, color=lcolor) 
                ax2.plot([2, 2], [0, ymax], '--', color='gray', alpha=0.15)
                fix_axes(ax2, lcolor=lcolor, fcolor=fcolor)
                if iR > 0:
                    ax2.yaxis.set_visible(False)
                    ax2.spines['right'].set_visible(False)
                
fig1.savefig('surface_profiles.png', facecolor=fig.get_facecolor(), edgecolor='none')
#%% Average surface profiles

fig = plt.figure(figsize=(4, 6))
fig.set_size_inches((4, 6))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12

for iD, depth_label in enumerate(all_profiles.keys()):
    print(depth_label)
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    for iStat, label in enumerate(statDetails['labels']):
        nProfiles = len(all_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [0, 6], '--', color=(.3, .3, .3))

        p.plot(x_smooth, stat_avg, color=statDetails['colors'][iStat])
        p.fill_between(x_smooth,
                        stat_avg - stat_std/np.sqrt(nProfiles),
                        stat_avg + stat_std/np.sqrt(nProfiles),
                        linewidth=0.,
                        alpha=0.4,
                        color=statDetails['colors'][iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, statDetails['labels'][iStat],
                color=statDetails['colors'][iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('norm distance from ROI center', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([0, 6])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        p.set_xlim([0, radMax])

fig.savefig('avg_surf_profiles.png', facecolor=fig.get_facecolor(), edgecolor='none')