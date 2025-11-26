#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cheryl

This code combines work from Cheryl and Joe for creating laminar and surface
profiles from the OriSeg data. This code specifically focuses on the analysis 
of contextual conditions.
"""
#import nibabel
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
from statsmodels.stats.multitest import multipletests

#Import custom functions
from oriseg_funcs_v3 import *

plt.close('all')
    
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]
savefigs = False #True #if true save all figures
figDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code/figs/'
fig_format = 'svg'
statCorrType = 'bonferroni'

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code'
#datasets = glob.glob(os.path.join(mainDir, 'roi_data', 'pnr???_??_??_?????.csv'))
datasets = [#'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_lh_target_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr256_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr256_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr328_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr328_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr510_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr510_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr739_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr739_V1_tgt_rh_rad10.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target/pnr756_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr756_V1_tgt_rh_rad10.csv'
  ]
datasets.sort()

roiRad = 2 #1.
import pandas as pd
all_data = {}
for dataset in datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    all_data[f] = pd.read_csv(dataset, sep=',', index_col=False)
    
## THIS IS A HACKY FIX TO GET RID OF DEPTH = 0 VOXELS; SHOULD REMOVE THIS IN THE FUTURE
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    df = df.drop(df[df['d'] == 0].index)
    
    all_data[label] = df

# check and see what the Stria profile looks like in each ROI
nDepths = 7
fig = plt.figure(num=1)
fig.clf()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = roi[roi['scale_xy_dist'] > 0]
    dataDict = makeProfile1D(roi['d'].values,
                             8, #number of depths
                             roi['t1'].values,
                             np.min(roi['d'].values), #min depth value
                             np.max(roi['d'].values), #max depth value
                             True) #Use LayNii values
    
    plt.subplot(int(np.ceil(len(all_data.keys())/2.)), 2, 1 + iR)
    plt.plot(dataDict['profile']['depth'],
             dataDict['profile']['avg'][0])
    plt.title('%s (%d vox)' %(label, len(roi)), fontsize=8)
    
if savefigs:
    fig.savefig(os.path.join(figDir,'t1w_profiles.%s' %(fig_format)))

#%% Check on xy to uv mapping
from matplotlib.patches import Ellipse

frad = plt.figure(figsize=(6.5,8))
floc = plt.figure(figsize=(6.5,8))
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]

    # recapitulate the fitting, which is a bit of overkill, but it gets us 
    # an accurate ellipse
    tgt_df = df[df['ctr-sur'] > 0]
    #tgt_df = df[df['scale_xy_dist'] <= 2]
    cov = np.cov(tgt_df['x'][df['scale_xy_dist'] < 2.2],
                 tgt_df['y'][df['scale_xy_dist'] < 2.2])
    com = (np.mean(tgt_df['x'][df['scale_xy_dist'] < 2.2]),
           np.mean(tgt_df['y'][df['scale_xy_dist'] < 2.2]))
    a = (cov[0,0] + cov[1,1])/2 + np.sqrt(((cov[0,0] - cov[1,1])/2)**2 + cov[0,1]**2)
    b = (cov[0,0] + cov[1,1])/2 - np.sqrt(((cov[0,0] - cov[1,1])/2)**2 + cov[0,1]**2)
    print('avg radius: %2.2f' %(2*(np.sqrt(a) + np.sqrt(b))/2))
    theta = np.arctan2(a - cov[0,0], cov[1,0])
    ellipse = Ellipse(com,
                      width=2*2*np.sqrt(a),
                      height=2*2*np.sqrt(b),
                      angle=180*theta/np.pi,
                      zorder=100, alpha=1., edgecolor='r', facecolor='None')
    # show localizer data
    minx = np.min(df['x'].values)
    miny = np.min(df['y'].values)
    ax = frad.add_subplot(int(np.ceil(len(datasets)/2)),2,iR+1)
    
    # Plot the radius determined by the normalized uv coordinates (this should be in SD of a 2D Gaussian fitted to the loc data)
    cmap = plt.cm.get_cmap('viridis')
    pcm = ax.scatter(df['x'],df['y'],c=df['scale_xy_dist'],s=10,cmap=cmap)
    plt.colorbar(pcm,ax=ax)
    ax.add_patch(ellipse)
    ax.patch.set_facecolor('r')
    ax.set_title(label+" radius: SD<2 Nvox = %d" %(np.sum(df['scale_xy_dist']<2)),fontsize=6)
    ax.axis('off')
    
    # Plot the ctr-sur betas
    floc = plt.figure(floc)
    cmap_rev = cmap.reversed()
    plt.subplot(int(np.ceil(len(datasets)/2)),2,iR+1)
    plt.scatter(df['x'],df['y'],c=df['ctr-sur'],s=10,cmap=cmap_rev)
    plt.colorbar()
    plt.title(label+" localizer",fontsize=6)
    plt.tight_layout(pad=0.1)
    
if savefigs:
    frad.savefig(os.path.join(figDir,'xy_map_rad.%s' %(fig_format)))
    floc.savefig(os.path.join(figDir,'xy_map_loc.%s' %(fig_format)))

#%% Histograms of p-values
floc = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(int(np.ceil(len(datasets)/2)),2,iR+1)
    plt.hist(roi['loc p-val'].values,bins=20,density=True)
    plt.title(label+" loc p-val",fontsize=8)
    plt.xlabel("pval")
    plt.text(0.8,5,'< 0.05 = %d %%' %(100*np.sum(roi['loc p-val'] <= 0.05)/len(roi['loc p-val'])))
    plt.ylim([0,10])
    plt.xlim([0,1])
floc.tight_layout(pad=0.1)

ftask = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    if 'task pval' in roi.keys():
        roi = roi.rename(columns={'task pval':'task p-val'})
    plt.subplot(int(np.ceil(len(datasets)/2)),2,iR+1)
    plt.hist(roi['task p-val'].values,bins=20,density=True)
    plt.title(label+" task p-val",fontsize=8)
    plt.xlabel("pval")
    plt.text(0.8,5,'< 0.05 = %d %%' %(100*np.sum(roi['task p-val'] <= 0.05)/len(roi['task p-val'])))
    plt.ylim([0,10])
    plt.xlim([0,1])
ftask.tight_layout(pad=0.1)

if savefigs:
    floc.savefig(os.path.join(figDir,'pvals_loc.%s' %(fig_format)))
    ftask.savefig(os.path.join(figDir,'pvals_task.%s' %(fig_format)))

#%% Depth Histograms
# I want to see how much coverage we are getting through depth.
fdhist = plt.figure(figsize=(15,4))

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = df[df['scale_xy_dist'] < roiRad]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,int(np.ceil(len(datasets)/2)),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.title(label)
    plt.xlabel("Normalize Depth WM -> GM")
    plt.ylabel("Num. Voxels")
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
fdhist.tight_layout(pad=0.0)

if savefigs:
    fdhist.savefig(os.path.join(figDir,'depth_hist.%s' %(fig_format)))
    
#%% Use the deepest layer as a proxy for non-vein contaminated voxels
# Then define the threshold based on this distribution

deep_pct = 10 #percentile to call deep layers
conditions = ['iso0','iso90','orth','sur']
depth_groups = {'deep': [0.2,0.4], 'middle': [0.4,0.6], 'superficial': [0.6,0.8]}
depth_labels = ['superficial','middle','deep'] #put them in the right order
depth_var = 'd'
x_var = 'x'
y_var = 'y'
sd_thresh = 2 #how many st. dev. of the deep layer mean to use as the threshold
mask_dict = {} #create a mask dictionary
lmnv_dict = {key:{'mean':0,'std':0,'thresh':0,'deep_mean':0,'deep_std':0} for key in all_data.keys()} #thresh dictionary
fsize=8 #fontsize of title
depth_groups = {'deep': [0.2,0.4], 'middle': [0.4,0.6], 'superficial': [0.6,0.8]}
depth_labels = ['superficial','middle','deep'] #put them in the right order
Ngroups = len(depth_groups.keys())
NROIs = len(all_data.keys())
for k_i, key in enumerate(all_data.keys()):

    # calculate log(MNV)
    df = all_data[key]
    lmnv = get_lmnv(df,key='stdev_xerrts') #log of the mean-normalized variance
    mnv = np.exp(lmnv) #get back mean normalized variance
    
    # get deep layer distribution
    z = df[depth_var]
    [deep_mean, deep_std, deep] = get_deep_layer_dist(df,depth_var,deep_pct)
    lmnv_dict[key]['deep_mean'] = deep_mean
    lmnv_dict[key]['deep_std'] = deep_std
    
    # define threshold based on deep layer distribution
    [mnv_mask, lmnv_thresh] = get_mnv_mask(df,depth_var,deep_pct,sd_thresh)
    lmnv_dict[key]['thresh'] = lmnv_thresh
    lmnv_dict[key]['mean'] = np.mean(lmnv)
    lmnv_dict[key]['std'] = np.std(lmnv)
    mask_dict[key] = mnv_mask
    all_data[key]['no_vein'] = mnv_mask
    
    # Plot distributions
    fthresh = plot_mnv_histograms(lmnv, lmnv[deep], mnv_mask, deep_pct, key, k_i, NROIs, fsize, pad=0.0, figsize=(15,3))
    
    # Plot depth maps
    dmap = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0,100], fsize, fname = 'dmap', pad=0.0)
        
    #plot thresholded map
    dmap_thresh = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0, 100], fsize, fname='dmap_thresh', mask=mnv_mask, pad=0.0)
        
    # Plot voxel loss at each depth after masking
    fdepth_hist = plot_depth_voxel_loss(z, mnv_mask, nDepths, NROIs, key, k_i, fsize)
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(mnv_mask),np.size(mnv),key))
    
if savefigs:
    fthresh.savefig(os.path.join(figDir,'mnv_hist.%s' %(fig_format)))
    dmap.savefig(os.path.join(figDir,'mnv_depth_map.%s' %(fig_format)))
    dmap_thresh.savefig(os.path.join(figDir,'mnv_depth_map_thresh.%s' %(fig_format)))
    fdepth_hist.savefig(os.path.join(figDir,'mnv_depth_hist.%s' %(fig_format)))

#%% Compare thresholds between subjects

f = plt.figure()
for k_i, key in enumerate(all_data.keys()):
    plt.errorbar(k_i,lmnv_dict[key]['mean'],lmnv_dict[key]['std'],color='b',linestyle='None',marker='o')
    plt.errorbar(k_i+0.1,lmnv_dict[key]['deep_mean'],lmnv_dict[key]['deep_std'],color='orange',linestyle='None',marker='o')
    plt.plot(k_i+0.2,lmnv_dict[key]['thresh'],color='r',marker='s')
plt.xticks(np.arange(0,len(all_data.keys())),all_data.keys(),rotation=15,fontsize=6)
plt.ylabel("log(MNV)")
#plt.legend(['full','thresh'])

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_errorbars.%s' %(fig_format)))

#try violin plots
spreadF = 2
f = plt.figure()
for k_i, key in enumerate(all_data.keys()):
    df = all_data[key]
    lmnv = get_lmnv(df,key='stdev_xerrts') #log of the mean-normalized variance
    [deep_mean, deep_std, deep] = get_deep_layer_dist(df,depth_var,deep_pct)
    violin_parts = plt.violinplot(lmnv,positions=[spreadF*k_i],showmedians=True)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('b')
        pc.set_edgecolor('b')
    for pc in violin_parts:
        if not isinstance(violin_parts[pc],list):
            violin_parts[pc].set_edgecolor('b')
    violin_parts = plt.violinplot(lmnv[deep],positions=[spreadF*(k_i+0.2)],showmedians=True)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('orange')
        pc.set_edgecolor('orange')
    for pc in violin_parts:
        if not isinstance(violin_parts[pc],list):
            violin_parts[pc].set_edgecolor('orange')
    plt.hlines(lmnv_dict[key]['thresh'],spreadF*k_i-0.5,spreadF*(k_i+0.2)+0.5,color='r')
plt.xticks(np.arange(0,spreadF*len(all_data.keys()),spreadF),all_data.keys(),rotation=15,fontsize=6)
plt.ylabel("log(MNV)")

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin.%s' %(fig_format)))
    
#%% Apply full model p-val mask if desired

use_fullmodel_mask = True
pthresh_fullmodel = 0.05
if use_fullmodel_mask:
    for k_i, key in enumerate(all_data.keys()):
        df = all_data[key]
        pvals = df['task p-val']
        pval_mask = pvals < pthresh_fullmodel
        print("%d/%d voxels survive full model p-val mask" %(np.sum(pval_mask),np.size(pval_mask)))
        mask_dict[key] = mask_dict[key] & pval_mask   
        all_data[key]['sig'] = pval_mask

#%% Plot Depth Profiles
    
statDetails = {'labels': ['sur', 'iso0', 'iso90', 'orth', 'ctr_unwarp', 'sur_unwarp', 'ctr-sur_unwarp'],
                'colors': [[.7, .7, .7], 'red', 'darkviolet', 'orange', 'gold', 'purple', 'coral']}
diffDetails = {}
diffDetails['statIDs'] = {'odss': ['orth','iso90'],
                          'fgm': ['iso90','iso0'],
                          'dsi': ['orth','iso0'],
                          'iso-sur': ['iso0','sur'],
                          'ctr-sur': ['ctr_unwarp','sur_unwarp']}
diffDetails['colors'] = ['green','magenta','cyan','black','coral']
profile_method = 'bin' # bin or smooth
#pick out ROIs where we're sure of localization
roi_dict = {}
for key in all_data.keys():
    df = all_data[key]
    roi_dict[key] = df['scale_xy_dist'] < roiRad
    all_data[key]['in_tgt'] = df['scale_xy_dist'] < roiRad
useSI = False #use suppression index rather than differences (cond1 - cond2 / cond1 + cond2)
#create full masks
masks = {roi:all_data[roi]['in_tgt']*all_data[roi]['sig']*all_data[roi]['no_vein'] for roi in all_data.keys()}
depthProfiles = compute_all_depth_profiles(all_data,statDetails,profile_method,nDepths,masks,depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
diffProfiles = compute_diff_profiles(all_data,statDetails,diffDetails['statIDs'],profile_method,nDepths,useSI,masks,depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
      
#%% Centroid plots
# Let's take a look at raw voxel betas across depth by condition

Nsubj = len(all_data.keys())

#plot centroids for each condition and ROI
plot_centroids(all_data, masks, statDetails, roiRad, nDepths=nDepths)
        
#calculate difference profiles
plot_centroids_diff(all_data, masks, statDetails, diffDetails, roiRad, nDepths)
          
if savefigs:
    for l in statDetails['labels']:
        plt.figure(l)
        plt.savefig(os.path.join(figDir,'centroids_%s.%s' %(l,fig_format)))
    for l in diffDetails['statIDs'].keys():
        plt.figure(l)
        plt.savefig(os.path.join(figDir,'centroids_%s.%s' %(l,fig_format)))
        
#%% Individual Subjects Histogram

roiParam = 'scale_xy_dist'
for iStat in range(len(statDetails['labels'])):
    fig, ax = plt.subplots(3,len(all_data.keys()),figsize=[14,5])
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = mask_dict[label]
        roi = df[roi_idx]
        if radParam != None:
            roi = df[df[radParam] < roiRad] # only very center, to be sure!
            roi = roi[roi[radParam] >= 0] #don't allow negative radii   
        maxDepth = np.max(roi['d'])
        minDepth = np.min(roi['d'])
        normDepths = (roi['d']-minDepth)/(maxDepth-minDepth)
        dataDict = makeProfile1D(roi['d'].values,
                             3, #number of depths
                             roi[statDetails['labels']].values,
                             np.min(roi['d'].values), #min depth value
                             np.max(roi['d'].values), #max depth value
                             True) #Use LayNii values
        avg_normDepths = (dataDict['profile']['depth'] - np.nanmin(dataDict['profile']['depth']))/(np.nanmax(dataDict['profile']['depth']) - np.nanmin(dataDict['profile']['depth']))
        
        #Plot
        minX = -15
        maxX = 15
        d_bins = np.flip(np.linspace(0,1,4))
        d_labels = ['superficial','middle','deep']
        for d_i, d in enumerate(d_bins[0:3]):
            d_mask = (d_bins[d_i+1] <= normDepths) & (normDepths < d_bins[d_i])
            bins = np.linspace(minX,maxX,30)
            ax[d_i,iR].hist(roi[statDetails['labels'][iStat]][d_mask],bins=bins,color=statDetails['colors'][iStat])
            legend_label = [np.sum(d_mask)]
            ax[d_i,iR].legend(legend_label,fontsize=4)
            ax[d_i,0].set_ylabel(d_labels[d_i])
            ax[d_i,iR].set_xlim([minX,maxX])
        ax[0,iR].set_title(label+' '+statDetails['labels'][iStat],fontsize=6)
    plt.tight_layout()
        
for iDiff, Diff in enumerate(diffDetails['statIDs'].keys()):
    fig, ax = plt.subplots(3,len(all_data.keys()),figsize=[14,5])
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = mask_dict[label]
        roi = df[roi_idx]
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
        
        #Plot
        minX = -10
        maxX = 10
        d_bins = np.flip(np.linspace(0,1,4))
        d_labels = ['superficial','middle','deep']
        for d_i, d in enumerate(d_bins[0:3]):
            d_mask = (d_bins[d_i+1] <= normDepths) & (normDepths < d_bins[d_i])
            bins = np.linspace(minX,maxX,30)
            ax[d_i,iR].hist(diff[d_mask],bins=bins,color=diffDetails['colors'][iDiff])
            legend_label = [np.sum(d_mask)]
            ax[d_i,iR].legend(legend_label,fontsize=4)
            ax[d_i,0].set_ylabel(d_labels[d_i])
            ax[d_i,iR].set_xlim([minX,maxX])
        ax[0,iR].set_title(label+' '+statDetails['labels'][iStat],fontsize=6)
    plt.tight_layout()
        
#%% Deconvolution

#reformat data to fit decon_rois specs
keep_rois = np.zeros((NROIs,len(statDetails['labels']),nDepths))
for iR, roiID in enumerate(all_data.keys()):
    for iStat, stat in enumerate(statDetails['labels']):
        keep_rois[iR,iStat,:] = depthProfiles[stat]['avg'][iR]
        
keep_diffs = np.zeros((NROIs,len(diffDetails['statIDs'].keys()),nDepths))
for iR, roiID in enumerate(all_data.keys()):
    for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
        keep_diffs[iR,iDiff,:] = diffProfiles[diff]['avg'][iR]

#define point spread function
p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
Nbins = nDepths #number of bins to use in this analysis

normalize_psf = False #True if you want to normalize the psf by the deepest layer  

decon_rois = depth_deconv(keep_rois,p2t_model,Nbins_model,Nbins,normalize_psf)
decon_diffs = depth_deconv(keep_diffs,p2t_model,Nbins_model,Nbins,normalize_psf)

#now put back in dictionary
for iStat, stat in enumerate(statDetails['labels']):
    depthProfiles[stat]['avg_decon'] = np.squeeze(np.array(decon_rois)[:,iStat,:])
    
for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
    diffProfiles[diff]['avg_decon'] = np.squeeze(np.array(decon_diffs)[:,iDiff,:])

#%% now make some average plots

prop_err = False # do error propagation?
use_decon = True
useSI = False

taskStats = statDetails['labels'][0:4]
taskColors = statDetails['colors'][0:4]
taskDiffs = list(diffDetails['statIDs'].keys())[0:4]
locStats = statDetails['labels'][4:]
locColors = statDetails['colors'][4:]
locDiffs = list(diffDetails['statIDs'].keys())[4:]

[avgTaskProfiles, avgTaskDiffs] = compute_avg_depth_profile(depthProfiles,statDetails,diffDetails['statIDs'],taskStats,taskDiffs,use_decon,prop_err,useSI)  
[avgLocProfiles, avgLocDiffs] = compute_avg_depth_profile(depthProfiles,statDetails,diffDetails['statIDs'],locStats,locDiffs,use_decon,prop_err,useSI)

# Plot task average profiles
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7

ylim = [-0.02,1.02]
xlim = [0,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,taskStats,taskColors,ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.3,1.5]
plot_avg_diff_profile(p2,avgTaskDiffs,taskDiffs,diffDetails['colors'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI)

# Plot loc average profiles
fig2 = plt.figure(figsize=(6, 4))
fig2.set_size_inches((6,4))
fig2.patch.set_facecolor(fcolor)
    
fig2.clf()
fsize = 14
    
p1 = fig2.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 4.
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.7,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgLocProfiles,locStats,locColors,ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig2.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.3,1.5]
    
plot_avg_diff_profile(p2,avgLocDiffs,locDiffs,diffDetails['colors'][4:],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI)


if savefigs:
    if use_decon:
        fig.savefig(os.path.join(figDir,'avg_profiles_task_deconv.%s' %(fig_format)))
        fig2.savefig(os.path.join(figDir,'avg_profiles_loc_deconv.%s' %(fig_format)))
    else:
        fig.savefig(os.path.join(figDir,'avg_profiles_task.%s' %(fig_format)))
        fig2.savefig(os.path.join(figDir,'avg_profiles_loc.%s' %(fig_format)))
        
#%% Plot each context modulation effect separately

pthresh = 0.05 #pvalue significance threshold

#iso and surround only
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

ylim = [-0.02,1.02]
xlim = [0,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['sur','iso0'],[[0.7,0.7,0.7],'red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#iso-sur
xlim = [-0.8,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    if use_decon:
        fig.savefig(os.path.join(figDir,'avg_profiles_iso_sur_deconv.%s' %(fig_format)))
    else:
        fig.savefig(os.path.join(figDir,'avg_profiles_iso_sur.%s' %(fig_format)))

#iso and orth
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

ylim = [-0.02,1.02]
xlim = [0,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['iso0','orth'],['red','orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#dSI
xlim = [-0.8,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['dsi'],['tab:cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    if use_decon:
        fig.savefig(os.path.join(figDir,'avg_profiles_dsi_deconv.%s' %(fig_format)))
    else:
        fig.savefig(os.path.join(figDir,'avg_profiles_dsi.%s' %(fig_format)))

#iso and iso90
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

ylim = [-0.02,1.02]
xlim = [0,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['iso0','iso90'],['red','darkviolet'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#FGM
xlim = [-0.8,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    if use_decon:
        fig.savefig(os.path.join(figDir,'avg_profiles_fgm_deconv.%s' %(fig_format)))
    else:
        fig.savefig(os.path.join(figDir,'avg_profiles_fgm.%s' %(fig_format)))

#iso90 and orth    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [0,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['iso90','orth'],['purple','orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#OTSS
xlim = [-0.8,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['odss'],['tab:green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    if use_decon:
        fig.savefig(os.path.join(figDir,'avg_profiles_otss_deconv.%s' %(fig_format)))
    else:
        fig.savefig(os.path.join(figDir,'avg_profiles_otss.%s' %(fig_format)))

#ctr and sur
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgLocProfiles,['ctr_unwarp','sur_unwarp'],['gold','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#ctr - sur
xlim = [-0.8,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgLocDiffs,['ctr-sur'],['coral'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')


if savefigs:
    if use_decon:
        fig.savefig(os.path.join(figDir,'avg_profiles_ctr_sur_deconv.%s' %(fig_format)))
    else:
        fig.savefig(os.path.join(figDir,'avg_profiles_ctr_sur.%s' %(fig_format)))
    
#%% Make condition depth profiles with individual subject data overlaid

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [0,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['sur'],['gray'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles)

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_sur.%s' %(fig_format)))

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [0,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['iso0'],['red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles)

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_iso0.%s' %(fig_format)))

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [0,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['iso90'],['darkviolet'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles)

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_iso90.%s' %(fig_format)))

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [0,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['orth'],['orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles)

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_orth.%s' %(fig_format)))
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-2.2,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgLocProfiles,['ctr_unwarp'],['gold'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles)

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_ctr.%s' %(fig_format)))
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-2.2,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgLocProfiles,['sur_unwarp'],['purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles)

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_sur1.%s' %(fig_format)))
    
#%% Make difference depth profiles with individual subject data overlaid

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.5,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgLocDiffs,['ctr-sur'],['coral'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_ctr-sur.%s' %(fig_format)))

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.5,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_iso-sur.%s' %(fig_format)))

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.5,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_fgm.%s' %(fig_format)))

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.5,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgTaskDiffs,['odss'],['green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_odss.%s' %(fig_format)))
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.5,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgTaskDiffs,['dsi'],['cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_dsi.%s' %(fig_format)))
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
   
#%% Loc Profiles across surface!
# smoothing according to line 441 in analysisROI_dev
#with smoothing
plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(14, 7))
fontsize = 8
fig1.patch.set_facecolor(fcolor)
locStatDetails = {'labels':['ctr-sur'], 'colors':['black']}


radMax = 4
nRadii = 20
ymax = 6
ymin = -6
highlight = False #['ctr-sur'] #['orth', 'iso90']#'orth'
depth_labels = ['deep', 'middle', 'superficial']
depthBoundaries = np.array([[0,0.3],[0.3,0.6],[0.6,1]])
all_profiles = {depth_label: {label: [] for label in locStatDetails['labels']} for depth_label in depth_labels}

for iR, label in enumerate(all_data.keys()):
    for iD in range(3):
        df = all_data[label][mask_dict[label]]
        df = df[(df['d'] >=depthBoundaries[iD,0]) & (df['d'] <depthBoundaries[iD,1])]
        aw = .8/len(all_data)
        plt.figure(1)
        ax = plt.axes([.07 + iR*1.1*aw, .14 + iD*.3,  aw, .2])
        ax2 = ax.twinx()
        for iStat in range(len(locStatDetails['labels'])):
            alpha = 1.
            if highlight:
                if locStatDetails['labels'][iStat] in highlight:
                    alpha = 1.
                else:
                    alpha = 0.2
            coef = df[locStatDetails['labels'][iStat]].values
            x = df['scale_xy_dist'].values
            #smooth data
            coef_smooth, x_smooth = smoothen(coef, x)
            all_profiles[depth_labels[iD]][locStatDetails['labels'][iStat]].append(coef_smooth)
            # dataDict = makeProfile1D(x, nRadii, coef)
            # plt.plot(dataDict['profile']['depth'],
            #          dataDict['profile']['avg'][0],
            #          '--',
            #          color=statDetails['colors'][iStat])

            #ax.plot(x_smooth, coef_smooth, color=locStatDetails['colors'][iStat], alpha=alpha)
            ax.scatter(x_smooth, coef_smooth, c=coef_smooth, cmap='plasma', alpha=alpha)
            ax.set_ylim([ymin, ymax])
            ax.set_xticks([0, 1, 2, 3])
            if iD == 2:
                ax.set_title(label, fontsize=8, color=lcolor) 
            ax.plot([2, 2], [ymin, ymax], '--', color='gray', alpha=0.15)
            ax.plot([np.min(x_smooth), np.max(x_smooth)], [0, 0], '--', color='gray', alpha=0.15)
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
            ax.set_xticklabels(['0', '', '2', ''], fontsize=8, color=lcolor)
            if iR == 0:
                if iD == 0:
                    ax.set_xlabel("$\sigma$ from ROI center", fontsize = 8, color=lcolor)
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
                ax2.set_xticks([0, 1, 2, 3])
                if iD == 3:
                    ax2.set_title(label, fontsize=8, color=lcolor) 
                ax2.plot([2, 2], [0, ymax], '--', color='gray', alpha=0.15)
                fix_axes(ax2, lcolor=lcolor, fcolor=fcolor)
                if iR > 0:
                    ax2.yaxis.set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    
if savefigs:
    fig1.savefig(os.path.join(figDir,"radial_profiles_loc.%s" %(fig_format)))

#%% Average Loc Porfiles with individual profiles overlaid

fig = plt.figure(figsize=(5, 6))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12
colors = ['black']
plotStats = ['ctr-sur']
plot_indiv = True #if true, overlay individiual traces

for iD, depth_label in enumerate(['deep','middle','superficial']):
    print(depth_label)
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    for iStat, label in enumerate(plotStats):
        nProfiles = len(all_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [-6, 6], '--', color=(.3, .3, .3))
        p.plot([0, 4], [0, 0], '--', color=(.3, .3 ,.3))

        p.plot(x_smooth, stat_avg, color=colors[iStat])
        p.fill_between(x_smooth,
                        stat_avg - stat_std/np.sqrt(nProfiles),
                        stat_avg + stat_std/np.sqrt(nProfiles),
                        linewidth=0.,
                        alpha=0.4,
                        color=colors[iStat])
        
        if plot_indiv:
            p.plot(np.tile(x_smooth,[np.shape(all_profiles[depth_label][label])[0],1]).T, np.array(all_profiles[depth_label][label]).T, color=colors[iStat], alpha=0.2)
        
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-4, 4])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
if savefigs:
    fig.savefig(os.path.join(figDir,"average_indiv_radial_profiles_loc.%s" %(fig_format)))
        

#%% Across surface!
# smoothing according to line 441 in analysisROI_dev
#with smoothing
plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(14, 7))
fontsize = 8
fig1.patch.set_facecolor(fcolor)


radMax = 4
nRadii = 20
ymax = 6
highlight = ['sur', 'iso0'] #['orth', 'iso90']#'orth'
depth_labels = ['deep', 'middle', 'superficial']
depthBoundaries = np.array([[0,0.3],[0.3,0.6],[0.6,1]])
all_profiles = {depth_label: {label: [] for label in statDetails['labels']} for depth_label in depth_labels}

for iR, label in enumerate(all_data.keys()):
    for iD in range(3):
        df = all_data[label][mask_dict[label]]
        df = df[(df['d'] >=depthBoundaries[iD,0]) & (df['d'] <depthBoundaries[iD,1])]
        aw = .8/len(all_data)
        plt.figure(1)
        ax = plt.axes([.07 + iR*1.1*aw, .14 + iD*.3,  aw, .2])
        ax2 = ax.twinx()
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

            ax.plot(x_smooth, coef_smooth, color=statDetails['colors'][iStat], alpha=alpha)
            ax.set_ylim([0, ymax])
            ax.set_xticks([0, 1, 2, 3])
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
            ax.set_xticklabels(['0', '', '2', ''], fontsize=8, color=lcolor)
            if iR == 0:
                if iD == 0:
                    ax.set_xlabel("$\sigma$ from ROI center", fontsize = 8, color=lcolor)
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
                ax2.set_xticks([0, 1, 2, 3])
                if iD == 3:
                    ax2.set_title(label, fontsize=8, color=lcolor) 
                ax2.plot([2, 2], [0, ymax], '--', color='gray', alpha=0.15)
                fix_axes(ax2, lcolor=lcolor, fcolor=fcolor)
                if iR > 0:
                    ax2.yaxis.set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    
if savefigs:
    fig1.savefig(os.path.join(figDir,"radial_profiles_task.%s" %(fig_format)))
                    
#%% Average surface profiles

fig = plt.figure(figsize=(5, 6))
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
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([0, 6])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
if savefigs:
    fig.savefig(os.path.join(figDir,"average_radial_profiles.%s" %(fig_format)))

fig = plt.figure(figsize=(5, 6))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12
    
#Tasks
for iD, depth_label in enumerate(all_profiles.keys()):
    print(depth_label)
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    for iStat, label in enumerate(taskStats):
        nProfiles = len(all_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [0, 6], '--', color=(.3, .3, .3))

        p.plot(x_smooth, stat_avg, color=taskColors[iStat])
        p.fill_between(x_smooth,
                        stat_avg - stat_std/np.sqrt(nProfiles),
                        stat_avg + stat_std/np.sqrt(nProfiles),
                        linewidth=0.,
                        alpha=0.4,
                        color=taskColors[iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, taskStats[iStat],
                color=taskColors[iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([0, 6])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
if savefigs:
    fig.savefig(os.path.join(figDir,"average_radial_task_profiles.%s" %(fig_format)))
    
fig = plt.figure(figsize=(5, 6))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12

#Loc
for iD, depth_label in enumerate(all_profiles.keys()):
    print(depth_label)
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    for iStat, label in enumerate(locStats):
        nProfiles = len(all_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [0, 6], '--', color=(.3, .3, .3))

        p.plot(x_smooth, stat_avg, color=locColors[iStat])
        p.fill_between(x_smooth,
                        stat_avg - stat_std/np.sqrt(nProfiles),
                        stat_avg + stat_std/np.sqrt(nProfiles),
                        linewidth=0.,
                        alpha=0.4,
                        color=locColors[iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, locStats[iStat],
                color=locColors[iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([0, 6])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
if savefigs:
    fig.savefig(os.path.join(figDir,"average_radial_loc_profiles.%s" %(fig_format)))
        
#%% FGM and ODSS Surface Profiles

plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(8.75, 4))
fig1.set_size_inches((8, 4))
fontsize = 8
fig1.patch.set_facecolor(fcolor)

useSI = False #if true, use suppression index ratherthan difference (cond1 - cond2 / cond1 + cond2)
radMax = 4
nRadii = 20
ymax = 4
highlight = ['odss','fgm','dsi','iso-sur'] #['sur', 'iso0'] #['orth', 'iso90']#'orth'
depth_labels = ['deep', 'middle', 'superficial']
diff_labels = ['odss','fgm','dsi','iso-sur']
all_diff_profiles = {depth_label: {label: [] for label in diff_labels} for depth_label in depth_labels}

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    #compute odss, fgm, and dsi
    if useSI:
        df['odss'] = (df['orth'] - df['iso90'])/(df['orth'] + df['iso90'])
        df['fgm'] = (df['iso90'] - df['iso0'])/(df['iso90'] + df['iso0'])
        df['dsi'] = (df['orth'] - df['iso0'])/(df['orth'] + df['iso0'])
        df['iso-sur'] = (df['iso0'] - df['sur'])/(df['iso0'] + df['sur'])
    else:
        df['odss'] = df['orth'] - df['iso90']
        df['fgm'] = df['iso90'] - df['iso0']
        df['dsi'] = df['orth'] - df['iso0']
        df['iso-sur'] = df['iso0'] - df['sur']
    
    all_data[label] = df
    
    for iD in range(3):
        df = all_data[label][mask_dict[label]]
        if iD == 0:
            df = df[(df['d'] >=0) & (df['d'] <0.3)]
        elif iD == 1:
            df = df[(df['d'] >=0.3) & (df['d'] <0.6)]
        elif iD == 2:
            df = df[(df['d'] >=0.6) & (df['d'] <=1)]
        aw = .8/len(all_data)
        plt.figure(1)
        ax = plt.axes([.07 + iR*1.1*aw, .14 + iD*.3,  aw, .2])
        ax2 = ax.twinx()
        for iStat, stat in enumerate(diff_labels):
            alpha = 1.
            if highlight:
                if stat in highlight:
                    alpha = 1.
                else:
                    alpha = 0.2
            coef = df[stat].values
            x = df['scale_xy_dist'].values
            #smooth data
            coef_smooth, x_smooth = smoothen(coef, x)
            if iR < 5:
                all_diff_profiles[depth_labels[iD]][stat].append(coef_smooth)

            ax.plot(x_smooth, coef_smooth, color=diffDetails['colors'][iStat], alpha=alpha)
            ax.set_ylim([-4, ymax])
            ax.set_xticks([0, 1, 2, 3])
            if iD == 2:
                ax.set_title(label, fontsize=8, color=lcolor) 
            ax.plot([2, 2], [0, ymax], '--', color='gray', alpha=0.15)
            fix_axes(ax, lcolor=lcolor, fcolor=fcolor)
            if iR > 0:
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
    
            else:
                plt.ylabel(r'$\beta$, %s' %depth_labels[iD], fontsize = 8, color=lcolor)
                # if iD == 2:
                    # ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
                    # ax.set_yticklabels(['0', '2', '4', '6', '8', '10', '12'], fontsize=8, color=lcolor)
                # else:
                    # ax.set_yticks([0, 2, 4, 6])
                    # ax.set_yticklabels(['0', '2', '4', '6'], fontsize=8, color=lcolor)
            ax.set_xticklabels(['0', '', '2', ''], fontsize=8, color=lcolor)
            if iR == 0:
                if iD == 0:
                    ax.set_xlabel("$\sigma$ from ROI center", fontsize = 8, color=lcolor)
            else:
                ax.set_xticklabels([])
                
if savefigs:
    fig1.savefig(os.path.join(figDir,"radial_difference_profiles.%s" %(fig_format)))
                
#%% Average Surface Difference Profiles

fig = plt.figure(figsize=(5, 6))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12

for iD, depth_label in enumerate(all_diff_profiles.keys()):
    print(depth_label)
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    for iStat, label in enumerate(diff_labels):
        nProfiles = len(all_diff_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_diff_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_diff_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [0, 6], '--', color=(.3, .3, .3))

        p.plot(x_smooth, stat_avg, color=diffDetails['colors'][iStat])
        p.fill_between(x_smooth,
                        stat_avg - stat_std/np.sqrt(nProfiles),
                        stat_avg + stat_std/np.sqrt(nProfiles),
                        linewidth=0.,
                        alpha=0.4,
                        color=diffDetails['colors'][iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, label,
                color=diffDetails['colors'][iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-1, 2])
        p.set_ylabel('$\Delta$ BOLD % change', fontsize=fsize, color=lcolor)
        
if savefigs:
    fig.savefig(os.path.join(figDir,"average_radial_difference_profiles.%s" %(fig_format)))
    
#%% Plot each difference profile separately and show statistics

#first, rerun the radial profiles analysis by binning the data across radial space
nRad = 3 #how many radius bins
maxRad = 4
prop_err = False
showSig = True
binSize = maxRad/nRad
radBins = np.linspace(0,maxRad,nRad+1)
radBinCtrs = radBins[:-1] + binSize/2

# layer masks
layer_mask_dict = {l: {} for l in depth_labels}
for iR, roi_label in enumerate(all_data.keys()):
    for iB, dB in enumerate(depthBoundaries):
        df = all_data[roi_label]
        lmask = (df['d'] > dB[0]) & (df['d'] < dB[1])
        df[depth_labels[iB]] = lmask
        layer_mask_dict[depth_labels[iB]][roi_label] = lmask & mask_dict[roi_label]

# now run binned analysis for each layer
radialProfiles = {}
radialDiffProfiles = {}
avgTaskRadialProfiles = {}
avgTaskRadialDiff = {}
avgLocRadialProfiles = {}
avgLocRadialDiff = {}
for il, l in enumerate(depth_labels):
    radialProfiles[l] = compute_all_rad_profiles(all_data, 
                                         statDetails, 
                                         'bin', 
                                         nRad, 
                                         layer_mask_dict[l],
                                         radParam='scale_xy_dist',
                                         spec_Drange=[0,maxRad],
                                         radMax=maxRad)
    radialDiffProfiles[l] = compute_rad_diff_profiles(all_data,
                                                   statDetails,
                                                   diffDetails['statIDs'],
                                                   'bin',
                                                   nRad,
                                                   prop_err,
                                                   layer_mask_dict[l],
                                                   radParam='scale_xy_dist',
                                                   spec_Drange=[0,maxRad],
                                                   radMax=maxRad)
    avgTaskRadialProfiles[l], avgTaskRadialDiff[l] = compute_avg_rad_profile(radialProfiles[l],statDetails,diffDetails['statIDs'],taskStats,taskDiffs,prop_err,useSI)
    avgLocRadialProfiles[l], avgLocRadialDiff[l] = compute_avg_rad_profile(radialProfiles[l],statDetails,diffDetails['statIDs'],locStats,locDiffs,prop_err,useSI)

#Now plot each differnce profile separately

for iStat, label in enumerate(taskDiffs):
    fig = plt.figure(figsize=(5, 6))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 12
    
    for iD, depth_label in enumerate(all_diff_profiles.keys()):
        print(depth_label)
        p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
        fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
        nProfiles = len(all_diff_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_diff_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_diff_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [-1, 6], '--', color=(.3, .3, .3))
        p.hlines(0,0,maxRad,linestyle='--',color=(0.3,0.3,0.3))

        p.plot(x_smooth, stat_avg, color=diffDetails['colors'][iStat])
        p.fill_between(x_smooth,
                        stat_avg - stat_std/np.sqrt(nProfiles),
                        stat_avg + stat_std/np.sqrt(nProfiles),
                        linewidth=0.,
                        alpha=0.4,
                        color=diffDetails['colors'][iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, label,
                color=diffDetails['colors'][iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-1, 2])
        p.set_ylabel('$\Delta$ BOLD % change', fontsize=fsize, color=lcolor)
        
        if showSig:
            if statCorrType == 'none':
                top = 1.8*np.ones(nRad)
                p.plot(radBinCtrs[avgTaskRadialDiff[depth_label][label]['p-vals'].pvalue <= pthresh],top[avgTaskRadialDiff[depth_label][label]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
            else:
                top = 1.8*np.ones(nRad)
                corrected_pvalues = multipletests(avgTaskRadialDiff[depth_label][label]['p-vals'].pvalue,method=statCorrType)[1]
                p.plot(radBinCtrs[corrected_pvalues <= pthresh],top[corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None')
            
            #also plot bins to show size
            p.vlines(radBins,-1*np.ones([nRad+1,]),2*np.ones([nRad+1,]),linestyle='dotted',alpha=0.3)
            
            #plot average in each bin
            p.errorbar(radBinCtrs,avgTaskRadialDiff[depth_label][label]['avg'],avgTaskRadialDiff[depth_label][label]['stdev']/np.sqrt(avgTaskRadialDiff[depth_label][label]['Nsamp']),marker='o',linestyle='None')
            
        if savefigs:
            fig.savefig(os.path.join(figDir,"%s_average_radial_difference_profiles_wStats.%s" %(label,fig_format)))
            
for iStat, label in enumerate(taskDiffs):
    fig = plt.figure(figsize=(5, 6))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 12
    
    for iD, depth_label in enumerate(all_diff_profiles.keys()):
        print(depth_label)
        p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
        fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
        nProfiles = len(all_diff_profiles[depth_label][label])
        
        p.plot([2, 2], [-1, 6], '--', color=(.3, .3, .3))
        p.hlines(0,0,maxRad,linestyle='--',color=(0.3,0.3,0.3))

        p.plot(radBinCtrs, avgTaskRadialDiff[depth_label][label]['avg'], color=diffDetails['colors'][iStat])
        p.fill_between(radBinCtrs,
                        avgTaskRadialDiff[depth_label][label]['avg'] - avgTaskRadialDiff[depth_label][label]['stdev']/np.sqrt(avgTaskRadialDiff[depth_label][label]['Nsamp']),
                        avgTaskRadialDiff[depth_label][label]['avg'] + avgTaskRadialDiff[depth_label][label]['stdev']/np.sqrt(avgTaskRadialDiff[depth_label][label]['Nsamp']),
                        linewidth=0.,
                        alpha=0.4,
                        color=diffDetails['colors'][iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, label,
                color=diffDetails['colors'][iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-1, 2])
        p.set_ylabel('$\Delta$ BOLD % change', fontsize=fsize, color=lcolor)
        
        if showSig:
            if statCorrType == 'none':
                top = 1.8*np.ones(nRad)
                p.plot(radBinCtrs[avgTaskRadialDiff[depth_label][label]['p-vals'].pvalue <= pthresh],top[avgTaskRadialDiff[depth_label][label]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
            else:
                top = 1.8*np.ones(nRad)
                corrected_pvalues = multipletests(avgTaskRadialDiff[depth_label][label]['p-vals'].pvalue,method=statCorrType)[1]
                p.plot(radBinCtrs[corrected_pvalues <= pthresh],top[corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None')
            
        if savefigs:
            fig.savefig(os.path.join(figDir,"%s_average_radial_difference_profiles_bins.%s" %(label,fig_format)))
            
#%%Now plot each stat profile separately

showSig = True

for iStat, label in enumerate(taskStats):
    fig = plt.figure(figsize=(5, 6))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 12
    
    for iD, depth_label in enumerate(all_profiles.keys()):
        print(depth_label)
        p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
        fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
        nProfiles = len(all_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [-1, 7.5], '--', color=(.3, .3, .3))
        p.hlines(0,0,maxRad,linestyle='--',color=(0.3,0.3,0.3))

        p.plot(x_smooth, stat_avg, color=statDetails['colors'][iStat])
        p.fill_between(x_smooth,
                        stat_avg - stat_std/np.sqrt(nProfiles),
                        stat_avg + stat_std/np.sqrt(nProfiles),
                        linewidth=0.,
                        alpha=0.4,
                        color=statDetails['colors'][iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, label,
                color=statDetails['colors'][iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-1, 7.5])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
        if showSig:
            if statCorrType == 'none':
                top = 6.8*np.ones(nRad)
                p.plot(radBinCtrs[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],top[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
            else:
                top = 6.8*np.ones(nRad)
                corrected_pvalues = multipletests(avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue,method=statCorrType)[1]
                p.plot(radBinCtrs[corrected_pvalues <= pthresh],top[corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None')
            
            #also plot bins to show size
            p.vlines(radBins,-1*np.ones([nRad+1,]),2*np.ones([nRad+1,]),linestyle='dotted',alpha=0.3)
            
            #plot average in each bin
            p.errorbar(radBinCtrs,avgTaskRadialProfiles[depth_label][label]['avg'],avgTaskRadialProfiles[depth_label][label]['stdev']/np.sqrt(avgTaskRadialProfiles[depth_label][label]['Nsamp']),marker='o',linestyle='None')
            
        if savefigs:
            fig.savefig(os.path.join(figDir,"%s_average_radial_profiles_wStats.%s" %(label,fig_format)))

        
for iStat, label in enumerate(taskStats):
    fig = plt.figure(figsize=(5, 6))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 12
    
    for iD, depth_label in enumerate(all_profiles.keys()):
        print(depth_label)
        p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
    
        nProfiles = len(all_profiles[depth_label][label])
        
        p.plot([2, 2], [-1, 7.5], '--', color=(.3, .3, .3))
        p.hlines(0,0,maxRad,linestyle='--',color=(0.3,0.3,0.3))

        p.plot(radBinCtrs, avgTaskRadialProfiles[depth_label][label]['avg'], color=statDetails['colors'][iStat])
        p.fill_between(radBinCtrs,
                        avgTaskRadialProfiles[depth_label][label]['avg'] - avgTaskRadialProfiles[depth_label][label]['stdev']/np.sqrt(avgTaskRadialProfiles[depth_label][label]['Nsamp']),
                        avgTaskRadialProfiles[depth_label][label]['avg'] + avgTaskRadialProfiles[depth_label][label]['stdev']/np.sqrt(avgTaskRadialProfiles[depth_label][label]['Nsamp']),
                        linewidth=0.,
                        alpha=0.4,
                        color=statDetails['colors'][iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, label,
                color=statDetails['colors'][iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-1, 7.5])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
        if showSig:
            if statCorrType == 'none':
                top = 6.8*np.ones(nRad)
                p.plot(radBinCtrs[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],top[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
            else:
                top = 6.8*np.ones(nRad)
                corrected_pvalues = multipletests(avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue,method=statCorrType)[1]
                p.plot(radBinCtrs[corrected_pvalues <= pthresh],top[corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None')
            
        if savefigs:
            fig.savefig(os.path.join(figDir,"%s_average_radial_profiles_bins.%s" %(label,fig_format)))
            
#%% Radial binned all together

#Task
fig = plt.figure(figsize=(5, 6))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 12
showSig = False

for iD, depth_label in enumerate(all_profiles.keys()):    
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    for iStat, label in enumerate(taskStats):
        print(depth_label)
    
        nProfiles = len(all_profiles[depth_label][label])
        
        p.plot([2, 2], [-1, 7.5], '--', color=(.3, .3, .3))
        p.hlines(0,0,maxRad,linestyle='--',color=(0.3,0.3,0.3))

        p.plot(radBinCtrs, avgTaskRadialProfiles[depth_label][label]['avg'], color=taskColors[iStat])
        p.fill_between(radBinCtrs,
                        avgTaskRadialProfiles[depth_label][label]['avg'] - avgTaskRadialProfiles[depth_label][label]['stdev']/np.sqrt(avgTaskRadialProfiles[depth_label][label]['Nsamp']),
                        avgTaskRadialProfiles[depth_label][label]['avg'] + avgTaskRadialProfiles[depth_label][label]['stdev']/np.sqrt(avgTaskRadialProfiles[depth_label][label]['Nsamp']),
                        linewidth=0.,
                        alpha=0.4,
                        color=taskColors[iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, label,
                color=taskColors[iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-1, 7.5])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
        if showSig:
            if statCorrType == 'none':
                top = 6.8*np.ones(nRad)
                p.plot(radBinCtrs[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],top[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
            else:
                top = 6.8*np.ones(nRad)
                corrected_pvalues = multipletests(avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue,method=statCorrType)[1]
                p.plot(radBinCtrs[corrected_pvalues <= pthresh],top[corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None')
            
if savefigs:
    fig.savefig(os.path.join(figDir,"average_radial_task_profiles_bins.%s" %(fig_format)))

#Iso-Sur
fig = plt.figure(figsize=(5, 6))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 12
showSig = False
iso_surStats = ['iso0','sur']
iso_surColors = ['red','gray']

for iD, depth_label in enumerate(all_profiles.keys()):    
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    for iStat, label in enumerate(iso_surStats):
        print(depth_label)
    
        nProfiles = len(all_profiles[depth_label][label])
        
        p.plot([2, 2], [-1, 7.5], '--', color=(.3, .3, .3))
        p.hlines(0,0,maxRad,linestyle='--',color=(0.3,0.3,0.3))

        p.plot(radBinCtrs, avgTaskRadialProfiles[depth_label][label]['avg'], color=iso_surColors[iStat])
        p.fill_between(radBinCtrs,
                        avgTaskRadialProfiles[depth_label][label]['avg'] - avgTaskRadialProfiles[depth_label][label]['stdev']/np.sqrt(avgTaskRadialProfiles[depth_label][label]['Nsamp']),
                        avgTaskRadialProfiles[depth_label][label]['avg'] + avgTaskRadialProfiles[depth_label][label]['stdev']/np.sqrt(avgTaskRadialProfiles[depth_label][label]['Nsamp']),
                        linewidth=0.,
                        alpha=0.4,
                        color=iso_surColors[iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, label,
                color=iso_surColors[iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-1, 7.5])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
        if showSig:
            if statCorrType == 'none':
                top = 6.8*np.ones(nRad)
                p.plot(radBinCtrs[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],top[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
            else:
                top = 6.8*np.ones(nRad)
                corrected_pvalues = multipletests(avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue,method=statCorrType)[1]
                p.plot(radBinCtrs[corrected_pvalues <= pthresh],top[corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None')
            
if savefigs:
    fig.savefig(os.path.join(figDir,"average_radial_isosur_profiles_bins.%s" %(fig_format)))
    
#conMod
fig = plt.figure(figsize=(5, 6))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 12
showSig = False
conModStats = ['iso0','iso90','orth']
conModColors = ['red','darkviolet','orange']

for iD, depth_label in enumerate(all_profiles.keys()):    
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    for iStat, label in enumerate(conModStats):
        print(depth_label)
    
        nProfiles = len(all_profiles[depth_label][label])
        
        p.plot([2, 2], [-1, 7.5], '--', color=(.3, .3, .3))
        p.hlines(0,0,maxRad,linestyle='--',color=(0.3,0.3,0.3))

        p.plot(radBinCtrs, avgTaskRadialProfiles[depth_label][label]['avg'], color=conModColors[iStat])
        p.fill_between(radBinCtrs,
                        avgTaskRadialProfiles[depth_label][label]['avg'] - avgTaskRadialProfiles[depth_label][label]['stdev']/np.sqrt(avgTaskRadialProfiles[depth_label][label]['Nsamp']),
                        avgTaskRadialProfiles[depth_label][label]['avg'] + avgTaskRadialProfiles[depth_label][label]['stdev']/np.sqrt(avgTaskRadialProfiles[depth_label][label]['Nsamp']),
                        linewidth=0.,
                        alpha=0.4,
                        color=conModColors[iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, label,
                color=conModColors[iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-1, 7.5])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
        if showSig:
            if statCorrType == 'none':
                top = 6.8*np.ones(nRad)
                p.plot(radBinCtrs[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],top[avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
            else:
                top = 6.8*np.ones(nRad)
                corrected_pvalues = multipletests(avgTaskRadialProfiles[depth_label][label]['p-vals'].pvalue,method=statCorrType)[1]
                p.plot(radBinCtrs[corrected_pvalues <= pthresh],top[corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None')
            
if savefigs:
    fig.savefig(os.path.join(figDir,"average_radial_conMod_profiles_bins.%s" %(fig_format)))
    
#%% Analyze variance across voxels

# Find the mean normalized variance
dsi_mnv = np.array(diffProfiles['dsi']['stdev'])**2/np.abs(diffProfiles['dsi']['avg']) #mean-normalized variance across voxels in each ROI
fgm_mnv = np.array(diffProfiles['fgm']['stdev'])**2/np.abs(diffProfiles['fgm']['avg']) #^
odss_mnv = np.array(diffProfiles['odss']['stdev'])**2/np.abs(diffProfiles['odss']['avg']) #^

# Now aggregate across ROIs and plot
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-200,500]

plt.errorbar(np.mean(dsi_mnv,axis=0),np.linspace(0,1,nDepths),xerr=np.std(dsi_mnv,axis=0),color='cyan') #dsi
plt.plot(dsi_mnv.T,np.tile(np.linspace(0,1,nDepths),(np.shape(dsi_mnv)[0],1)).T,'.',color='cyan') #individual dsi data
plt.errorbar(np.mean(fgm_mnv,axis=0),np.linspace(0,1,nDepths),xerr=np.std(fgm_mnv,axis=0),color='magenta') #fgm
plt.plot(fgm_mnv.T,np.tile(np.linspace(0,1,nDepths),(np.shape(fgm_mnv)[0],1)).T,'.',color='magenta') #individual fgm data
plt.errorbar(np.mean(odss_mnv,axis=0),np.linspace(0,1,nDepths),xerr=np.std(odss_mnv,axis=0),color='green') #odss
plt.plot(odss_mnv.T,np.tile(np.linspace(0,1,nDepths),(np.shape(odss_mnv)[0],1)).T,'.',color='green') #individual odss data

plt.ylabel(r"Relative Depth (WM $\rightarrow$ Pia)")
plt.xlabel("Mean-Normalized Variance")
plt.title("Contrast-condition variance across voxels (mean-normalized)", fontsize=8)

# Let's also plot the variances of the contrasts
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-200,500]

plt.errorbar(np.mean(np.array(diffProfiles['dsi']['stdev'])**2,axis=0),np.linspace(0,1,nDepths),xerr=np.std(np.array(diffProfiles['dsi']['stdev'])**2,axis=0),color='cyan') #dsi
plt.plot(np.array(diffProfiles['dsi']['stdev']).T,np.tile(np.linspace(0,1,nDepths),(np.shape(diffProfiles['dsi']['stdev'])[0],1)).T,'.',color='cyan') #individual dsi data
plt.errorbar(np.mean(np.array(diffProfiles['fgm']['stdev'])**2,axis=0),np.linspace(0,1,nDepths),xerr=np.std(np.array(diffProfiles['fgm']['stdev'])**2,axis=0),color='magenta') #fgm
plt.plot(np.array(diffProfiles['fgm']['stdev']).T,np.tile(np.linspace(0,1,nDepths),(np.shape(diffProfiles['fgm']['stdev'])[0],1)).T,'.',color='magenta') #individual fgm data
plt.errorbar(np.mean(np.array(diffProfiles['odss']['stdev'])**2,axis=0),np.linspace(0,1,nDepths),xerr=np.std(np.array(diffProfiles['odss']['stdev'])**2,axis=0),color='green') #odss
plt.plot(np.array(diffProfiles['odss']['stdev']).T,np.tile(np.linspace(0,1,nDepths),(np.shape(diffProfiles['odss']['stdev'])[0],1)).T,'.',color='green') #individual odss data

plt.ylabel(r"Relative Depth (WM $\rightarrow$ Pia)")
plt.xlabel("Variance")
plt.title("Contrast-condition variance across voxels", fontsize=8)

# Let's also plot the means of the contrasts
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-200,500]

plt.errorbar(np.mean(np.array(diffProfiles['dsi']['avg']),axis=0),np.linspace(0,1,nDepths),xerr=np.std(np.array(diffProfiles['dsi']['avg']),axis=0),color='cyan') #dsi
plt.plot(np.array(diffProfiles['dsi']['avg']).T,np.tile(np.linspace(0,1,nDepths),(np.shape(diffProfiles['dsi']['avg'])[0],1)).T,'.',color='cyan') #individual dsi data
plt.errorbar(np.mean(np.array(diffProfiles['fgm']['avg']),axis=0),np.linspace(0,1,nDepths),xerr=np.std(np.array(diffProfiles['fgm']['avg']),axis=0),color='magenta') #fgm
plt.plot(np.array(diffProfiles['fgm']['avg']).T,np.tile(np.linspace(0,1,nDepths),(np.shape(diffProfiles['fgm']['avg'])[0],1)).T,'.',color='magenta') #individual fgm data
plt.errorbar(np.mean(np.array(diffProfiles['odss']['avg']),axis=0),np.linspace(0,1,nDepths),xerr=np.std(np.array(diffProfiles['odss']['avg']),axis=0),color='green') #odss
plt.plot(np.array(diffProfiles['odss']['avg']).T,np.tile(np.linspace(0,1,nDepths),(np.shape(diffProfiles['odss']['avg'])[0],1)).T,'.',color='green') #individual odss data

plt.ylabel(r"Relative Depth (WM $\rightarrow$ Pia)")
plt.xlabel("Mean")
plt.title("Contrast-condition mean across voxels", fontsize=8)

# Let's also plot the N of the contrasts
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-200,500]

plt.errorbar(np.mean(np.array(diffProfiles['dsi']['N']),axis=0),np.linspace(0,1,nDepths),xerr=np.std(np.array(diffProfiles['dsi']['N']),axis=0),color='black') #dsi
plt.plot(np.array(diffProfiles['dsi']['N']).T,np.tile(np.linspace(0,1,nDepths),(np.shape(diffProfiles['dsi']['N'])[0],1)).T,'.',color='black') #dsi

plt.ylabel(r"Relative Depth (WM $\rightarrow$ Pia)")
plt.xlabel("# of Voxels")
plt.title("N voxels", fontsize=8)

#%% Analyze variance in individual subjects

#Contrast Variance
dsi_var = np.array(diffProfiles['dsi']['stdev'])**2
fgm_var = np.array(diffProfiles['fgm']['stdev'])**2
odss_var = np.array(diffProfiles['odss']['stdev'])**2
roi_names = list(all_data.keys())
fig, ax = plt.subplots(2,int(np.ceil(NROIs/2)),figsize = (10,6))
for r_i in range(NROIs):
    col = int(r_i % np.ceil(NROIs/2))
    row = int(np.floor(r_i/np.ceil(NROIs/2)))
    ax[row,col].plot(dsi_var[r_i],np.linspace(0,1,nDepths),color='cyan',label='dsi')
    ax[row,col].plot(fgm_var[r_i],np.linspace(0,1,nDepths),color='magenta',label='fgm')
    ax[row,col].plot(odss_var[r_i],np.linspace(0,1,nDepths),color='green',label='odss')
    ax[row,col].set_title(roi_names[r_i],fontsize=6)
    ax[row,col].legend(fontsize=6)
    if col == 0:
        ax[row,col].set_ylabel(r"Relative Depth (WM $\rightarrow$ Pia)")
    if row == 1:
        ax[row,col].set_xlabel("Var Across Voxels")
        
#Contrast MNV
roi_names = list(all_data.keys())
fig, ax = plt.subplots(2,int(np.ceil(NROIs/2)),figsize = (10,6))
for r_i in range(NROIs):
    col = int(r_i % np.ceil(NROIs/2))
    row = int(np.floor(r_i/np.ceil(NROIs/2)))
    ax[row,col].plot(dsi_mnv[r_i],np.linspace(0,1,nDepths),color='cyan',label='dsi')
    ax[row,col].plot(fgm_mnv[r_i],np.linspace(0,1,nDepths),color='magenta',label='fgm')
    ax[row,col].plot(odss_mnv[r_i],np.linspace(0,1,nDepths),color='green',label='odss')
    ax[row,col].set_title(roi_names[r_i],fontsize=6)
    ax[row,col].legend(fontsize=6)
    if col == 0:
        ax[row,col].set_ylabel(r"Relative Depth (WM $\rightarrow$ Pia)")
    if row == 1:
        ax[row,col].set_xlabel("MNV Across Voxels")
        
#Condition Variance
iso_var = np.array(depthProfiles['iso0']['stdev'])**2
iso90_var = np.array(depthProfiles['iso90']['stdev'])**2
orth_var = np.array(depthProfiles['orth']['stdev'])**2
sur_var = np.array(depthProfiles['sur']['stdev'])**2
roi_names = list(all_data.keys())
fig, ax = plt.subplots(2,int(np.ceil(NROIs/2)),figsize = (10,6))
for r_i in range(NROIs):
    col = int(r_i % np.ceil(NROIs/2))
    row = int(np.floor(r_i/np.ceil(NROIs/2)))
    ax[row,col].plot(iso_var[r_i],np.linspace(0,1,nDepths),color='red',label='iso')
    ax[row,col].plot(iso90_var[r_i],np.linspace(0,1,nDepths),color='purple',label='iso90')
    ax[row,col].plot(orth_var[r_i],np.linspace(0,1,nDepths),color='orange',label='orth')
    ax[row,col].plot(sur_var[r_i],np.linspace(0,1,nDepths),color='gray',label='sur')
    ax[row,col].set_title(roi_names[r_i],fontsize=6)
    ax[row,col].legend(fontsize=6)
    if col == 0:
        ax[row,col].set_ylabel(r"Relative Depth (WM $\rightarrow$ Pia)")
    if row == 1:
        ax[row,col].set_xlabel("Var Across Voxels")
        
#Condition MNV
iso_mnv = np.array(depthProfiles['iso0']['stdev'])**2/np.abs(depthProfiles['iso0']['avg'])
iso90_mnv = np.array(depthProfiles['iso90']['stdev'])**2/np.abs(depthProfiles['iso90']['avg'])
orth_mnv = np.array(depthProfiles['orth']['stdev'])**2/np.abs(depthProfiles['orth']['avg'])
sur_mnv = np.array(depthProfiles['sur']['stdev'])**2/np.abs(depthProfiles['sur']['avg'])
roi_names = list(all_data.keys())
fig, ax = plt.subplots(2,int(np.ceil(NROIs/2)),figsize = (10,6))
for r_i in range(NROIs):
    col = int(r_i % np.ceil(NROIs/2))
    row = int(np.floor(r_i/np.ceil(NROIs/2)))
    ax[row,col].plot(iso_mnv[r_i],np.linspace(0,1,nDepths),color='red',label='iso')
    ax[row,col].plot(iso90_mnv[r_i],np.linspace(0,1,nDepths),color='purple',label='iso90')
    ax[row,col].plot(orth_mnv[r_i],np.linspace(0,1,nDepths),color='orange',label='orth')
    ax[row,col].plot(sur_mnv[r_i],np.linspace(0,1,nDepths),color='gray',label='sur')
    ax[row,col].set_title(roi_names[r_i],fontsize=6)
    ax[row,col].legend(fontsize=6)
    if col == 0:
        ax[row,col].set_ylabel(r"Relative Depth (WM $\rightarrow$ Pia)")
    if row == 1:
        ax[row,col].set_xlabel("MNV Across Voxels")
        
#%% Analyze Variance in entire ROI
roiCond = compute_all_depth_profiles(all_data,statDetails,profile_method,1,masks,depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
roiDiff = compute_diff_profiles(all_data,statDetails,diffDetails['statIDs'],profile_method,1,useSI,masks,depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')

# Contrast Variance
dsi_roi_var = np.array(roiDiff['dsi']['stdev'])**2
fgm_roi_var = np.array(roiDiff['fgm']['stdev'])**2
odss_roi_var = np.array(roiDiff['odss']['stdev'])**2
diff_roi_var = np.concatenate((odss_roi_var,fgm_roi_var,dsi_roi_var),axis=1).T
roi_names = list(all_data.keys())
fig, ax = plt.subplots()
contrasts = ['odss','fgm','dsi']
ax.plot(np.arange(len(contrasts)),diff_roi_var,'k')
ax.set_xticks(np.arange(len(contrasts)))
ax.set_xticklabels(contrasts)
for c_i, con in enumerate(contrasts):
    ax.plot(np.ones(NROIs)*c_i,diff_roi_var[c_i,:],'o',color=diffDetails['colors'][c_i])
ax.set_title("ROI Condition Contrast Variance")
ax.set_ylabel(r"Variance ([$\Delta$ % BOLD Change]$^{2}$)")

# Contrast MNV
dsi_roi_mnv = np.array(roiDiff['dsi']['stdev'])**2/np.abs(np.array(roiDiff['dsi']['avg']))
fgm_roi_mnv = np.array(roiDiff['fgm']['stdev'])**2/np.abs(np.array(roiDiff['fgm']['avg']))
odss_roi_mnv = np.array(roiDiff['odss']['stdev'])**2/np.abs(np.array(roiDiff['odss']['avg']))
diff_roi_mnv = np.concatenate((odss_roi_mnv,fgm_roi_mnv,dsi_roi_mnv),axis=1).T
roi_names = list(all_data.keys())
fig, ax = plt.subplots()
ax.plot(np.arange(len(contrasts)),diff_roi_mnv,'k')
ax.set_xticks(np.arange(len(contrasts)))
ax.set_xticklabels(contrasts)
for c_i, con in enumerate(contrasts):
    ax.plot(np.ones(NROIs)*c_i,diff_roi_mnv[c_i,:],'o',color=diffDetails['colors'][c_i])
ax.set_title("ROI Condition Contrast MNV")
ax.set_ylabel(r"MNV (Var/Mean)")

# Condition Variance
iso_roi_var = np.array(roiCond['iso0']['stdev'])**2
iso90_roi_var = np.array(roiCond['iso90']['stdev'])**2
orth_roi_var = np.array(roiCond['orth']['stdev'])**2
sur_roi_var = np.array(roiCond['sur']['stdev'])**2
cond_roi_var = np.concatenate((sur_roi_var[:,np.newaxis],iso_roi_var[:,np.newaxis],iso90_roi_var[:,np.newaxis],orth_roi_var[:,np.newaxis]),axis=1).T
roi_names = list(all_data.keys())
fig, ax = plt.subplots()
conditions = ['sur','iso0','iso90','orth']
ax.plot(np.arange(len(conditions)),cond_roi_var,'k')
ax.set_xticks(np.arange(len(conditions)))
ax.set_xticklabels(conditions)
for c_i, con in enumerate(conditions):
    ax.plot(np.ones(NROIs)*c_i,cond_roi_var[c_i,:],'o',color=statDetails['colors'][c_i])
ax.set_title("ROI Condition Variance")
ax.set_ylabel(r"Variance ([% BOLD Change]$^{2}$)")

# Condition MNV
iso_roi_mnv = np.array(roiCond['iso0']['stdev'])**2/np.array(roiCond['iso0']['avg'])
iso90_roi_mnv = np.array(roiCond['iso90']['stdev'])**2/np.array(roiCond['iso90']['avg'])
orth_roi_mnv = np.array(roiCond['orth']['stdev'])**2/np.array(roiCond['orth']['avg'])
sur_roi_mnv = np.array(roiCond['sur']['stdev'])**2/np.array(roiCond['sur']['avg'])
cond_roi_mnv = np.concatenate((sur_roi_mnv[:,np.newaxis],iso_roi_mnv[:,np.newaxis],iso90_roi_mnv[:,np.newaxis],orth_roi_mnv[:,np.newaxis]),axis=1).T
roi_names = list(all_data.keys())
fig, ax = plt.subplots()
ax.plot(np.arange(len(conditions)),cond_roi_mnv,'k')
ax.set_xticks(np.arange(len(conditions)))
ax.set_xticklabels(conditions)
for c_i, con in enumerate(conditions):
    ax.plot(np.ones(NROIs)*c_i,cond_roi_mnv[c_i,:],'o',color=statDetails['colors'][c_i])
ax.set_title("ROI Condition MNV")
ax.set_ylabel(r"Variance (Var/Mean)")
