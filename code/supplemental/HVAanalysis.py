#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:10:04 2023

@author: joe
"""

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
statCorrType = 'bonferroni'

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code'
#datasets = glob.glob(os.path.join(mainDir, 'roi_data', 'pnr???_??_??_?????.csv'))
datasets = [#'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_lh_target_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V1_lh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_lh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V1_lh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_lh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_rh_target_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr756_V1_lh_target_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr756_V1_rh_target_laynii.csv'
  ]
datasets.sort()

roiRad = 2 #1.
import pandas as pd
all_data = {}
for dataset in datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    all_data[f] = pd.read_csv(dataset, sep=',', index_col=False)

hva_datasets = [#'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_lh_target_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V23_lh_tgt_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V23_rh_tgt_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V23_lh_tgt_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V23_rh_tgt_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V23_lh_tgt_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V23_rh_tgt_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V23_lh_tgt_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V23_rh_tgt_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr756_V1_lh_target_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr756_V1_rh_target_laynii.csv'
  ]
hva_datasets.sort()

hva_data = {}
for dataset in hva_datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    hva_data[f] = pd.read_csv(dataset, sep=',', index_col=False)
    
# make a dummy variable called scale_uv_dist for the hva ROIs, we may not end 
# up using this, but it will make sure that the dataframes play nice with
# existing code
for iR, label in enumerate(hva_data.keys()):
    hva_data[label]['scale_uv_dist'] = np.sqrt(hva_data[label]['u'].values**2 + hva_data[label]['v'].values**2)
    
    #remove zero depths (in the future, I should not have zero depth voxels)
    zeroD = (hva_data[label]['d'] == 0)
    zeroLabels = hva_data[label].index[zeroD]
    hva_data[label] = hva_data[label].drop(labels=zeroLabels,axis=0)

#%% Histograms of p-values
# floc = plt.figure()
# for iR, label in enumerate(all_data.keys()):
#     df = all_data[label]
    
#     roi = df[df['scale_uv_dist'] < roiRad]
#     if 'loc pval' in roi.keys():
#         roi = roi.rename(columns={'loc pval':'loc p-val'})
#     plt.subplot(np.ceil(len(datasets)/2),2,iR+1)
#     plt.hist(roi['loc p-val'].values,bins=20,density=True)
#     plt.title(label+" loc p-val",fontsize=8)
#     plt.xlabel("pval")
#     plt.text(0.8,5,'< 0.05 = %d %%' %(100*np.sum(roi['loc p-val'] <= 0.05)/len(roi['loc p-val'])))
#     plt.ylim([0,10])
#     plt.xlim([0,1])
# floc.tight_layout(pad=0.1)

# floc_hva = plt.figure()
# for iR, label in enumerate(hva_data.keys()):
#     df = hva_data[label]
    
#     roi = df[df['scale_uv_dist'] < roiRad]
#     if 'loc pval' in roi.keys():
#         roi = roi.rename(columns={'loc pval':'loc p-val'})
#     plt.subplot(np.ceil(len(datasets)/2),2,iR+1)
#     plt.hist(roi['loc p-val'].values,bins=20,density=True)
#     plt.title(label+" loc p-val",fontsize=8)
#     plt.xlabel("pval")
#     plt.text(0.8,5,'< 0.05 = %d %%' %(100*np.sum(roi['loc p-val'] <= 0.05)/len(roi['loc p-val'])))
#     plt.ylim([0,10])
#     plt.xlim([0,1])
# floc_hva.tight_layout(pad=0.1)

ftask = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_uv_dist'] < roiRad]
    if 'task pval' in roi.keys():
        roi = roi.rename(columns={'task pval':'task p-val'})
    plt.subplot(np.ceil(len(datasets)/2),2,iR+1)
    plt.hist(roi['task p-val'].values,bins=20,density=True)
    plt.title(label+" task p-val",fontsize=8)
    plt.xlabel("pval")
    plt.text(0.8,5,'< 0.05 = %d %%' %(100*np.sum(roi['task p-val'] <= 0.05)/len(roi['task p-val'])))
    plt.ylim([0,10])
    plt.xlim([0,1])
ftask.tight_layout(pad=0.1)

ftask_hva = plt.figure()
for iR, label in enumerate(hva_data.keys()):
    df = hva_data[label]
    
    roi = df[df['scale_uv_dist'] < roiRad]
    if 'task pval' in roi.keys():
        roi = roi.rename(columns={'task pval':'task p-val'})
    plt.subplot(np.ceil(len(datasets)/2),2,iR+1)
    plt.hist(roi['task p-val'].values,bins=20,density=True)
    plt.title(label+" task p-val",fontsize=8)
    plt.xlabel("pval")
    plt.text(0.8,5,'< 0.05 = %d %%' %(100*np.sum(roi['task p-val'] <= 0.05)/len(roi['task p-val'])))
    plt.ylim([0,10])
    plt.xlim([0,1])
ftask_hva.tight_layout(pad=0.1)

if savefigs:
    # floc.savefig(os.path.join(figDir,'pvals_loc.svg'))
    # floc_hva.savefig(os.path.join(figDir,'pvals_loc_hva.svg'))
    ftask.savefig(os.path.join(figDir,'pvals_task.svg'))
    ftask_hva.savefig(os.path.join(figDir,'pvals_task_hva.svg'))

#%% Depth Histograms
# I want to see how much coverage we are getting through depth.
roiRad = 2
nDepths = 6
fdhist = plt.figure(figsize=(15,4))
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_uv_dist'] < roiRad]
    roi = df[df['scale_uv_dist'] < roiRad]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,np.ceil(len(datasets)/2),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.title(label)
    plt.xlabel("Normalize Depth WM -> GM")
    plt.ylabel("Num. Voxels")
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
fdhist.tight_layout(pad=0.0)

if savefigs:
    fdhist.savefig(os.path.join(figDir,'depth_hist.svg'))
    
roiRad = 2
fdhist_hva = plt.figure(figsize=(15,4))
for iR, label in enumerate(hva_data.keys()):
    df = hva_data[label]
    
    roi = df[df['scale_uv_dist'] < roiRad]
    roi = df[df['scale_uv_dist'] < roiRad]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,np.ceil(len(datasets)/2),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.title(label)
    plt.xlabel("Normalize Depth WM -> GM")
    plt.ylabel("Num. Voxels")
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
fdhist.tight_layout(pad=0.0)

if savefigs:
    fdhist_hva.savefig(os.path.join(figDir,'depth_hist_hva.svg'))
    
#%% Use the deepest layer as a proxy for non-vein contaminated voxels
# Then define the threshold based on this distribution

deep_pct = 10 #percentile to call deep layers
conditions = ['iso0','iso90','orth','sur']
depth_groups = {'deep': [0.2,0.4], 'middle': [0.4,0.6], 'superficial': [0.6,0.8]}
depth_labels = ['superficial','middle','deep'] #put them in the right order
depth_var = 'd'
x_var = 'u'
y_var = 'v'
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
    lmnv = get_lmnv(df,key='task stdev_xerrts') #log of the mean-normalized variance
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
    fthresh.savefig(os.path.join(figDir,'mnv_hist.svg'))
    dmap.savefig(os.path.join(figDir,'mnv_depth_map.svg'))
    dmap_thresh.savefig(os.path.join(figDir,'mnv_depth_map_thresh.svg'))
    fdepth_hist.savefig(os.path.join(figDir,'mnv_depth_hist.svg'))

#%% Now for HVAs
for k_i, key in enumerate(hva_data.keys()):

    # calculate log(MNV)
    df = hva_data[key]
    lmnv = get_lmnv(df,key='task stdev_xerrts') #log of the mean-normalized variance
    mnv = np.exp(lmnv) #get back mean normalized variance
    
    # get deep layer distribution
    z = df[depth_var]
    [deep_mean, deep_std, deep] = get_deep_layer_dist(df,depth_var,deep_pct)
    lmnv_dict[key] = {'mean':0,'std':0,'thresh':0,'deep_mean':0,'deep_std':0}
    lmnv_dict[key]['deep_mean'] = deep_mean
    lmnv_dict[key]['deep_std'] = deep_std
    
    # define threshold based on deep layer distribution
    [mnv_mask, lmnv_thresh] = get_mnv_mask(df,depth_var,deep_pct,sd_thresh)
    lmnv_dict[key]['thresh'] = lmnv_thresh
    lmnv_dict[key]['mean'] = np.mean(lmnv)
    lmnv_dict[key]['std'] = np.std(lmnv)
    mask_dict[key] = mnv_mask
    hva_data[key]['no_vein'] = mnv_mask
    
    # Plot distributions
    fthresh_hva = plot_mnv_histograms(lmnv, lmnv[deep], mnv_mask, deep_pct, key, k_i, NROIs, fsize, pad=0.0, figsize=(15,3))
    
    # Plot depth maps
    dmap_hva = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0,100], fsize, fname = 'dmap', pad=0.0)
        
    #plot thresholded map
    dmap_thresh_hva = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0, 100], fsize, fname='dmap_thresh', mask=mnv_mask, pad=0.0)
        
    # Plot voxel loss at each depth after masking
    #fdepth_hist_hva = plot_depth_voxel_loss(z, mnv_mask, nDepths, NROIs, key, k_i, fsize)
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(mnv_mask),np.size(mnv),key))
    
if savefigs:
    fthresh_hva.savefig(os.path.join(figDir,'mnv_hist_hva.svg'))
    dmap_hva.savefig(os.path.join(figDir,'mnv_depth_map_hva.svg'))
    dmap_thresh_hva.savefig(os.path.join(figDir,'mnv_depth_map_thresh_hva.svg'))
    fdepth_hist_hva.savefig(os.path.join(figDir,'mnv_depth_hist_hva.svg'))

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
    f.savefig(os.path.join(figDir,'mnv_summary_errorbars.svg'))

#try violin plots
spreadF = 2
f = plt.figure()
for k_i, key in enumerate(all_data.keys()):
    df = all_data[key]
    lmnv = get_lmnv(df,key='task stdev_xerrts') #log of the mean-normalized variance
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
    plt.plot(spreadF*(k_i+0.4),lmnv_dict[key]['thresh'],color='r',marker='s')
plt.xticks(np.arange(0,spreadF*len(all_data.keys()),spreadF),all_data.keys(),rotation=15,fontsize=6)
plt.ylabel("log(MNV)")

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin.svg'))
    
#try violin plots
spreadF = 2
f = plt.figure()
for k_i, key in enumerate(hva_data.keys()):
    df = hva_data[key]
    lmnv = get_lmnv(df,key='task stdev_xerrts') #log of the mean-normalized variance
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
    plt.plot(spreadF*(k_i+0.4),lmnv_dict[key]['thresh'],color='r',marker='s')
plt.xticks(np.arange(0,spreadF*len(all_data.keys()),spreadF),all_data.keys(),rotation=15,fontsize=6)
plt.ylabel("log(MNV)")

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin_hva.svg'))
    
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
if use_fullmodel_mask:
    for k_i, key in enumerate(hva_data.keys()):
        df = hva_data[key]
        pvals = df['task p-val']
        pval_mask = pvals < pthresh_fullmodel
        print("%d/%d voxels survive full model p-val mask" %(np.sum(pval_mask),np.size(pval_mask)))
        mask_dict[key] = mask_dict[key] & pval_mask   
        hva_data[key]['sig'] = pval_mask
        
#%% gPPI Analysis

gPPIDetails_V1tgt = {'labels': ['V23_lh_tgt', 'V23_rh_tgt', 
                          'V23_lh_tgt_iso0', 'V23_rh_tgt_iso0',
                          'V23_lh_tgt_iso90', 'V23_rh_tgt_iso90',
                          'V23_lh_tgt_orth', 'V23_rh_tgt_orth',
                          'V23_lh_tgt_sur', 'V23_rh_tgt_sur'],
                'colors': ['black', 'black',
                           'black', 'black',
                           'black', 'black',
                           'black', 'black',
                           'black', 'black']}

gPPIDetails_V23tgt = {'labels': ['V1_lh_tgt', 'V1_rh_tgt', 
                          'V1_lh_tgt_iso0', 'V1_rh_tgt_iso0',
                          'V1_lh_tgt_iso90', 'V1_rh_tgt_iso90',
                          'V1_lh_tgt_orth', 'V1_rh_tgt_orth',
                          'V1_lh_tgt_sur', 'V1_rh_tgt_sur'],
                'colors': ['gray', 'gray',
                           'gray', 'gray',
                           'gray', 'gray',
                           'gray', 'gray',
                           'gray', 'gray']}

profile_method = 'bin' # bin or smooth
roiRad = 2
#pick out ROIs where we're sure of localization
roi_dict = {}
for key in all_data.keys():
    df = all_data[key]
    roi_dict[key] = df['scale_uv_dist'] < roiRad
    all_data[key]['in_tgt'] = df['scale_uv_dist'] < roiRad
for key in hva_data.keys():
    df = hva_data[key]
    roi_dict[key] = df['scale_uv_dist'] < roiRad
    hva_data[key]['in_tgt'] = df['scale_uv_dist'] < roiRad
useSI = False #use suppression index rather than differences (cond1 - cond2 / cond1 + cond2)
#create full masks
masks_V1 = {roi:all_data[roi]['in_tgt']*all_data[roi]['sig']*all_data[roi]['no_vein'] for roi in all_data.keys()}
depthProfiles_V1tgt = compute_all_depth_profiles(all_data,gPPIDetails_V1tgt,profile_method,nDepths,masks_V1,depthParam='d',radialParam='scale_uv_dist',spec_Drange='MinMax')
masks_hva = {roi:hva_data[roi]['in_tgt']*hva_data[roi]['sig']*hva_data[roi]['no_vein'] for roi in hva_data.keys()}
depthProfiles_V23tgt = compute_all_depth_profiles(hva_data,gPPIDetails_V23tgt,profile_method,nDepths,masks_hva,depthParam='d',radialParam='scale_uv_dist',spec_Drange='MinMax')

#%% gPPI Centroid Plots
# Let's take a look at raw voxel betas across depth by condition

roiRad = 2
Nsubj = len(all_data.keys())

#plot centroids for each condition and ROI
plot_centroids(all_data, masks_V1, gPPIDetails_V1tgt, roiRad, nDepths)
        
if savefigs:
    for l in gPPIDetails_V1tgt['labels']:
        plt.figure(l)
        plt.savefig(os.path.join(figDir,'centroids_%s.svg' %l))

Nsubj = len(hva_data.keys())

#plot centroids for each condition and ROI
plot_centroids(hva_data, masks_hva, gPPIDetails_V23tgt, roiRad, nDepths)
        
if savefigs:
    for l in gPPIDetails_V23tgt['labels']:
        plt.figure(l)
        plt.savefig(os.path.join(figDir,'centroids_%s.svg' %l))    

#%% Deconvolution

#reformat data to fit decon_rois specs
keep_rois = np.zeros((NROIs,len(gPPIDetails_V1tgt['labels']),nDepths))
for iR, roiID in enumerate(all_data.keys()):
    for iStat, stat in enumerate(gPPIDetails_V1tgt['labels']):
        keep_rois[iR,iStat,:] = depthProfiles_V1tgt[stat]['avg'][iR]
keep_rois_hva = np.zeros((NROIs,len(gPPIDetails_V23tgt['labels']),nDepths))
for iR, roiID in enumerate(hva_data.keys()):
    for iStat, stat in enumerate(gPPIDetails_V23tgt['labels']):
        keep_rois_hva[iR,iStat,:] = depthProfiles_V23tgt[stat]['avg'][iR]

#define point spread function
p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
Nbins = nDepths #number of bins to use in this analysis

normalize_psf = False #True if you want to normalize the psf by the deepest layer  

decon_rois = depth_deconv(keep_rois,p2t_model,Nbins_model,Nbins,normalize_psf)
decon_rois_hva = depth_deconv(keep_rois_hva,p2t_model,Nbins_model,Nbins,normalize_psf)

#now put back in dictionary
for iStat, stat in enumerate(gPPIDetails_V1tgt['labels']):
    depthProfiles_V1tgt[stat]['avg_decon'] = np.squeeze(np.array(decon_rois)[:,iStat,:])
for iStat, stat in enumerate(gPPIDetails_V23tgt['labels']):
    depthProfiles_V23tgt[stat]['avg_decon'] = np.squeeze(np.array(decon_rois_hva)[:,iStat,:])
    
#%% now make some average plots

prop_err = False # do error propagation?
use_decon = True
useSI = False

V1Stats = gPPIDetails_V1tgt['labels']
V1Colors = gPPIDetails_V1tgt['colors']
V23Stats = gPPIDetails_V23tgt['labels']
V23Colors = gPPIDetails_V23tgt['colors']

[avgV1Profiles, tmp] = compute_avg_depth_profile(depthProfiles_V1tgt,gPPIDetails_V1tgt,{},V1Stats,[],use_decon,prop_err,useSI)  
[avgV23Profiles, tmp] = compute_avg_depth_profile(depthProfiles_V23tgt,gPPIDetails_V23tgt,{},V23Stats,[],use_decon,prop_err,useSI)  


# Plot V1 average profiles
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
xlim = [0,0.5]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,V1Stats,V1Colors,ylim,xlim,dx,dy,Ntext,lcolor,fsize)
plt.title('V1 tgt')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.3,1.5]
plot_avg_depth_profile(p2,avgV23Profiles,V23Stats,V23Colors,ylim,xlim,dx,dy,Ntext,lcolor,fsize)
plt.title('V2 tgt')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI.svg'))
    
#%% Plot each individually

# Plot V1-V23
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.4
    dy = .7
else:
    dx = 1.
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.7]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_lh_tgt','V23_rh_tgt'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.7]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_lh_tgt','V1_rh_tgt'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt.svg'))
    
# Plot V1-V23 iso0
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.4
    dy = .7
else:
    dx = 1.
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.7]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_lh_tgt_iso0','V23_rh_tgt_iso0'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1 iso0')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.7]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_lh_tgt_iso0','V1_rh_tgt_iso0'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23 iso0')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_iso0.svg'))
    
# Plot V1-V23 iso90
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.4
    dy = .7
else:
    dx = 1.
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.7]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_lh_tgt_iso90','V23_rh_tgt_iso90'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1 iso90')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.7]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_lh_tgt_iso90','V1_rh_tgt_iso90'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23 iso90')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_iso90.svg'))
    
# Plot V1-V23 orth
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.4
    dy = .7
else:
    dx = 1.
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.7]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_lh_tgt_orth','V23_rh_tgt_orth'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1 orth')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.7]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_lh_tgt_orth','V1_rh_tgt_orth'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23 orth')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_orth.svg'))
    
    
# Plot V1-V23 sur
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.4
    dy = .7
else:
    dx = 1.
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.7]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_lh_tgt_sur','V23_rh_tgt_sur'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1 sur')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.7]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_lh_tgt_sur','V1_rh_tgt_sur'],['blue','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23 sur')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_sur.svg'))