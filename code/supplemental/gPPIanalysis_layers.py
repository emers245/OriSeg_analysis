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
savefigs = True #if true save all figures
figType = "svg" #figure format
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
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr256_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr256_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr328_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr328_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr510_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr510_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr739_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr739_V1_tgt_rh_rad10.csv',
   # '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr756_V1_tgt_lh_rad10.csv',
   # '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr756_V1_tgt_rh_rad10.csv'
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
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr256_V23_tgt_lh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr256_V23_tgt_rh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr328_V23_tgt_lh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr328_V23_tgt_rh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr510_V23_tgt_lh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr510_V23_tgt_rh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr739_V23_tgt_lh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr739_V23_tgt_rh.csv',
   # '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr756_V23_tgt_lh.csv',
   # '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr756_V23_tgt_rh.csv'
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
    hva_data[label]['scale_uv_dist'] = np.sqrt(hva_data[label]['x'].values**2 + hva_data[label]['y'].values**2)
    
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

ftask_hva = plt.figure()
for iR, label in enumerate(hva_data.keys()):
    df = hva_data[label]
    
    roi = df
    if 'task pval' in roi.keys():
        roi = roi.rename(columns={'task pval':'task p-val'})
    plt.subplot(int(np.ceil(len(datasets)/2)),2,iR+1)
    plt.hist(roi['task p-val'].values,bins=20,density=True)
    plt.title(label+" task p-val",fontsize=8)
    plt.xlabel("pval")
    plt.text(0.8,5,'< 0.05 = %d %%' %(100*np.sum(roi['task p-val'] <= 0.05)/len(roi['task p-val'])))
    plt.ylim([0,10])
    plt.xlim([0,1])
ftask_hva.tight_layout(pad=0.1)

if savefigs:
    # floc.savefig(os.path.join(figDir,'pvals_loc'))
    # floc_hva.savefig(os.path.join(figDir,'pvals_loc_hva'))
    ftask.savefig(os.path.join(figDir,'pvals_task.%s' %figType),format=figType)
    ftask_hva.savefig(os.path.join(figDir,'pvals_task_hva.%s' %figType),format=figType)

#%% Depth Histograms
# I want to see how much coverage we are getting through depth.
roiRad = 2
fdhist = plt.figure(figsize=(15,4))
nDepths = 3

## THIS IS A HACKY FIX TO GET RID OF DEPTH = 0 VOXELS; SHOULD REMOVE THIS IN THE FUTURE
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    df = df.drop(df[df['d'] == 0].index)
    
    all_data[label] = df

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
    fdhist.savefig(os.path.join(figDir,'depth_hist.%s' %figType),format=figType)
    
roiRad = 2
fdhist_hva = plt.figure(figsize=(15,4))
for iR, label in enumerate(hva_data.keys()):
    df = hva_data[label]
    
    roi = df
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
    fdhist_hva.savefig(os.path.join(figDir,'depth_hist_hva.%s' %figType),format=figType)
    
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
    fthresh.savefig(os.path.join(figDir,'mnv_hist.%s' %figType),format=figType)
    dmap.savefig(os.path.join(figDir,'mnv_depth_map.%s' %figType),format=figType)
    dmap_thresh.savefig(os.path.join(figDir,'mnv_depth_map_thresh.%s' %figType),format=figType)
    fdepth_hist.savefig(os.path.join(figDir,'mnv_depth_hist.%s' %figType),format=figType)

#%% Now for HVAs
for k_i, key in enumerate(hva_data.keys()):

    # calculate log(MNV)
    df = hva_data[key]
    lmnv = get_lmnv(df,key='stdev_xerrts') #log of the mean-normalized variance
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
    fthresh_hva.savefig(os.path.join(figDir,'mnv_hist_hva.%s' %figType),format=figType)
    dmap_hva.savefig(os.path.join(figDir,'mnv_depth_map_hva.%s' %figType),format=figType)
    dmap_thresh_hva.savefig(os.path.join(figDir,'mnv_depth_map_thresh_hva.%s' %figType),format=figType)
    #fdepth_hist_hva.savefig(os.path.join(figDir,'mnv_depth_hist_hva.%s' %figType),format=figType)

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
    f.savefig(os.path.join(figDir,'mnv_summary_errorbars.%s' %figType),format=figType)

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
    plt.plot(spreadF*(k_i+0.4),lmnv_dict[key]['thresh'],color='r',marker='s')
plt.xticks(np.arange(0,spreadF*len(all_data.keys()),spreadF),all_data.keys(),rotation=15,fontsize=6)
plt.ylabel("log(MNV)")

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin.%s' %figType),format=figType)
    
#try violin plots
spreadF = 2
f = plt.figure()
for k_i, key in enumerate(hva_data.keys()):
    df = hva_data[key]
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
    plt.plot(spreadF*(k_i+0.4),lmnv_dict[key]['thresh'],color='r',marker='s')
plt.xticks(np.arange(0,spreadF*len(all_data.keys()),spreadF),hva_data.keys(),rotation=15,fontsize=6)
plt.ylabel("log(MNV)")

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin_hva.%s' %figType),format=figType)
    
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

gPPIDetails_V1tgt = {'labels': ['V23_deep_deveined','V23_middle_deveined','V23_superficial_deveined',
                                 'V23_deep_deveined_iso0','V23_middle_deveined_iso0','V23_superficial_deveined_iso0',
                                 'V23_deep_deveined_iso90','V23_middle_deveined_iso90','V23_superficial_deveined_iso90',
                                 'V23_deep_deveined_orth','V23_middle_deveined_orth','V23_superficial_deveined_orth',
                                 'V23_deep_deveined_sur','V23_middle_deveined_sur','V23_superficial_deveined_sur'],
                     'colors': [np.array([0, 0, 0.5]), np.array([0, 0, 0.5]) * 0.8, np.array([0, 0, 0.5]) * 0.5,
                                np.array([1, 0, 0]), np.array([1, 0, 0]) * 0.8, np.array([1, 0, 0]) * 0.5,
                                np.array([0.5804, 0, 0.8275]), np.array([0.5804, 0, 0.8275]) * 0.8, np.array([0.5804, 0, 0.8275]) * 0.5,
                                np.array([1, 0.6471, 0]), np.array([1, 0.6471, 0]) * 0.8, np.array([1, 0.6471, 0]) * 0.5,
                                np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]) * 0.8, np.array([0.5, 0.5, 0.5]) * 0.5]
                     }
gPPIDetails_V23tgt = {'labels': ['V1_tgt_deep_deveined','V1_tgt_middle_deveined','V1_tgt_superficial_deveined',
                                                     'V1_tgt_deep_deveined_iso0','V1_tgt_middle_deveined_iso0','V1_tgt_superficial_deveined_iso0',
                                                     'V1_tgt_deep_deveined_iso90','V1_tgt_middle_deveined_iso90','V1_tgt_superficial_deveined_iso90',
                                                     'V1_tgt_deep_deveined_orth','V1_tgt_middle_deveined_orth','V1_tgt_superficial_deveined_orth',
                                                     'V1_tgt_deep_deveined_sur','V1_tgt_middle_deveined_sur','V1_tgt_superficial_deveined_sur'],
                     'colors': [np.array([0, 0, 0.5]), np.array([0, 0, 0.5]) * 0.8, np.array([0, 0, 0.5]) * 0.5,
                                np.array([1, 0, 0]), np.array([1, 0, 0]) * 0.8, np.array([1, 0, 0]) * 0.5,
                                np.array([0.5804, 0, 0.8275]), np.array([0.5804, 0, 0.8275]) * 0.8, np.array([0.5804, 0, 0.8275]) * 0.5,
                                np.array([1, 0.6471, 0]), np.array([1, 0.6471, 0]) * 0.8, np.array([1, 0.6471, 0]) * 0.5,
                                np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5]) * 0.8, np.array([0.5, 0.5, 0.5]) * 0.5]
                     }


# gPPIDetails_V1tgt = {'labels': ['V23_superficial_deveined', 
#                           'V23_superficial_deveined_iso0',
#                           'V23_superficial_deveined_iso90',
#                           'V23_superficial_deveined_orth',
#                           'V23_superficial_deveined_sur',
#                           'V23_middle_deveined', 
#                           'V23_middle_deveined_iso0',
#                           'V23_middle_deveined_iso90',
#                           'V23_middle_deveined_orth',
#                           'V23_middle_deveined_sur',
#                           'V23_deep_deveined', 
#                           'V23_deep_deveined_iso0',
#                           'V23_deep_deveined_iso90',
#                           'V23_deep_deveined_orth',
#                           'V23_deep_deveined_sur'],
#                 'colors': ['black', 'black',
#                            'black', 'black',
#                            'black', 'black',
#                            'black', 'black',
#                            'black', 'black',
#                            'black', 'black',
#                            'black', 'black',
#                            'black']}

# gPPIDetails_V23tgt = {'labels': ['V1_tgt_superficial_deveined', 
#                          'V1_tgt_superficial_deveined_iso0',
#                          'V1_tgt_superficial_deveined_iso90',
#                          'V1_tgt_superficial_deveined_orth',
#                          'V1_tgt_superficial_deveined_sur',
#                          'V1_tgt_middle_deveined', 
#                          'V1_tgt_middle_deveined_iso0',
#                          'V1_tgt_middle_deveined_iso90',
#                          'V1_tgt_middle_deveined_orth',
#                          'V1_tgt_middle_deveined_sur',
#                          'V1_tgt_deep_deveined', 
#                          'V1_tgt_deep_deveined_iso0',
#                          'V1_tgt_deep_deveined_iso90',
#                          'V1_tgt_deep_deveined_orth',
#                          'V1_tgt_deep_deveined_sur'],
#                 'colors': ['gray', 'gray',
#                            'gray', 'gray',
#                            'gray', 'gray',
#                            'gray', 'gray',
#                            'gray', 'gray',
#                            'gray', 'gray',
#                            'gray', 'gray',
#                            'gray']}
diffDetails_V1tgt = {}
diffDetails_V1tgt['statIDs'] = {'V23_superficial_deveined_odss': ['V23_superficial_deveined_orth','V23_superficial_deveined_iso90'],
                          'V23_superficial_deveined_fgm': ['V23_superficial_deveined_iso90','V23_superficial_deveined_iso0'],
                          'V23_superficial_deveined_dsi': ['V23_superficial_deveined_orth','V23_superficial_deveined_iso0'],
                          'V23_superficial_deveined_iso-sur': ['V23_superficial_deveined_iso0','V23_superficial_deveined_sur'],
                          'V23_middle_deveined_odss': ['V23_middle_deveined_orth','V23_middle_deveined_iso90'],
                          'V23_middle_deveined_fgm': ['V23_middle_deveined_iso90','V23_middle_deveined_iso0'],
                          'V23_middle_deveined_dsi': ['V23_middle_deveined_orth','V23_middle_deveined_iso0'],
                          'V23_middle_deveined_iso-sur': ['V23_middle_deveined_iso0','V23_middle_deveined_sur'],
                          'V23_deep_deveined_odss': ['V23_deep_deveined_orth','V23_deep_deveined_iso90'],
                          'V23_deep_deveined_fgm': ['V23_deep_deveined_iso90','V23_deep_deveined_iso0'],
                          'V23_deep_deveined_dsi': ['V23_deep_deveined_orth','V23_deep_deveined_iso0'],
                          'V23_deep_deveined_iso-sur': ['V23_deep_deveined_iso0','V23_deep_deveined_sur']
                          }
diffDetails_V1tgt['colors'] = ['green','magenta','cyan','black']

diffDetails_V23tgt = {}
diffDetails_V23tgt['statIDs'] = {'V1_tgt_superficial_deveined_odss': ['V1_tgt_superficial_deveined_orth','V1_tgt_superficial_deveined_iso90'],
                          'V1_tgt_superficial_deveined_fgm': ['V1_tgt_superficial_deveined_iso90','V1_tgt_superficial_deveined_iso0'],
                          'V1_tgt_superficial_deveined_dsi': ['V1_tgt_superficial_deveined_orth','V1_tgt_superficial_deveined_iso0'],
                          'V1_tgt_superficial_deveined_iso-sur': ['V1_tgt_superficial_deveined_iso0','V1_tgt_superficial_deveined_sur'],
                          'V1_tgt_middle_deveined_odss': ['V1_tgt_middle_deveined_orth','V1_tgt_middle_deveined_iso90'],
                          'V1_tgt_middle_deveined_fgm': ['V1_tgt_middle_deveined_iso90','V1_tgt_middle_deveined_iso0'],
                          'V1_tgt_middle_deveined_dsi': ['V1_tgt_middle_deveined_orth','V1_tgt_middle_deveined_iso0'],
                          'V1_tgt_middle_deveined_iso-sur': ['V1_tgt_middle_deveined_iso0','V1_tgt_middle_deveined_sur'],
                          'V1_tgt_deep_deveined_odss': ['V1_tgt_deep_deveined_orth','V1_tgt_deep_deveined_iso90'],
                          'V1_tgt_deep_deveined_fgm': ['V1_tgt_deep_deveined_iso90','V1_tgt_deep_deveined_iso0'],
                          'V1_tgt_deep_deveined_dsi': ['V1_tgt_deep_deveined_orth','V1_tgt_deep_deveined_iso0'],
                          'V1_tgt_deep_deveined_iso-sur': ['V1_tgt_deep_deveined_iso0','V1_tgt_deep_deveined_sur']
                          }
diffDetails_V23tgt['colors'] = ['green','magenta','cyan','black']
profile_method = 'bin' # bin or smooth
roiRad = 2
#pick out ROIs where we're sure of localization
roi_dict = {}
for key in all_data.keys():
    df = all_data[key]
    roi_dict[key] = df['scale_xy_dist'] < roiRad
    all_data[key]['in_tgt'] = df['scale_xy_dist'] < roiRad
for key in hva_data.keys():
    df = hva_data[key]
    roi_dict[key] = df
useSI = False #use suppression index rather than differences (cond1 - cond2 / cond1 + cond2)

#%% Create Full Masks
masks_V1 = {roi:all_data[roi]['in_tgt']*all_data[roi]['sig']*all_data[roi]['no_vein'] for roi in all_data.keys()}
masks_hva = {roi:hva_data[roi]['sig']*hva_data[roi]['no_vein'] for roi in hva_data.keys()}

#%% Compute depth profiles
depthProfiles_V1tgt = compute_all_depth_profiles(all_data,gPPIDetails_V1tgt,profile_method,nDepths,masks_V1,depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
diffProfiles_V1tgt = compute_diff_profiles(all_data,gPPIDetails_V1tgt,diffDetails_V1tgt['statIDs'],profile_method,nDepths,useSI,masks_V1,depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
depthProfiles_V23tgt = compute_all_depth_profiles(hva_data,gPPIDetails_V23tgt,profile_method,nDepths,masks_hva,depthParam='d',radialParam=None,spec_Drange='MinMax')
diffProfiles_V23tgt = compute_diff_profiles(hva_data,gPPIDetails_V23tgt,diffDetails_V23tgt['statIDs'],profile_method,nDepths,useSI,masks_hva,depthParam='d',radialParam=None,spec_Drange='MinMax')

#%% gPPI Centroid Plots
# Let's take a look at raw voxel betas across depth by condition

roiRad = 2
Nsubj = len(all_data.keys())

#plot centroids for each condition and ROI
plot_centroids(all_data, masks_V1, gPPIDetails_V1tgt, roiRad, nDepths, xlim=[-1,1])
        
if savefigs:
    for l in gPPIDetails_V1tgt['labels']:
        plt.figure(l)
        plt.savefig(os.path.join(figDir,'centroids_%s.%s' %(l,figType)),format=figType)

Nsubj = len(hva_data.keys())

#plot centroids for each condition and ROI
plot_centroids(hva_data, masks_hva, gPPIDetails_V23tgt, roiRad, nDepths, xlim=[-1,1], radParam=None)
        
if savefigs:
    for l in gPPIDetails_V23tgt['labels']:
        plt.figure(l)
        plt.savefig(os.path.join(figDir,'centroids_%s.%s' %(l,figType)),format=figType) 

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
V1Diffs = list(diffDetails_V1tgt['statIDs'].keys())
V23Stats = gPPIDetails_V23tgt['labels']
V23Colors = gPPIDetails_V23tgt['colors']
V23Diffs = list(diffDetails_V23tgt['statIDs'].keys())

[avgV1Profiles, avgV1Diffs] = compute_avg_depth_profile(depthProfiles_V1tgt,gPPIDetails_V1tgt,diffDetails_V1tgt['statIDs'],V1Stats,V1Diffs,use_decon,prop_err,useSI)  
[avgV23Profiles, avgV23Diffs] = compute_avg_depth_profile(depthProfiles_V23tgt,gPPIDetails_V23tgt,diffDetails_V23tgt['statIDs'],V23Stats,V23Diffs,use_decon,prop_err,useSI)  


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
xlim = [-0.2,0.2]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,V1Stats,V1Colors,ylim,xlim,dx,dy,Ntext,lcolor,fsize)
plt.title('V1 tgt')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.2]
plot_avg_depth_profile(p2,avgV23Profiles,V23Stats,V23Colors,ylim,xlim,dx,dy,Ntext,lcolor,fsize)
plt.title('V2 tgt')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI.%s' %figType),format=figType)
    
#%% Plot each individually
# A bit of documentation of terminology
# Each plot has a title VX -> VY where VX is the location of the seed regressor
# and VY is the location of the voxel beta weights being plotted. Each of these
# seed and target locations have three subfields (deep, middle, superficial) 
# based on depth. The subfields for the seeds are set by which seed ROIs we 
# used for the gPPI analyses, while the subfields for the targets are defined 
# within this code and can therefore have any number of depths. I chose 3 to
# match the number of seed ROIs for each region.

# Plot V1-V23
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.19
    dy = .7
else:
    dx = 0.19
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.3]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_superficial_deveined','V23_middle_deveined','V23_deep_deveined'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.3]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_tgt_superficial_deveined','V1_tgt_middle_deveined','V1_tgt_deep_deveined'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_combined.%s' %figType),format=figType)
    
# Plot V1-V23 iso0
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.19
    dy = .7
else:
    dx = 0.19
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.3]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_superficial_deveined_iso0','V23_middle_deveined_iso0','V23_deep_deveined_iso0'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1 iso0')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.3]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_tgt_superficial_deveined_iso0','V1_tgt_middle_deveined_iso0','V1_tgt_deep_deveined_iso0'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23 iso0')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_iso0_combined.%s' %figType),format=figType)
    
# Plot V1-V23 iso90
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.19
    dy = .7
else:
    dx = 0.19
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.3]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_superficial_deveined_iso90','V23_middle_deveined_iso90','V23_deep_deveined_iso90'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1 iso90')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.3]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_tgt_superficial_deveined_iso90','V1_tgt_middle_deveined_iso90','V1_tgt_deep_deveined_iso90'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23 iso90')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_iso90_combined.%s' %figType),format=figType)
    
# Plot V1-V23 orth
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.19
    dy = .7
else:
    dx = 0.19
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.3]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_superficial_deveined_orth','V23_middle_deveined_orth','V23_deep_deveined_orth'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1 orth')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.3]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_tgt_superficial_deveined_orth','V1_tgt_middle_deveined_orth','V1_tgt_deep_deveined_orth'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23 orth')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_orth_combined.%s' %figType),format=figType)
    
    
# Plot V1-V23 sur
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)
    
fig.clf()
fsize = 14
    
p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 0.19
    dy = .7
else:
    dx = 0.19
    dy = .7

ylim = [-0.02,1.02]
xlim = [-0.2,0.3]
Ntext = [0.4,0.05]
plot_avg_depth_profile(p1,avgV1Profiles,['V23_superficial_deveined_sur','V23_middle_deveined_sur','V23_deep_deveined_sur'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V23 -> V1 sur')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.2,0.3]
plot_avg_depth_profile(p2,avgV23Profiles,['V1_tgt_superficial_deveined_sur','V1_tgt_middle_deveined_sur','V1_tgt_deep_deveined_sur'],[[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,showSig=True,statCorrType=statCorrType)
plt.title('V1 -> V23 sur')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_sur_combined.%s' %figType),format=figType)
    
#%% For each condition, create nDepthsxnDepths grid

def plot_gPPI_mat(data_mat,p_mat,seed,target,Nvox=None,cbar_lims = [-0.15,0.15],title_add='',fig=None,ax=None,fontsize=12,invert_yaxis=True,cbar=True):
    """ Plots the gPPI matrix """
    
    # Creating a figure and axis if one was not passed
    if fig==None or ax==None:
        fig, ax = plt.subplots()

    # Plotting the heatmap
    heatmap = ax.imshow(data_mat, cmap='bwr', interpolation='nearest', vmin=cbar_lims[0], vmax=cbar_lims[1])

    # Adding text labels to the cells
    for i in range(data_mat.shape[0]):
        for j in range(data_mat.shape[1]):
            if p_mat[i][j] < 0.05 and p_mat[i][j] >= 0.01:
                text = ax.text(j, i, f"{data_mat[i, j]:.2f}%*\np={p_mat[i, j]:.3f}", ha='center', va='center', color='black', fontsize=0.6*fontsize, weight='bold')
            elif p_mat[i][j] < 0.01 and p_mat[i][j] >= 0.0005:
                text = ax.text(j, i, f"{data_mat[i, j]:.2f}%**\np={p_mat[i, j]:.3f}", ha='center', va='center', color='black', fontsize=0.6*fontsize, weight='bold')
            elif p_mat[i][j] < 0.0005:
                text = ax.text(j, i, f"{data_mat[i, j]:.2f}%**\np<0.001", ha='center', va='center', color='black', fontsize=0.6*fontsize, weight='bold')
            else:
                text = ax.text(j, i, f"{data_mat[i, j]:.2f}%\np={p_mat[i, j]:.2f}", ha='center', va='center', color='black', fontsize=0.6*fontsize)

    if cbar:
        cbar = plt.colorbar(heatmap,ax=ax)
        cbar.set_label("% BOLD Change",fontsize=0.6*fontsize)
        cbar.ax.tick_params(labelsize=0.6*fontsize)
    ax.set_xticks(np.arange(data_mat.shape[1]))
    ax.set_yticks(np.arange(data_mat.shape[0]))
    if Nvox==None:
        ax.set_xticklabels(['deep','middle','superficial'],fontsize=0.6*fontsize)
        ax.set_yticklabels(['deep','middle','superficial'],fontsize=0.6*fontsize)
    else:
        ax.set_xticklabels(['deep \n N=%d' %Nvox['seed'][0],'middle \n N=%d' %Nvox['seed'][1],'superficial \n N=%d' %Nvox['seed'][2]],fontsize=0.6*fontsize)
        ax.set_yticklabels(['deep \n N=%d' %Nvox['targ'][0],'middle \n N=%d' %Nvox['targ'][1],'superficial \n N=%d' %Nvox['targ'][2]],fontsize=0.6*fontsize)
    ax.set_xlabel(seed,fontsize=0.8*fontsize)
    ax.set_ylabel(target,fontsize=0.8*fontsize)

    # Setting the title for the plot
    ax.set_title('%s -> %s %s' %(seed,target,title_add),fontsize=fontsize)
    if invert_yaxis:
        ax.invert_yaxis()
    
    plt.show()
    
    return(heatmap,fig,ax)
    
#create plot
plot_baseline = False #True if you would like to analyze the baseline V1-V2/3 interactions
use_all_meas = True #True if you would like to use all measurements in your correction for multiple comparisons

if plot_baseline:
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2,5)
else:
    fig, ((ax2, ax3, ax4, ax5), (ax7, ax8, ax9, ax10)) = plt.subplots(2,4)
fig.set_figwidth(14)
fig.set_figheight(6)
fontsize=12

#Get all corrected p-values
all_corrected_pvalues = {"V1":{},"V23":{}}
if use_all_meas:
    if plot_baseline:
        V1_condition_list = gPPIDetails_V1tgt['labels']
        V23_condition_list = gPPIDetails_V23tgt['labels']
    else:
        V1_condition_list = ['V23_deep_deveined_iso0',
               'V23_middle_deveined_iso0', 'V23_superficial_deveined_iso0',
               'V23_deep_deveined_iso90', 'V23_middle_deveined_iso90',
               'V23_superficial_deveined_iso90', 'V23_deep_deveined_orth',
               'V23_middle_deveined_orth', 'V23_superficial_deveined_orth',
               'V23_deep_deveined_sur', 'V23_middle_deveined_sur',
               'V23_superficial_deveined_sur']
        V23_condition_list = ['V1_tgt_deep_deveined_iso0', 'V1_tgt_middle_deveined_iso0',
               'V1_tgt_superficial_deveined_iso0', 'V1_tgt_deep_deveined_iso90',
               'V1_tgt_middle_deveined_iso90',
               'V1_tgt_superficial_deveined_iso90', 'V1_tgt_deep_deveined_orth',
               'V1_tgt_middle_deveined_orth', 'V1_tgt_superficial_deveined_orth',
               'V1_tgt_deep_deveined_sur', 'V1_tgt_middle_deveined_sur',
               'V1_tgt_superficial_deveined_sur']
    all_pvalues = np.concatenate(([avgV1Profiles[c]['p-vals'].pvalue for c in V1_condition_list],[avgV23Profiles[c]['p-vals'].pvalue for c in V23_condition_list])) #get all p-values for all measurements in one list
    condition_list = np.concatenate((V1_condition_list,V23_condition_list))
    corrected_list = multipletests(all_pvalues.flatten(),method=statCorrType)[1]
    for c_i, c in enumerate(condition_list):
        if 'V1' in c:
            all_corrected_pvalues['V1'][c] = corrected_list[c_i:c_i+nDepths]
        if 'V23' in c:
            all_corrected_pvalues['V23'][c] = corrected_list[c_i:c_i+nDepths]

if plot_baseline:
    # V2/3 -> V1
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined','V23_middle_deveined','V23_deep_deveined']
    if use_all_meas:
        for c_i, c in enumerate(condition_list):
            data_mat[c_i,:] = avgV1Profiles[c]['avg']
            p_mat[c_i,:] = all_corrected_pvalues['V23'][c]
    else:
        for c_i, c in enumerate(condition_list):
            data_mat[c_i,:] = avgV1Profiles[c]['avg']
            corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
            p_mat[c_i,:] = corrected_pvalues
        
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',fig=fig,ax=ax1,cbar=False)

    # if savefigs:
    #     fig.savefig(os.path.join(figDir,'V23->V1_mat.%s' %figType),format=figType)
    
    # V1 -> V2/3
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined','V1_tgt_middle_deveined','V1_tgt_deep_deveined']
    if use_all_meas:
        for c_i, c in enumerate(condition_list):
            data_mat[c_i,:] = avgV23Profiles[c]['avg']
            p_mat[c_i,:] = all_corrected_pvalues['V1'][c]
    else:
        for c_i, c in enumerate(condition_list):
            data_mat[c_i,:] = avgV23Profiles[c]['avg']
            corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
            p_mat[c_i,:] = corrected_pvalues
            
    plot_gPPI_mat(data_mat,p_mat,'V1','V23',fig=fig,ax=ax6,cbar=False)
    
    # if savefigs:
    #     fig.savefig(os.path.join(figDir,'V1->V23_mat.%s' %figType),format=figType)

# V2/3 -> V1 iso0
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_superficial_deveined_iso0','V23_middle_deveined_iso0','V23_deep_deveined_iso0']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso0','V1 iso0',fig=fig,ax=ax2,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_iso0_mat'))

# V1 -> V2/3 iso0
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_superficial_deveined_iso0','V1_tgt_middle_deveined_iso0','V1_tgt_deep_deveined_iso0']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V1 iso0','V23 iso0',fig=fig,ax=ax7,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_iso0_mat.%s' %figType),format=figType)

# V2/3 -> V1 iso90
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_superficial_deveined_iso90','V23_middle_deveined_iso90','V23_deep_deveined_iso90']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso90','V1 iso90',fig=fig,ax=ax3,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_iso90_mat'))

# V1 -> V2/3 iso90
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_superficial_deveined_iso90','V1_tgt_middle_deveined_iso90','V1_tgt_deep_deveined_iso90']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V1 iso90','V23 iso90',fig=fig,ax=ax8,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_iso90_mat.%s' %figType),format=figType)

# V2/3 -> V1 orth
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_superficial_deveined_orth','V23_middle_deveined_orth','V23_deep_deveined_orth']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 orth','V1 orth',fig=fig,ax=ax4,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_orth_mat'))

# V1 -> V2/3 orth
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_superficial_deveined_orth','V1_tgt_middle_deveined_orth','V1_tgt_deep_deveined_orth']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V1 orth','V23 orth',fig=fig,ax=ax9,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_orth_mat.%s' %figType),format=figType)

# V2/3 -> V1 sur
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_superficial_deveined_sur','V23_middle_deveined_sur','V23_deep_deveined_sur']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 sur','V1 sur',fig=fig,ax=ax5,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_sur_mat'))

# V1 -> V2/3 sur
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_superficial_deveined_sur','V1_tgt_middle_deveined_sur','V1_tgt_deep_deveined_sur']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
heatmap, fig, ax = plot_gPPI_mat(data_mat,p_mat,'V1 sur','V23 sur',fig=fig,ax=ax10,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_sur_mat.%s' %figType),format=figType)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(heatmap, cax=cbar_ax)
cbar.set_label('Colorbar Label')
cbar.set_label("% BOLD Change",fontsize=0.6*fontsize)
cbar.ax.tick_params(labelsize=0.6*fontsize)

plt.tight_layout(rect=(0,0,0.9,1))

if savefigs:
    fig.savefig(os.path.join(figDir,'gPPI_conditions_mat.%s' %figType),format=figType)

#%% Now do the same thing for contrast conditions

#create plot
use_all_meas = True #True if you would like to use all measurements in your correction for multiple comparisons


fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4)
fig.set_figwidth(14)
fig.set_figheight(6)
fontsize=12

#Get all corrected p-values
all_corrected_diff_pvalues = {"V1":{},"V23":{}}
if use_all_meas:
    V1_condition_list = list(diffDetails_V1tgt['statIDs'].keys())
    V23_condition_list = list(diffDetails_V23tgt['statIDs'].keys())
    all_diff_pvalues = np.concatenate(([avgV1Diffs[c]['p-vals'].pvalue for c in V1_condition_list],[avgV23Diffs[c]['p-vals'].pvalue for c in V23_condition_list])) #get all p-values for all measurements in one list
    condition_list = np.concatenate((V1_condition_list,V23_condition_list))
    corrected_list = multipletests(all_diff_pvalues.flatten(),method=statCorrType)[1]
    for c_i, c in enumerate(condition_list):
        if 'V1' in c:
            all_corrected_diff_pvalues['V1'][c] = corrected_list[c_i:c_i+nDepths]
        if 'V23' in c:
            all_corrected_diff_pvalues['V23'][c] = corrected_list[c_i:c_i+nDepths]

# V2/3 -> V1 iso0 - sur
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_superficial_deveined_iso-sur','V23_middle_deveined_iso-sur','V23_deep_deveined_iso-sur']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Diffs[c]['avg']
        p_mat[c_i,:] = all_corrected_diff_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Diffs[c]['avg']
        top = avgV1Diffs[c]['avg'] + avgV1Diffs[c]['stdev']/np.sqrt(np.shape(avgV1Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV1Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso-sur','V1 iso-sur',fig=fig,ax=ax1,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_iso-sur_mat.%s' %figType),format=figType)

# V1 -> V2/3 iso0 - sur
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_superficial_deveined_iso-sur','V1_tgt_middle_deveined_iso-sur','V1_tgt_deep_deveined_iso-sur']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        p_mat[c_i,:] = all_corrected_diff_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        top = avgV23Diffs[c]['avg'] + avgV23Diffs[c]['stdev']/np.sqrt(np.shape(avgV23Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV23Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
plot_gPPI_mat(data_mat,p_mat,'V1 iso-sur','V23 iso-sur',fig=fig,ax=ax5,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_iso-sur_mat.%s' %figType),format=figType)

# V2/3 -> V1 odss
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_superficial_deveined_odss','V23_middle_deveined_odss','V23_deep_deveined_odss']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Diffs[c]['avg']
        p_mat[c_i,:] = all_corrected_diff_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Diffs[c]['avg']
        top = avgV1Diffs[c]['avg'] + avgV1Diffs[c]['stdev']/np.sqrt(np.shape(avgV1Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV1Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 ODSS','V1 ODSS',fig=fig,ax=ax2,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_odss_mat.%s' %figType),format=figType)

# V1 -> V2/3 odss
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_superficial_deveined_odss','V1_tgt_middle_deveined_odss','V1_tgt_deep_deveined_odss']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        p_mat[c_i,:] = all_corrected_diff_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        top = avgV23Diffs[c]['avg'] + avgV23Diffs[c]['stdev']/np.sqrt(np.shape(avgV23Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV23Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V1 ODSS','V23 ODSS',fig=fig,ax=ax6,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_odss_mat.%s' %figType),format=figType)

# V2/3 -> V1 fgm
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_superficial_deveined_fgm','V23_middle_deveined_fgm','V23_deep_deveined_fgm']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Diffs[c]['avg']
        p_mat[c_i,:] = all_corrected_diff_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Diffs[c]['avg']
        top = avgV1Diffs[c]['avg'] + avgV1Diffs[c]['stdev']/np.sqrt(np.shape(avgV1Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV1Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 FGM','V1 FGM',fig=fig,ax=ax3,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_fgm_mat.%s' %figType),format=figType)

# V1 -> V2/3 fgm
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_superficial_deveined_fgm','V1_tgt_middle_deveined_fgm','V1_tgt_deep_deveined_fgm']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        p_mat[c_i,:] = all_corrected_diff_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        top = avgV23Diffs[c]['avg'] + avgV23Diffs[c]['stdev']/np.sqrt(np.shape(avgV23Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV23Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V1 FGM','V23 FGM',fig=fig,ax=ax7,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_fgm_mat.%s' %figType),format=figType)

# V2/3 -> V1 dsi
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_superficial_deveined_dsi','V23_middle_deveined_dsi','V23_deep_deveined_dsi']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Diffs[c]['avg']
        p_mat[c_i,:] = all_corrected_diff_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV1Diffs[c]['avg']
        top = avgV1Diffs[c]['avg'] + avgV1Diffs[c]['stdev']/np.sqrt(np.shape(avgV1Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV1Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 $\Delta$SI','V1 $\Delta$SI',fig=fig,ax=ax4,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_dsi_mat.%s' %figType),format=figType)

# V1 -> V2/3 dsi
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_superficial_deveined_dsi','V1_tgt_middle_deveined_dsi','V1_tgt_deep_deveined_dsi']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        p_mat[c_i,:] = all_corrected_diff_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        top = avgV23Diffs[c]['avg'] + avgV23Diffs[c]['stdev']/np.sqrt(np.shape(avgV23Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV23Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
heatmap, fig, ax = plot_gPPI_mat(data_mat,p_mat,'V1 $\Delta$SI','V23 $\Delta$SI',fig=fig,ax=ax8,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_dsi_mat.%s' %figType),format=figType)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(heatmap, cax=cbar_ax)
cbar.set_label('Colorbar Label')
cbar.set_label("% BOLD Change",fontsize=0.6*fontsize)
cbar.ax.tick_params(labelsize=0.6*fontsize)

plt.tight_layout(rect=(0,0,0.9,1))

if savefigs:
    fig.savefig(os.path.join(figDir,'gPPI_diffs_mat.%s' %figType),format=figType)
   
#%% Try individual level analysis for main conditions
fsize=7

for iR, label in enumerate(all_data.keys()):
    
    #create plot
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5,2)
    fig.set_figwidth(5)
    fig.set_figheight(7.5)
    
    # V2/3 -> V1 iso0 - sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined','V23_middle_deveined','V23_deep_deveined']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target; should be the same for all conditions
        
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="",fig=fig,ax=ax1,fontsize=fsize)
        
    # V1 -> V2/3 iso0 - sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined','V1_tgt_middle_deveined','V1_tgt_deep_deveined']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="",fig=fig,ax=ax2,fontsize=fsize)

    # V2/3 -> V1 iso0
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined_iso0','V23_middle_deveined_iso0','V23_deep_deveined_iso0']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="iso0",fig=fig,ax=ax3,fontsize=fsize)
        
    # V1 -> V2/3 iso0
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined_iso0','V1_tgt_middle_deveined_iso0','V1_tgt_deep_deveined_iso0']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="iso0",fig=fig,ax=ax4,fontsize=fsize)

    # V2/3 -> V1 iso90
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined_iso90','V23_middle_deveined_iso90','V23_deep_deveined_iso90']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="iso90",fig=fig,ax=ax5,fontsize=fsize)
        
    # V1 -> V2/3 iso90
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined_iso90','V1_tgt_middle_deveined_iso90','V1_tgt_deep_deveined_iso90']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="iso90",fig=fig,ax=ax6,fontsize=fsize)
    
    # V2/3 -> V1 orth
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined_orth','V23_middle_deveined_orth','V23_deep_deveined_orth']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="orth",fig=fig,ax=ax7,fontsize=fsize)
        
    # V1 -> V2/3 orth
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined_orth','V1_tgt_middle_deveined_orth','V1_tgt_deep_deveined_orth']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="orth",fig=fig,ax=ax8,fontsize=fsize)

    # V2/3 -> V1 sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined_sur','V23_middle_deveined_sur','V23_deep_deveined_sur']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="sur",fig=fig,ax=ax9,fontsize=fsize)
        
    # V1 -> V2/3 sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined_sur','V1_tgt_middle_deveined_sur','V1_tgt_deep_deveined_sur']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="sur",fig=fig,ax=ax10,fontsize=fsize)

    plt.suptitle(label)
    plt.tight_layout(pad=0.1,rect=(0,0,1,0.95))

    if savefigs:
        fig.savefig(os.path.join(figDir,'V23-V1_mat_%s.%s' %(label,figType)),format=figType)

#%% Try individual level analysis of gPPI for condition differences
fsize=8

for iR, label in enumerate(all_data.keys()):
    
    #create plot
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4,2)
    fig.set_figwidth(5)
    fig.set_figheight(7.5)
    
    # V2/3 -> V1 iso0 - sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined_iso-sur','V23_middle_deveined_iso-sur','V23_deep_deveined_iso-sur']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V1tgt[c]['avg'][iR]
        top = diffProfiles_V1tgt[c]['avg'][iR] + diffProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    Nvox = {'seed': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target; should be the same for all conditions
        
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="iso-sur",fig=fig,ax=ax1,fontsize=fsize)
        
    # V1 -> V2/3 iso0 - sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined_iso-sur','V1_tgt_middle_deveined_iso-sur','V1_tgt_deep_deveined_iso-sur']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V23tgt[c]['avg'][iR]
        top = diffProfiles_V23tgt[c]['avg'][iR] + diffProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="iso-sur",fig=fig,ax=ax2,fontsize=fsize)

    # V2/3 -> V1 odss
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined_odss','V23_middle_deveined_odss','V23_deep_deveined_odss']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V1tgt[c]['avg'][iR]
        top = diffProfiles_V1tgt[c]['avg'][iR] + diffProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="ODSS",fig=fig,ax=ax3,fontsize=fsize)
        
    # V1 -> V2/3 odss
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined_odss','V1_tgt_middle_deveined_odss','V1_tgt_deep_deveined_odss']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V23tgt[c]['avg'][iR]
        top = diffProfiles_V23tgt[c]['avg'][iR] + diffProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="ODSS",fig=fig,ax=ax4,fontsize=fsize)

    # V2/3 -> V1 fgm
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined_fgm','V23_middle_deveined_fgm','V23_deep_deveined_fgm']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V1tgt[c]['avg'][iR]
        top = diffProfiles_V1tgt[c]['avg'][iR] + diffProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="FGM",fig=fig,ax=ax5,fontsize=fsize)
        
    # V1 -> V2/3 fgm
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined_fgm','V1_tgt_middle_deveined_fgm','V1_tgt_deep_deveined_fgm']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V23tgt[c]['avg'][iR]
        top = diffProfiles_V23tgt[c]['avg'][iR] + diffProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="FGM",fig=fig,ax=ax6,fontsize=fsize)
    
    # V2/3 -> V1 dsi
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_superficial_deveined_dsi','V23_middle_deveined_dsi','V23_deep_deveined_dsi']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V1tgt[c]['avg'][iR]
        top = diffProfiles_V1tgt[c]['avg'][iR] + diffProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="$\Delta$SI",fig=fig,ax=ax7,fontsize=fsize)
        
    # V1 -> V2/3 dsi
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_superficial_deveined_dsi','V1_tgt_middle_deveined_dsi','V1_tgt_deep_deveined_dsi']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V23tgt[c]['avg'][iR]
        top = diffProfiles_V23tgt[c]['avg'][iR] + diffProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-0.5,0.5],title_add="$\Delta$SI",fig=fig,ax=ax8,fontsize=fsize)

    plt.suptitle(label)
    plt.tight_layout(pad=0.1,rect=(0,0,1,0.95))

    if savefigs:
        fig.savefig(os.path.join(figDir,'V23-V1_diffs_mat_%s.%s' %(label,figType)),format=figType)