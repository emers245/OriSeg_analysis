#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 15:12:45 2023

@author: joe

Inclusion Criteria: This code is meant to serve as documentation for the 
inclusion criteria for the OriSeg V1 target ROI analysis on contextual 
modulation.

"""

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
savefigs = True #True #if true save all figures
figDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code/figs/'
fig_format = 'png'
statCorrType = 'bonferroni'

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code'
#datasets = glob.glob(os.path.join(mainDir, 'roi_data', 'pnr???_??_??_?????.csv'))
datasets = ['/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr102_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr161_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr256_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr256_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr328_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr328_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr510_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr510_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr739_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr739_V1_tgt_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr756_V1_tgt_lh_rad10.csv',
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
    
#%% Criterion: Identifiable Stria of Gennari in T1w Profiles
# check and see what the Stria profile looks like in each ROI
nDepths = 6
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
    plt.tight_layout()
    
#%% Criterion: Average SNR > 10 in ROI
# This criteron was already applied before extracting data from server.
# See SNR code on CMRR Servers.

#%% Localizer (ctr - sur) > 50% voxels with < 0.05 p-val

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

#%% Task (iso0+iso90+orth+sur) > 50% voxels with < 0.05 p-val

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

#%% >=10 voxels per depth bin

fdhist = plt.figure(figsize=(15,4))

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = roi[roi['task p-val'] < 0.05]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,int(np.ceil(len(datasets)/2)),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.hlines(10,0,1,'r')
    plt.title(label)
    plt.xlabel("Normalize Depth WM -> GM")
    plt.ylabel("Num. Voxels")
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
fdhist.tight_layout(pad=0.0)

#%% Vein removal
# Use the deepest layer as a proxy for non-vein contaminated voxels
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

#%%
fdhist = plt.figure(figsize=(15,4))

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = roi[roi['task p-val'] < 0.05]
    roi = roi[roi['no_vein']]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,int(np.ceil(len(datasets)/2)),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.hlines(10,0,1,'r')
    plt.title(label)
    plt.xlabel("Normalize Depth WM -> GM")
    plt.ylabel("Num. Voxels")
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
fdhist.tight_layout(pad=0.0)