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
from oriseg_funcs import *

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
mainDir = '.'
datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual', 'pnr???_??_???_??.csv'))
#datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_filled', 'pnr???_??_???_??.csv'))
#or exclude
exclude_initial = ['pnr352_V1_tgt_lh']#,'pnr352_V1_tgt_rh']
#exclude_initial = ['pnr352_V1_tgt_lh_rad10']
for e_i, excl in enumerate(exclude_initial):
    datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_roi_manual',excl+'.csv'))
    #datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_filled',excl+'.csv'))        
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
nDepths = 7
fig = plt.figure(num=1,figsize=(12,8))
fig.clf()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = roi[roi['scale_xy_dist'] > 0]
    dataDict = makeProfile1D(roi['d'].values,
                             10, #number of depths
                             roi['t1'].values,
                             np.min(roi['d'].values), #min depth value
                             np.max(roi['d'].values), #max depth value
                             True) #Use LayNii values
    
    plt.subplot(int(np.ceil(np.sqrt(len(all_data.keys())))), int(np.ceil(np.sqrt(len(all_data.keys())))), 1 + iR)
    plt.plot(dataDict['profile']['depth'],
             dataDict['profile']['avg'][0])
    plt.title('%s (%d vox)' %(label, len(roi)), fontsize=8)
plt.tight_layout()
    
#%% Criterion: Average SNR > 10 in ROI

#try violin plots
spreadF = 2
f = plt.figure()
for k_i, key in enumerate(all_data.keys()):
    df = all_data[key]
    violin_parts = plt.violinplot(df['snr'],positions=[spreadF*k_i],showmedians=True)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('b')
        pc.set_edgecolor('b')
    for pc in violin_parts:
        if not isinstance(violin_parts[pc],list):
            violin_parts[pc].set_edgecolor('b')
    plt.hlines(10,spreadF*k_i-0.5,spreadF*(k_i+0.2)+0.5,color='r')
plt.xticks(np.arange(0,spreadF*len(all_data.keys()),spreadF),all_data.keys(),rotation=15,fontsize=6)
plt.ylabel("SNR")

#%% Criterion: Average Spatial ACF FWHM

#try violin plots
spreadF = 2
f = plt.figure()
for k_i, key in enumerate(all_data.keys()):
    df = all_data[key]
    violin_parts = plt.violinplot(df['acf_fwhm'],positions=[spreadF*k_i],showmedians=True)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('b')
        pc.set_edgecolor('b')
    for pc in violin_parts:
        if not isinstance(violin_parts[pc],list):
            violin_parts[pc].set_edgecolor('b')
    #plt.hlines(1,spreadF*k_i-0.5,spreadF*(k_i+0.2)+0.5,color='r')
plt.xticks(np.arange(0,spreadF*len(all_data.keys()),spreadF),all_data.keys(),rotation=15,fontsize=6)
plt.ylabel("ACF FWHM")

#%% Localizer (ctr - sur) > 50% voxels with < 0.01 p-val

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
    plt.text(0.8,5,'< 0.01 = %d %%' %(100*np.sum(roi['loc p-val'] <= 0.01)/len(roi['loc p-val'])))
    plt.ylim([0,10])
    plt.xlim([0,1])
floc.tight_layout(pad=0.0)

#%% Task (iso0+iso90+orth+sur) > 50% voxels with < 0.01 p-val

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
    plt.text(0.8,5,'< 0.01 = %d %%' %(100*np.sum(roi['task p-val'] <= 0.01)/len(roi['task p-val'])))
    plt.ylim([0,10])
    plt.xlim([0,1])
ftask.tight_layout(pad=0.1)

#%% >=5 voxels per depth bin

fdhist = plt.figure(figsize=(15,4))

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = roi[roi['task p-val'] < 0.01]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,int(np.ceil(len(datasets)/2)),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.hlines(5,0,1,'r')
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
dropout = {'superficial': np.zeros(NROIs),
           'middle': np.zeros(NROIs),
           'deep': np.zeros(NROIs),
           'total': np.zeros(NROIs)} #dropout rates
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
    fthresh = plot_mnv_histograms(lmnv, lmnv[deep], mnv_mask, deep_pct, key, 0, 1, fsize, pad=0.0, figsize=(5,5), fname = 'thresh'+str(k_i))
    
    # Plot depth maps
    dmap = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, 0, 1, [0,100], fsize, fname = 'dmap'+str(k_i), pad=0.0, figsize=(5,5))
        
    #plot thresholded map
    dmap_thresh = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, 0, 1, [0, 100], fsize, fname='dmap_thresh'+str(k_i), mask=mnv_mask, pad=0.0, figsize=(5,5))
        
    # Plot voxel loss at each depth after masking
    fdepth_hist = plot_depth_voxel_loss(z, mnv_mask, nDepths, 1, key, 0, fsize, pad = 0, fname = 'depth'+str(k_i), figsize=(5,5))
    
    # report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(mnv_mask),np.size(mnv),key))
    superficial_mask = (z>=depth_groups['superficial'][0])
    middle_mask = (z>=depth_groups['middle'][0])*(z<depth_groups['middle'][1])
    deep_mask = (z<depth_groups['deep'][1])
    print("\t %d/%d Voxels Survive for superficial %s" %(np.sum(mnv_mask*superficial_mask),np.sum(superficial_mask),key))
    print("\t %d/%d Voxels Survive for middle %s" %(np.sum(mnv_mask*middle_mask),np.sum(middle_mask),key))
    print("\t %d/%d Voxels Survive for deep %s" %(np.sum(mnv_mask*deep_mask),np.sum(deep_mask),key))
    dropout['superficial'][k_i] = 1-np.sum(mnv_mask*superficial_mask)/np.sum(superficial_mask)
    dropout['middle'][k_i] = 1-np.sum(mnv_mask*middle_mask)/np.sum(middle_mask)
    dropout['deep'][k_i] = 1-np.sum(mnv_mask*deep_mask)/np.sum(deep_mask)
    dropout['total'][k_i] = 1-np.sum(mnv_mask)/np.size(mnv)

#Report dropout statistics
print("Average total dropout rate: %s +/- %s" %(np.mean(dropout['total']),np.std(dropout['total'])))
print("\t Superficial: %s +/- %s" %(np.mean(dropout['superficial']),np.std(dropout['superficial'])))
print("\t Middle: %s +/- %s" %(np.mean(dropout['middle']),np.std(dropout['middle'])))
print("\t Deep: %s +/- %s" %(np.mean(dropout['deep']),np.std(dropout['deep'])))

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
    roi = roi[roi['task p-val'] < 0.01]
    roi = roi[roi['no_vein']]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,int(np.ceil(len(datasets)/2)),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.hlines(5,0,1,'r')
    plt.title(label)
    plt.xlabel("Normalize Depth WM -> GM")
    plt.ylabel("Num. Voxels")
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
fdhist.tight_layout(pad=0.0)