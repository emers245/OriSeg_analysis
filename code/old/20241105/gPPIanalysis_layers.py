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
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.formula.api import ols

#Import custom functions
from oriseg_funcs import *

plt.close('all')
    
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]
savefigs = False #True #if true save all figures
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '.'
figDir = mainDir+'/figs/individual_ROIs/'
figType = 'svg'
statCorrType = 'fdr_bh' #'bonferroni'

#Set random seed
np.random.seed(68752)

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '.'
datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual', 'pnr???_??_???_??.csv'))
exclude_initial = ['pnr143_V1_tgt_lh','pnr143_V1_tgt_rh',
                   'pnr161_V1_tgt_lh','pnr161_V1_tgt_rh',
                   'pnr352_V1_tgt_lh','pnr352_V1_tgt_rh',
                   'pnr495_V1_tgt_lh','pnr495_V1_tgt_rh',
                   'pnr579_V1_tgt_lh','pnr579_V1_tgt_rh',
                   'pnr668_V1_tgt_lh','pnr668_V1_tgt_rh',
                   'pnr685_V1_tgt_lh','pnr685_V1_tgt_rh',
                   'pnr713_V1_tgt_lh','pnr713_V1_tgt_rh',
                   'pnr822_V1_tgt_lh','pnr822_V1_tgt_rh']
for e_i, excl in enumerate(exclude_initial):
    datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_roi_manual',excl+'.csv'))
datasets.sort()
Ndsets = len(datasets)

roiRad = 2 #1.
import pandas as pd
all_data = {}
for dataset in datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    all_data[f] = pd.read_csv(dataset, sep=',', index_col=False)

hva_datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/V23_filled', 'pnr???_???_???_??.csv'))
exclude_hva = []
for e_i, excl in enumerate(exclude_hva):
    hva_datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_roi_manual',excl+'.csv'))
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

#%% Standardize pval names
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    if 'task pval' in df.keys():
        df = df.rename(columns={'task pval':'task p-val'})
    if 'loc pval' in df.keys():
        df = df.rename(columns={'loc pval':'loc p-val'})
        
    all_data[label] = df
    
#%% Localize ROIs in V1

for key in all_data.keys():
    df = all_data[key]
    all_data[key]['in_tgt'] = df['scale_xy_dist'] < roiRad

#%% Histograms of p-values

def data_hist(all_data, key, pthresh=0.01, ylim=[0,10], xlim=[0,1], mask=None):
    # Plots a histogram of data from the data dictionary given a key
    
    f = plt.figure()
    
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        
        if mask:
            roi = df[df[mask]]
        else:
            roi = df
        plt.subplot(int(np.ceil(len(datasets)/2)),2,iR+1)
        plt.hist(roi[key].values,bins=20,density=True)
        plt.title(label+" "+key,fontsize=8)
        plt.xlabel(key)
        if 'p-val' in key:
            plt.text(0.8,5,'p < %.3f = %d %%' %(pthresh,100*np.sum(roi[key] <= pthresh)/len(roi[key])))
        plt.ylim([0,10])
        plt.xlim([0,1])
    f.tight_layout(pad=0.1)
    
    return f

ftask = data_hist(all_data,'task p-val',mask='in_tgt')

ftask_hva = data_hist(hva_data, 'task p-val')

if savefigs:
    # floc.savefig(os.path.join(figDir,'pvals_loc'))
    # floc_hva.savefig(os.path.join(figDir,'pvals_loc_hva'))
    ftask.savefig(os.path.join(figDir,'pvals_task.%s' %figType),format=figType)
    ftask_hva.savefig(os.path.join(figDir,'pvals_task_hva.%s' %figType),format=figType)

#%% Depth Histograms

def depth_hist(all_data, mask=None):
    # Plot depth histograms for all ROIs
    f = plt.figure(figsize=(15,4))
    
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        
        if mask:
            roi = df[df[mask]]
        else:
            roi = df
        plt.subplot(2,int(np.ceil(len(datasets)/2)),iR+1)
        plt.hist(roi['d'].values,bins=nDepths)
        plt.title(label)
        plt.xlabel("Normalize Depth WM -> GM")
        plt.ylabel("Num. Voxels")
        plt.legend(['N='+str(len(roi)),], fontsize = 6)
    f.tight_layout(pad=0.0)
    
    return f

# I want to see how much coverage we are getting through depth.
nDepths = 3

## THIS IS A HACKY FIX TO GET RID OF DEPTH = 0 VOXELS; SHOULD REMOVE THIS IN THE FUTURE
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    df = df.drop(df[df['d'] == 0].index)
    
    all_data[label] = df

# Plot
fdhist = depth_hist(all_data, mask='in_tgt')

if savefigs:
    fdhist.savefig(os.path.join(figDir,'depth_hist.%s' %figType),format=figType)
    
# Plot
fdhist_hva = depth_hist(hva_data)

if savefigs:
    fdhist_hva.savefig(os.path.join(figDir,'depth_hist_hva.%s' %figType),format=figType)
    
#%% Use the deepest layer as a proxy for non-vein contaminated voxels
# Then define the threshold based on this distribution

deep_pct = 10 #percentile to call deep layers
conditions = ['iso0','iso90','orth','sur']
depth_groups = {'deep': None, 'middle': None, 'superficial': None} #depth bins for visualization; if empty this is decided separately for each ROI
depth_labels = ['superficial','middle','deep'] #put them in the right order
depth_var = 'd'
x_var = 'x'
y_var = 'y'
sd_thresh = 2 #how many st. dev. of the deep layer mean to use as the threshold
mask_dict = {} #create a mask dictionary
lmnv_dict = {key:{'mean':0,'std':0,'thresh':0,'deep_mean':0,'deep_std':0} for key in all_data.keys()} #thresh dictionary
fsize=8 #fontsize of title
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
    
    # determine depth bins for visualization
    set_indiv_depth = False
    for d_i, dlabel in enumerate(depth_labels):
        min_depth = df[depth_var].min()
        max_depth = df[depth_var].max()
        range_depth = max_depth-min_depth
        step_depth = range_depth/len(depth_labels)
        if not depth_groups[dlabel]:
            set_indiv_depth = True
            print("Adding depth bin boundaries for "+key+" "+dlabel)
            lower_lim = min_depth+d_i*step_depth
            print(f"\t lower lim = {lower_lim}")
            upper_lim = min_depth+(d_i+1)*step_depth
            print(f"\t upper lim = {upper_lim}")
            depth_groups[dlabel] = [lower_lim, upper_lim]
    
    # Plot distributions
    fthresh = plot_mnv_histograms(lmnv, lmnv[deep], mnv_mask, deep_pct, key, k_i, NROIs, fsize, pad=0.0, figsize=(15,3))
    
    # Plot depth maps
    dmap = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0,100], fsize, fname = 'dmap', pad=0.0)
        
    #plot thresholded map
    dmap_thresh = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0, 100], fsize, fname='dmap_thresh', mask=mnv_mask, pad=0.0)
        
    # Plot voxel loss at each depth after masking
    fdepth_hist = plot_depth_voxel_loss(z, mnv_mask, nDepths, NROIs, key, k_i, fsize)
    
    # Reset depth groups if necessary
    if set_indiv_depth:
        for d_i, dlabel in enumerate(depth_labels):
            depth_groups[dlabel] = None
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(mnv_mask),np.size(mnv),key))
    
if savefigs:
    fthresh.savefig(os.path.join(figDir,'mnv_hist.%s' %figType),format=figType)
    dmap.savefig(os.path.join(figDir,'mnv_depth_map.%s' %figType),format=figType)
    dmap_thresh.savefig(os.path.join(figDir,'mnv_depth_map_thresh.%s' %figType),format=figType)
    fdepth_hist.savefig(os.path.join(figDir,'mnv_depth_hist.%s' %figType),format=figType)

#%% Compare thresholds between subjects

def lmnv_violin_plots(all_data, spreadF=2):
    # Plot violin plots for log mean normalized variance
    f = plt.figure()
    for k_i, key in enumerate(all_data.keys()):
        df = all_data[key]
        lmnv = get_lmnv(df,key='stdev_xerrts') #log of the mean-normalized variance
        notInf = (lmnv != np.inf) & (lmnv != -np.inf)
        lmnv = lmnv[notInf] #sometimes there are infinities from division by zero. I'll ignore these.
        [deep_mean, deep_std, deep] = get_deep_layer_dist(df,depth_var,deep_pct)
        deep = deep[notInf]
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
    
    return f

#try violin plots
f = lmnv_violin_plots(all_data)

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin.%s' %(fig_format)))

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
    
    # determine depth bins for visualization
    set_indiv_depth = False
    for d_i, dlabel in enumerate(depth_labels):
        min_depth = df[depth_var].min()
        max_depth = df[depth_var].max()
        range_depth = max_depth-min_depth
        step_depth = range_depth/len(depth_labels)
        if not depth_groups[dlabel]:
            set_indiv_depth = True
            print("Adding depth bin boundaries for "+key+" "+dlabel)
            lower_lim = min_depth+d_i*step_depth
            print(f"\t lower lim = {lower_lim}")
            upper_lim = min_depth+(d_i+1)*step_depth
            print(f"\t upper lim = {upper_lim}")
            depth_groups[dlabel] = [lower_lim, upper_lim]
    
    # Plot distributions
    fthresh_hva = plot_mnv_histograms(lmnv, lmnv[deep], mnv_mask, deep_pct, key, k_i, NROIs, fsize, pad=0.0, figsize=(15,3))
    
    # Plot depth maps
    dmap_hva = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0,100], fsize, fname = 'dmap', pad=0.0)
        
    #plot thresholded map
    dmap_thresh_hva = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0, 100], fsize, fname='dmap_thresh', mask=mnv_mask, pad=0.0)
        
    # Plot voxel loss at each depth after masking
    #fdepth_hist_hva = plot_depth_voxel_loss(z, mnv_mask, nDepths, NROIs, key, k_i, fsize)
    
    # Reset depth groups if necessary
    if set_indiv_depth:
        for d_i, dlabel in enumerate(depth_labels):
            depth_groups[dlabel] = None
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(mnv_mask),np.size(mnv),key))
    
if savefigs:
    fthresh_hva.savefig(os.path.join(figDir,'mnv_hist_hva.%s' %figType),format=figType)
    dmap_hva.savefig(os.path.join(figDir,'mnv_depth_map_hva.%s' %figType),format=figType)
    dmap_thresh_hva.savefig(os.path.join(figDir,'mnv_depth_map_thresh_hva.%s' %figType),format=figType)
    #fdepth_hist_hva.savefig(os.path.join(figDir,'mnv_depth_hist_hva.%s' %figType),format=figType)

#%% Compare thresholds between subjects
    
#for HVAs
f = lmnv_violin_plots(hva_data)

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin_hva.%s' %(fig_format)))
    
#%% Apply full model p-val mask if desired

use_fullmodel_mask = True
pthresh_fullmodel = 0.01
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
plot_centroids(all_data, masks_V1, gPPIDetails_V1tgt, roiRad, nDepths, xlim=[-10,10])
        
if savefigs:
    for l in gPPIDetails_V1tgt['labels']:
        plt.figure(l)
        plt.savefig(os.path.join(figDir,'centroids_%s.%s' %(l,figType)),format=figType)

Nsubj = len(hva_data.keys())

#plot centroids for each condition and ROI
plot_centroids(hva_data, masks_hva, gPPIDetails_V23tgt, roiRad, nDepths, xlim=[-10,10], radParam=None)
        
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
keep_diffs = np.zeros((NROIs,len(diffDetails_V1tgt['statIDs'].keys()),nDepths))
for iR, roiID in enumerate(all_data.keys()):
    for iDiff, diff in enumerate(diffDetails_V1tgt['statIDs'].keys()):
        keep_diffs[iR,iDiff,:] = diffProfiles_V1tgt[diff]['avg'][iR]
keep_diffs_hva = np.zeros((NROIs,len(diffDetails_V23tgt['statIDs'].keys()),nDepths))
for iR, roiID in enumerate(all_data.keys()):
    for iDiff, diff in enumerate(diffDetails_V23tgt['statIDs'].keys()):
        keep_diffs_hva[iR,iDiff,:] = diffProfiles_V23tgt[diff]['avg'][iR]

#define point spread function
p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
Nbins = nDepths #number of bins to use in this analysis

normalize_psf = False #True if you want to normalize the psf by the deepest layer  

decon_rois = depth_deconv(keep_rois,p2t_model,Nbins_model,Nbins,normalize_psf)
decon_rois_hva = depth_deconv(keep_rois_hva,p2t_model,Nbins_model,Nbins,normalize_psf)
decon_diffs = depth_deconv(keep_diffs,p2t_model,Nbins_model,Nbins,normalize_psf)
decon_diffs_hva = depth_deconv(keep_diffs_hva,p2t_model,Nbins_model,Nbins,normalize_psf)

#now put back in dictionary
for iStat, stat in enumerate(gPPIDetails_V1tgt['labels']):
    depthProfiles_V1tgt[stat]['avg_decon'] = np.squeeze(np.array(decon_rois)[:,iStat,:])
for iStat, stat in enumerate(gPPIDetails_V23tgt['labels']):
    depthProfiles_V23tgt[stat]['avg_decon'] = np.squeeze(np.array(decon_rois_hva)[:,iStat,:])
    
# Put together into single dictionary
depthProfiles = {}
depthProfiles.update(depthProfiles_V1tgt)
depthProfiles.update(depthProfiles_V23tgt)

# Diff Profiles
for iDiff, diff in enumerate(diffDetails_V1tgt['statIDs'].keys()):
    diffProfiles_V1tgt[diff]['avg_decon'] = np.squeeze(np.array(decon_diffs)[:,iDiff,:])
for iDiff, diff in enumerate(diffDetails_V23tgt['statIDs'].keys()):
    diffProfiles_V23tgt[diff]['avg_decon'] = np.squeeze(np.array(decon_diffs_hva)[:,iDiff,:])
    
# Put together into single dictionary
diffProfiles = {}
diffProfiles.update(diffProfiles_V1tgt)
diffProfiles.update(diffProfiles_V23tgt)

#%% run the radial profiles analysis by binning the data across radial space
nRad = 4 #how many radius bins
maxRad = 4
prop_err = False
showSig = True
binSize = maxRad/nRad
radBins = np.linspace(0,maxRad,nRad+1)
radBinCtrs = radBins[:-1] + binSize/2
depth_labels = ['deep', 'middle', 'superficial']
depthBoundaries = np.array([[0,0.333],[0.333,0.666],[0.666,1]])

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


    
#%% Run Stats
# In this section I'll correct the p-values for all of the depth bins in each
# context modulation condition. Plus, I will test all of the bins against all of 
# the other bins. I'll need to do an ANOVA for this part. In the end, I will 
# need to correct for multiple comparisons taking into account all the 
# comparisons made. This will be 
#     N_depths x N_conditions + (N_depths choose 2) x N_conditions comparisons

# I have already collected the p-values when running compute_depth_profiles.
# This tested each depth against a null hypothesis that the distribution has a
# mean of zero.

prop_err = False # do error propagation?
use_decon = False
useSI = False

V1Stats = gPPIDetails_V1tgt['labels']
V1Colors = gPPIDetails_V1tgt['colors']
V1Diffs = list(diffDetails_V1tgt['statIDs'].keys())
V23Stats = gPPIDetails_V23tgt['labels']
V23Colors = gPPIDetails_V23tgt['colors']
V23Diffs = list(diffDetails_V23tgt['statIDs'].keys())

all_pvals = np.array([]) #initialize a 1D array that will contain all pvals
pvals_lookup = {} #initialize a dictionary that will keep track of which p-vals correspond to which comparison
iter_var = 0

[avgV1Profiles, avgV1Diffs] = compute_avg_depth_profile(depthProfiles_V1tgt,gPPIDetails_V1tgt,diffDetails_V1tgt['statIDs'],V1Stats,V1Diffs,use_decon,prop_err,useSI)  
[avgV23Profiles, avgV23Diffs] = compute_avg_depth_profile(depthProfiles_V23tgt,gPPIDetails_V23tgt,diffDetails_V23tgt['statIDs'],V23Stats,V23Diffs,use_decon,prop_err,useSI)  

# 1-sample t-tests
for c_i, c in enumerate(avgV1Diffs.keys()):
    all_pvals = np.append(all_pvals,avgV1Diffs[c]['p-vals'].pvalue)
    n_pvals = len(avgV1Diffs[c]['p-vals'].pvalue)
    pvals_lookup[c] = [iter_var,iter_var+n_pvals]
    iter_var += n_pvals
for c_i, c in enumerate(avgV23Diffs.keys()):
    all_pvals = np.append(all_pvals,avgV23Diffs[c]['p-vals'].pvalue)
    pvals_lookup[c] = [iter_var,iter_var+n_pvals]
    iter_var += n_pvals
# for c_i, c in enumerate(avgOtherDiffs.keys()):
#     all_pvals = np.append(all_pvals,avgOtherDiffs[c]['p-vals'].pvalue)
#     pvals_lookup[c] = [cond_iter*nD,cond_iter*nD+nD]
#     iter_var += 1
    
# 2-way ANOVA
diffProfiles_list = {}
for cond in diffProfiles.keys():
    if use_decon:
        diffProfiles_list[cond] = diffProfiles[cond]['avg_decon']
    else:
        diffProfiles_list[cond] = diffProfiles[cond]['avg']
    
def run_two_way_anova(data):
    """
    This function runs a 2-way ANOVA on a dataset provided as a JSON file.
    
    Parameters:
    - json_file_path: A string path to the JSON file containing the dataset.
    
    Returns:
    - A pandas DataFrame containing the ANOVA table with the main effects and interaction.
    """
    
    # Transform the JSON data into a flat DataFrame
    condition_list = []
    depth_bin_list = []
    value_list = []

    # Loop over each condition and its samples
    for condition, samples in data.items():
        for sample_id, depths in enumerate(samples):
            for depth_bin, value in enumerate(depths):
                # Append data to lists
                condition_list.append(condition)
                depth_bin_list.append(depth_bin)
                value_list.append(value)

    # Create a dataframe with the lists
    df = pd.DataFrame({
        'Condition': condition_list,
        'Depth_Bin': depth_bin_list,
        'Value': value_list
    })
    
    # Define the model formula for 2-way ANOVA
    model_formula = 'Value ~ C(Condition) + Depth_Bin + C(Condition):Depth_Bin'

    # Fit the model
    model = ols(model_formula, data=df).fit()

    # Perform ANOVA and get the table
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    return anova_table

anova_results = run_two_way_anova(diffProfiles_list)
print("ANOVA Results: ")
print(anova_results)

# To perform a two-sample t-test between each of the depth bins for each condition, we will:
# 1. Loop through each condition.
# 2. For each condition, perform a t-test between each unique pair of depth bins.
# 3. Collect and return the p-values for each test.

# Compare different conditions at each depth (2-sample paired t-tests)
# test_list = ['fgm','odss'] #diffProfiles_list.keys():
# for condition1 in test_list: 
#     test_list.remove(condition1) #remove condition1 from test list and compare all other conditions to condition1
#     if len(test_list) > 0: #if there are still conditions to test against
#         for condition2 in test_list:
#             # Get the samples for the condition
#             samples1 = diffProfiles_list[condition1]
#             samples2 = diffProfiles_list[condition2]
#             # Perform a t-test between each unique pair of depth bins
#             pvals_lookup[roi_type][condition1+'vs'+condition2] = [iter_var,iter_var+n_pvals] 
#             for i in range(np.shape(samples1)[1]):
#                 # Perform a paired two-sample t-test
#                 t_stat, p_val = stats.ttest_rel(samples1[:,i], samples2[:,i])
#                 # Add to big list
#                 all_pvals = np.append(all_pvals,p_val)
#             iter_var += n_pvals

# Dictionary to hold the p-values
p_values = {}

# # Loop through each condition
# pval_i = len(all_pvals)
# for condition in diffProfiles_list.keys():
#     p_values[condition] = {}
#     # Get the samples for the condition
#     samples = diffProfiles_list[condition]
#     # Perform a t-test between each unique pair of depth bins
#     pvals_lookup[condition+'_2samp'] = {}
#     for i in range(len(samples[0])):
#         for j in range(i+1, len(samples[0])):
#             depth_bin_i_samples = [sample[i] for sample in samples]
#             depth_bin_j_samples = [sample[j] for sample in samples]
#             # Perform the t-test
#             t_stat, p_val = stats.ttest_ind(depth_bin_i_samples, depth_bin_j_samples, equal_var=False)
#             # Store the p-value
#             p_values[condition][(i, j)] = p_val
#             # Add to big list
#             all_pvals = np.append(all_pvals,p_val)
#             pvals_lookup[condition+'_2samp'][(i,j)] = pval_i
#             pval_i += 1
            
# Now do a big multiple-comparisons correction
all_pvals_corrected = multipletests(all_pvals,method=statCorrType)[1]
all_pvals_dict = {}

# Repackage all of the corrected p-values into a new dictionary
for key in pvals_lookup:
    if isinstance(pvals_lookup[key],dict):
        all_pvals_dict[key] = {}
        for combo in pvals_lookup[key]:
            all_pvals_dict[key][combo] = all_pvals_corrected[pvals_lookup[key][combo]]
    else:
        all_pvals_dict[key] = all_pvals_corrected[pvals_lookup[key][0]:pvals_lookup[key][1]]
    
#%% now make some average plots

def two_depth_profiles(avgProfiles1, avgProfiles2, Stats1, Stats2, Colors1, Colors2, title1=None, title2=None, ylim=[-0.02,1.02], xlim=[-1,1], dx=1.0, dy=0.7, Ntext=[3,0.05], lcolor='k', fcolor='w', fsize=14):
    # Wraps all the functions to make a standard depth profile plot into one function
    
    # Plot V1 average profiles
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)
        
    fig.clf()
        
    p1 = fig.add_axes([.15, .2, .25, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
    plot_avg_depth_profile(p1,avgProfiles1,Stats1,Colors1,ylim,xlim,dx,dy,Ntext,lcolor,fsize)
    plt.title(title1)
    
    p2 = fig.add_axes([.6, .2, .25, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    plot_avg_depth_profile(p2,avgProfiles2,Stats2,Colors2,ylim,xlim,dx,dy,Ntext,lcolor,fsize)
    plt.title(title2)
    
    return fig

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
fig = two_depth_profiles(avgV1Profiles, avgV23Profiles, V1Stats, V23Stats, V1Colors, V23Colors, title1 = 'V1 tgt', title2 = 'V23 tgt', xlim = [-4.0,4.0])

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
fig = two_depth_profiles(avgV1Profiles, avgV23Profiles, 
                         ['V23_superficial_deveined','V23_middle_deveined','V23_deep_deveined'], 
                         ['V1_tgt_superficial_deveined','V1_tgt_middle_deveined','V1_tgt_deep_deveined'], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],
                         title1 = 'V23 -> V1', title2 = 'V1 -> V23',
                         xlim = [-4.0,4.0])

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_combined.%s' %figType),format=figType)
    
# Plot V1-V23 iso0
fig = two_depth_profiles(avgV1Profiles, avgV23Profiles, 
                         ['V23_superficial_deveined_iso0','V23_middle_deveined_iso0','V23_deep_deveined_iso0'], 
                         ['V1_tgt_superficial_deveined_iso0','V1_tgt_middle_deveined_iso0','V1_tgt_deep_deveined_iso0'], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],
                         title1 = 'V23 -> V1 iso0', title2 = 'V1 -> V23 iso0',
                         xlim = [-4.0,4.0])

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_iso0_combined.%s' %figType),format=figType)
    
# Plot V1-V23 iso90
fig = two_depth_profiles(avgV1Profiles, avgV23Profiles, 
                         ['V23_superficial_deveined_iso90','V23_middle_deveined_iso90','V23_deep_deveined_iso90'], 
                         ['V1_tgt_superficial_deveined_iso90','V1_tgt_middle_deveined_iso90','V1_tgt_deep_deveined_iso90'], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],
                         title1 = 'V23 -> V1 iso90', title2 = 'V1 -> V23 iso90',
                         xlim = [-4.0,4.0])

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_iso90_combined.%s' %figType),format=figType)
    
# Plot V1-V23 orth
fig = two_depth_profiles(avgV1Profiles, avgV23Profiles, 
                         ['V23_superficial_deveined_orth','V23_middle_deveined_orth','V23_deep_deveined_orth'], 
                         ['V1_tgt_superficial_deveined_orth','V1_tgt_middle_deveined_orth','V1_tgt_deep_deveined_orth'], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],
                         title1 = 'V23 -> V1 orth', title2 = 'V1 -> V23 orth',
                         xlim = [-4.0,4.0])

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_orth_combined.%s' %figType),format=figType)
    
    
# Plot V1-V23 sur
fig = two_depth_profiles(avgV1Profiles, avgV23Profiles, 
                         ['V23_superficial_deveined_sur','V23_middle_deveined_sur','V23_deep_deveined_sur'], 
                         ['V1_tgt_superficial_deveined_sur','V1_tgt_middle_deveined_sur','V1_tgt_deep_deveined_sur'], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]], 
                         [[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0]],
                         title1 = 'V23 -> V1 sur', title2 = 'V1 -> V23 sur',
                         xlim = [-4.0,4.0])

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_gPPI_V1tgt-V2tgt_sur_combined.%s' %figType),format=figType)
    
#%% For each condition, create nDepthsxnDepths grid

def plot_gPPI_mat(data_mat,p_mat,ROIx,ROIy,seed=None,target=None,Nvox=None,cbar_lims = [-2.0,2.0],title_add='',fig=None,ax=None,fontsize=12,invert_yaxis=True,cbar=True):
    """ Plots the gPPI matrix """
    
    # Set seed
    if not seed:
        seed = ROIx
    if not target:
        target = ROIy
    
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
            elif p_mat[i][j] < 0.01 and p_mat[i][j] >= 0.001:
                text = ax.text(j, i, f"{data_mat[i, j]:.2f}%**\np={p_mat[i, j]:.3f}", ha='center', va='center', color='black', fontsize=0.6*fontsize, weight='bold')
            elif p_mat[i][j] < 0.001:
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
    ax.set_xlabel(ROIx,fontsize=0.8*fontsize)
    ax.set_ylabel(ROIy,fontsize=0.8*fontsize)

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
    all_pvalues = np.concatenate((np.array([avgV1Profiles[c]['p-vals'].pvalue for c in V1_condition_list]).flatten(),np.array([avgV23Profiles[c]['p-vals'].pvalue for c in V23_condition_list]).flatten())) #get all p-values for all measurements in one list
    condition_list = np.concatenate((V1_condition_list,V23_condition_list))
    corrected_list = multipletests(all_pvalues.flatten(),method=statCorrType)[1]
    for c_i, c in enumerate(condition_list):
        if 'V1' in c:
            all_corrected_pvalues['V1'][c] = corrected_list[c_i*nDepths:c_i*nDepths+nDepths]
        if 'V23' in c:
            all_corrected_pvalues['V23'][c] = corrected_list[c_i*nDepths:c_i*nDepths+nDepths]

if plot_baseline:
    # V2/3 -> V1
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_deep_deveined','V23_middle_deveined','V23_superficial_deveined']
    if use_all_meas:
        for c_i, c in enumerate(condition_list):
            data_mat[:,c_i] = avgV1Profiles[c]['avg']
            p_mat[:,c_i] = all_corrected_pvalues['V23'][c]
    else:
        for c_i, c in enumerate(condition_list):
            data_mat[:,c_i] = avgV1Profiles[c]['avg']
            corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
            p_mat[:,c_i] = corrected_pvalues
        
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',seed='V2/3',target='V1',fig=fig,ax=ax1,cbar=False)

    # if savefigs:
    #     fig.savefig(os.path.join(figDir,'V23->V1_mat.%s' %figType),format=figType)
    
    # V1 -> V2/3
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined','V1_tgt_middle_deveined','V1_tgt_superficial_deveined']
    if use_all_meas:
        for c_i, c in enumerate(condition_list):
            data_mat[c_i,:] = avgV23Profiles[c]['avg']
            p_mat[c_i,:] = all_corrected_pvalues['V1'][c]
    else:
        for c_i, c in enumerate(condition_list):
            data_mat[c_i,:] = avgV23Profiles[c]['avg']
            corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
            p_mat[c_i,:] = corrected_pvalues
            
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',seed='V1',target='V2/3',fig=fig,ax=ax6,cbar=False)
    
    # if savefigs:
    #     fig.savefig(os.path.join(figDir,'V1->V23_mat.%s' %figType),format=figType)

# V2/3 -> V1 iso0
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_iso0','V23_middle_deveined_iso0','V23_superficial_deveined_iso0']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Profiles[c]['avg']
        p_mat[:,c_i] = all_corrected_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso0','V1 iso0',seed='V2/3 iso0',target='V1 iso0',fig=fig,ax=ax2,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_iso0_mat'))

# V1 -> V2/3 iso0
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_deep_deveined_iso0','V1_tgt_middle_deveined_iso0','V1_tgt_superficial_deveined_iso0']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso0','V1 iso0',seed='V1 iso0',target='V2/3 iso0',fig=fig,ax=ax7,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_iso0_mat.%s' %figType),format=figType)

# V2/3 -> V1 iso90
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_iso90','V23_middle_deveined_iso90','V23_superficial_deveined_iso90']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Profiles[c]['avg']
        p_mat[:,c_i] = all_corrected_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso90','V1 iso90',seed='V2/3 iso90',target='V1 iso90',fig=fig,ax=ax3,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_iso90_mat'))

# V1 -> V2/3 iso90
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_deep_deveined_iso90','V1_tgt_middle_deveined_iso90','V1_tgt_superficial_deveined_iso90']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso90','V1 iso90',seed='V1 iso90',target='V2/3 iso90',fig=fig,ax=ax8,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_iso90_mat.%s' %figType),format=figType)

# V2/3 -> V1 orth
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_orth','V23_middle_deveined_orth','V23_superficial_deveined_orth']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Profiles[c]['avg']
        p_mat[:,c_i] = all_corrected_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 orth','V1 orth',seed='V2/3 orth',target='V1 orth',fig=fig,ax=ax4,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_orth_mat'))

# V1 -> V2/3 orth
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_deep_deveined_orth','V1_tgt_middle_deveined_orth','V1_tgt_superficial_deveined_orth']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        p_mat[:,c_i] = all_corrected_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 orth','V1 orth',seed='V1 orth',target='V2/3 orth',fig=fig,ax=ax9,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_orth_mat.%s' %figType),format=figType)

# V2/3 -> V1 sur
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_sur','V23_middle_deveined_sur','V23_superficial_deveined_sur']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Profiles[c]['avg']
        p_mat[:,c_i] = all_corrected_pvalues['V23'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV1Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 sur','V1 sur',seed='V2/3 sur',target='V1 sur',fig=fig,ax=ax5,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_sur_mat'))

# V1 -> V2/3 sur
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_deep_deveined_sur','V1_tgt_middle_deveined_sur','V1_tgt_superficial_deveined_sur']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        p_mat[c_i,:] = all_corrected_pvalues['V1'][c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Profiles[c]['avg']
        corrected_pvalues = multipletests(avgV23Profiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
heatmap, fig, ax = plot_gPPI_mat(data_mat,p_mat,'V2/3 sur','V1 sur',seed='V1 sur',target='V2/3 sur',fig=fig,ax=ax10,cbar=False)

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

cbar_lims = [-4,4]

#create plot
use_all_meas = True #True if you would like to use all measurements in your correction for multiple comparisons


fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4)
fig.set_figwidth(14)
fig.set_figheight(6)
fontsize=12

#Get all corrected p-values
# all_corrected_diff_pvalues = {"V1":{},"V23":{}}
# if use_all_meas:
#     V1_condition_list = list(diffDetails_V1tgt['statIDs'].keys())
#     V23_condition_list = list(diffDetails_V23tgt['statIDs'].keys())
#     all_diff_pvalues = np.concatenate(([avgV1Diffs[c]['p-vals'].pvalue for c in V1_condition_list],[avgV23Diffs[c]['p-vals'].pvalue for c in V23_condition_list])) #get all p-values for all measurements in one list
#     condition_list = np.concatenate((V1_condition_list,V23_condition_list))
#     corrected_list = multipletests(all_diff_pvalues.flatten(),method=statCorrType)[1]
#     for c_i, c in enumerate(condition_list):
#         if 'V1' in c:
#             all_corrected_diff_pvalues['V1'][c] = corrected_list[c_i:c_i+nDepths]
#         if 'V23' in c:
#             all_corrected_diff_pvalues['V23'][c] = corrected_list[c_i:c_i+nDepths]

# V2/3 -> V1 iso0 - sur
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_iso-sur','V23_middle_deveined_iso-sur','V23_superficial_deveined_iso-sur']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Diffs[c]['avg']
        p_mat[:,c_i] = all_pvals_dict[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Diffs[c]['avg']
        top = avgV1Diffs[c]['avg'] + avgV1Diffs[c]['stdev']/np.sqrt(np.shape(avgV1Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV1Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso-sur','V1 iso-sur',seed='V2/3 iso-sur',target='V1 iso-sur',fig=fig,ax=ax1,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_iso-sur_mat.%s' %figType),format=figType)

# V1 -> V2/3 iso0 - sur
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_deep_deveined_iso-sur','V1_tgt_middle_deveined_iso-sur','V1_tgt_superficial_deveined_iso-sur']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        p_mat[c_i,:] = all_pvals_dict[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        top = avgV23Diffs[c]['avg'] + avgV23Diffs[c]['stdev']/np.sqrt(np.shape(avgV23Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV23Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso-sur','V1 iso-sur',seed='V1 iso-sur',target='V2/3 iso-sur',fig=fig,ax=ax5,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_iso-sur_mat.%s' %figType),format=figType)

# V2/3 -> V1 odss
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_odss','V23_middle_deveined_odss','V23_superficial_deveined_odss']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Diffs[c]['avg']
        p_mat[:,c_i] = all_pvals_dict[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Diffs[c]['avg']
        top = avgV1Diffs[c]['avg'] + avgV1Diffs[c]['stdev']/np.sqrt(np.shape(avgV1Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV1Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 ODSS','V1 ODSS',seed='V2/3 ODSS',target='V1 ODSS',fig=fig,ax=ax2,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_odss_mat.%s' %figType),format=figType)

# V1 -> V2/3 odss
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_deep_deveined_odss','V1_tgt_middle_deveined_odss','V1_tgt_superficial_deveined_odss']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        p_mat[c_i,:] = all_pvals_dict[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        top = avgV23Diffs[c]['avg'] + avgV23Diffs[c]['stdev']/np.sqrt(np.shape(avgV23Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV23Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 ODSS','V1 ODSS',seed='V1 ODSS',target='V2/3 ODSS',fig=fig,ax=ax6,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_odss_mat.%s' %figType),format=figType)

# V2/3 -> V1 fgm
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_fgm','V23_middle_deveined_fgm','V23_superficial_deveined_fgm']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Diffs[c]['avg']
        p_mat[:,c_i] = all_pvals_dict[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Diffs[c]['avg']
        top = avgV1Diffs[c]['avg'] + avgV1Diffs[c]['stdev']/np.sqrt(np.shape(avgV1Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV1Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 FGM','V1 FGM',seed='V2/3 FGM',target='V1 FGM',fig=fig,ax=ax3,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_fgm_mat.%s' %figType),format=figType)

# V1 -> V2/3 fgm
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_deep_deveined_fgm','V1_tgt_middle_deveined_fgm','V1_tgt_superficial_deveined_fgm']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        p_mat[c_i,:] = all_pvals_dict[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        top = avgV23Diffs[c]['avg'] + avgV23Diffs[c]['stdev']/np.sqrt(np.shape(avgV23Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV23Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 FGM','V1 FGM',seed='V1 FGM',target='V2/3 FGM',fig=fig,ax=ax7,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V1->V23_fgm_mat.%s' %figType),format=figType)

# V2/3 -> V1 dsi
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_dsi','V23_middle_deveined_dsi','V23_superficial_deveined_dsi']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Diffs[c]['avg']
        p_mat[:,c_i] = all_pvals_dict[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgV1Diffs[c]['avg']
        top = avgV1Diffs[c]['avg'] + avgV1Diffs[c]['stdev']/np.sqrt(np.shape(avgV1Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV1Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 $\Delta$SI','V1 $\Delta$SI',seed='V2/3 $\Delta$SI',target='V1 $\Delta$SI',fig=fig,ax=ax4,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_dsi_mat.%s' %figType),format=figType)

# V1 -> V2/3 dsi
data_mat = np.zeros([nDepths,nDepths])
p_mat = np.zeros([nDepths,nDepths])
p_thresh = 0.05
condition_list = ['V1_tgt_deep_deveined_dsi','V1_tgt_middle_deveined_dsi','V1_tgt_superficial_deveined_dsi']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        p_mat[c_i,:] = all_pvals_dict[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = avgV23Diffs[c]['avg']
        top = avgV23Diffs[c]['avg'] + avgV23Diffs[c]['stdev']/np.sqrt(np.shape(avgV23Diffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgV23Diffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
    
heatmap, fig, ax = plot_gPPI_mat(data_mat,p_mat,'V2/3 $\Delta$SI','V1 $\Delta$SI',seed='V1 $\Delta$SI',target='V2/3 $\Delta$SI',fig=fig,ax=ax8,cbar=False,cbar_lims=cbar_lims)

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
    
#%% Compute differences and put them back in all_data
diffDetails = {'statIDs': 
               {'odss_gPPI_superficial_V23':['V23_superficial_deveined_orth', 'V23_superficial_deveined_iso90'],
                'odss_gPPI_middle_V23':['V23_middle_deveined_orth','V23_middle_deveined_iso90'],
                'odss_gPPI_deep_V23':['V23_deep_deveined_orth','V23_deep_deveined_iso90'],
                'fgm_gPPI_superficial_V23':['V23_superficial_deveined_iso90','V23_superficial_deveined_iso0'],
                'fgm_gPPI_middle_V23':['V23_middle_deveined_iso90','V23_middle_deveined_iso0'],
                'fgm_gPPI_deep_V23':['V23_deep_deveined_iso90','V23_deep_deveined_iso0'],
                'iso-sur_gPPI_superficial_V23':['V23_superficial_deveined_iso0','V23_superficial_deveined_sur'],
                'iso-sur_gPPI_middle_V23':['V23_middle_deveined_iso0','V23_middle_deveined_sur'],
                'iso-sur_gPPI_deep_V23':['V23_deep_deveined_iso0','V23_deep_deveined_sur']
               }
               
              }

for diff in diffDetails['statIDs'].keys():
    for iR, label in enumerate(all_data.keys()):
        stat1 = diffDetails['statIDs'][diff][0]
        stat2 = diffDetails['statIDs'][diff][1]
        all_data[label][diff] = all_data[label][stat1] - all_data[label][stat2]
      
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
    condition_list = ['V23_deep_deveined','V23_middle_deveined','V23_superficial_deveined']
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target; should be the same for all conditions
        
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="",fig=fig,ax=ax1,fontsize=fsize)
        
    # V1 -> V2/3 iso0 - sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined','V1_tgt_middle_deveined','V1_tgt_superficial_deveined']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="",fig=fig,ax=ax2,fontsize=fsize)

    # V2/3 -> V1 iso0
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_deep_deveined_iso0','V23_middle_deveined_iso0','V23_superficial_deveined_iso0']
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="iso0",fig=fig,ax=ax3,fontsize=fsize)
        
    # V1 -> V2/3 iso0
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined_iso0','V1_tgt_middle_deveined_iso0','V1_tgt_superficial_deveined_iso0']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="iso0",fig=fig,ax=ax4,fontsize=fsize)

    # V2/3 -> V1 iso90
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_deep_deveined_iso90','V23_middle_deveined_iso90','V23_superficial_deveined_iso90']
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="iso90",fig=fig,ax=ax5,fontsize=fsize)
        
    # V1 -> V2/3 iso90
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined_iso90','V1_tgt_middle_deveined_iso90','V1_tgt_superficial_deveined_iso90']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="iso90",fig=fig,ax=ax6,fontsize=fsize)
    
    # V2/3 -> V1 orth
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_deep_deveined_orth','V23_middle_deveined_orth','V23_superficial_deveined_orth']
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="orth",fig=fig,ax=ax7,fontsize=fsize)
        
    # V1 -> V2/3 orth
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined_orth','V1_tgt_middle_deveined_orth','V1_tgt_superficial_deveined_orth']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="orth",fig=fig,ax=ax8,fontsize=fsize)

    # V2/3 -> V1 sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_deep_deveined_sur','V23_middle_deveined_sur','V23_superficial_deveined_sur']
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = depthProfiles_V1tgt[c]['avg'][iR]
        top = depthProfiles_V1tgt[c]['avg'][iR] + depthProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="sur",fig=fig,ax=ax9,fontsize=fsize)
        
    # V1 -> V2/3 sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined_sur','V1_tgt_middle_deveined_sur','V1_tgt_superficial_deveined_sur']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = depthProfiles_V23tgt[c]['avg'][iR]
        top = depthProfiles_V23tgt[c]['avg'][iR] + depthProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(depthProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(depthProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': depthProfiles_V1tgt[list(depthProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': depthProfiles_V23tgt[list(depthProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-2.0,2.0],title_add="sur",fig=fig,ax=ax10,fontsize=fsize)

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
    condition_list = ['V23_deep_deveined_iso-sur','V23_middle_deveined_iso-sur','V23_superficial_deveined_iso-sur']
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = diffProfiles_V1tgt[c]['avg'][iR]
        top = diffProfiles_V1tgt[c]['avg'][iR] + diffProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    Nvox = {'seed': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target; should be the same for all conditions
        
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-5.0,5.0],title_add="iso-sur",fig=fig,ax=ax1,fontsize=fsize)
        
    # V1 -> V2/3 iso0 - sur
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined_iso-sur','V1_tgt_middle_deveined_iso-sur','V1_tgt_superficial_deveined_iso-sur']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V23tgt[c]['avg'][iR]
        top = diffProfiles_V23tgt[c]['avg'][iR] + diffProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-5.0,5.0],title_add="iso-sur",fig=fig,ax=ax2,fontsize=fsize)

    # V2/3 -> V1 odss
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_deep_deveined_odss','V23_middle_deveined_odss','V23_superficial_deveined_odss']
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = diffProfiles_V1tgt[c]['avg'][iR]
        top = diffProfiles_V1tgt[c]['avg'][iR] + diffProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-5.0,5.0],title_add="ODSS",fig=fig,ax=ax3,fontsize=fsize)
        
    # V1 -> V2/3 odss
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined_odss','V1_tgt_middle_deveined_odss','V1_tgt_superficial_deveined_odss']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V23tgt[c]['avg'][iR]
        top = diffProfiles_V23tgt[c]['avg'][iR] + diffProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-5.0,5.0],title_add="ODSS",fig=fig,ax=ax4,fontsize=fsize)

    # V2/3 -> V1 fgm
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_deep_deveined_fgm','V23_middle_deveined_fgm','V23_superficial_deveined_fgm']
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = diffProfiles_V1tgt[c]['avg'][iR]
        top = diffProfiles_V1tgt[c]['avg'][iR] + diffProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-5.0,5.0],title_add="FGM",fig=fig,ax=ax5,fontsize=fsize)
        
    # V1 -> V2/3 fgm
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined_fgm','V1_tgt_middle_deveined_fgm','V1_tgt_superficial_deveined_fgm']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V23tgt[c]['avg'][iR]
        top = diffProfiles_V23tgt[c]['avg'][iR] + diffProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-5.0,5.0],title_add="FGM",fig=fig,ax=ax6,fontsize=fsize)
    
    # V2/3 -> V1 dsi
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V23_deep_deveined_dsi','V23_middle_deveined_dsi','V23_superficial_deveined_dsi']
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = diffProfiles_V1tgt[c]['avg'][iR]
        top = diffProfiles_V1tgt[c]['avg'][iR] + diffProfiles_V1tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V1tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V1tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',Nvox=Nvox,cbar_lims=[-5.0,5.0],title_add="$\Delta$SI",fig=fig,ax=ax7,fontsize=fsize)
        
    # V1 -> V2/3 dsi
    data_mat = np.zeros([nDepths,nDepths])
    p_mat = np.zeros([nDepths,nDepths])
    p_thresh = 0.05
    condition_list = ['V1_tgt_deep_deveined_dsi','V1_tgt_middle_deveined_dsi','V1_tgt_superficial_deveined_dsi']
    for c_i, c in enumerate(condition_list):
        data_mat[c_i,:] = diffProfiles_V23tgt[c]['avg'][iR]
        top = diffProfiles_V23tgt[c]['avg'][iR] + diffProfiles_V23tgt[c]['stdev'][iR]/np.sqrt(np.shape(diffProfiles_V23tgt[c]['avg'][iR])[0])
        corrected_pvalues = multipletests(diffProfiles_V23tgt[c]['p-vals'][iR],method=statCorrType)[1]
        p_mat[c_i,:] = corrected_pvalues
        
    Nvox = {'seed': diffProfiles_V1tgt[list(diffProfiles_V1tgt.keys())[0]]['N'][iR], 'targ': diffProfiles_V23tgt[list(diffProfiles_V23tgt.keys())[0]]['N'][iR]} #show number of voxels in seed and target
    plot_gPPI_mat(data_mat,p_mat,'V1','V2/3',Nvox=Nvox,cbar_lims=[-5.0,5.0],title_add="$\Delta$SI",fig=fig,ax=ax8,fontsize=fsize)

    plt.suptitle(label)
    plt.tight_layout(pad=0.1,rect=(0,0,1,0.95))

    if savefigs:
        fig.savefig(os.path.join(figDir,'V23-V1_diffs_mat_%s.%s' %(label,figType)),format=figType)