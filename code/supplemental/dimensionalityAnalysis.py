#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:21:09 2024

@author: Joe


Dimensionality analysis: I saw a poster that showed that inter-areal
interactions between visual areas have lower dimensionality than intra-areal
interactions. Do we see this difference between the FGM and OTSS contrasts in
our high res fMRI data?
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
figDir = mainDir+'/figs/individual_ROIs/dimensionality/'
fig_format = 'svg'
statCorrType = 'fdr_bh' #'bonferroni'

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset

datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual', 'pnr???_??_???_??.csv'))
#datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_filled', 'pnr???_??_???_??.csv'))
#or exclude
exclude_initial = ['pnr143_V1_tgt_rh','pnr161_V1_tgt_lh','pnr161_V1_tgt_rh','pnr352_V1_tgt_lh','pnr352_V1_tgt_rh','pnr579_V1_tgt_lh','pnr668_V1_tgt_rh']
#exclude_initial = ['pnr352_V1_tgt_lh_rad10']
for e_i, excl in enumerate(exclude_initial):
    datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_roi_manual',excl+'.csv'))
    #datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_filled',excl+'.csv'))        
datasets.sort()

roiRad = 4 #2 #1.
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

show_rad = False
frad = plt.figure(figsize=(8,8))
floc = plt.figure(figsize=(8,8))
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    subjID = label[0:6]
    hemi = label[14:16]

    # redo the ellipse fitting, which is a bit of overkill, but it gets us 
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
                      zorder=100, alpha=1., edgecolor='cyan', facecolor='None')
    
    if show_rad:
        # show localizer data
        minx = np.min(df['x'].values)
        miny = np.min(df['y'].values)
        ax = frad.add_subplot(int(np.ceil(np.sqrt(len(datasets)))),int(np.ceil(np.sqrt(len(datasets)))),iR+1)
        
        # Plot the radius determined by the normalized uv coordinates (this should be in SD of a 2D Gaussian fitted to the loc data)
        cmap = plt.cm.get_cmap('viridis')
        pcm = ax.scatter(df['x'],df['y'],c=df['scale_xy_dist'],s=2,cmap=cmap)
        cbar = plt.colorbar(pcm,ax=ax)
        ax.add_patch(ellipse)
        ax.patch.set_facecolor('r')
        ax.set_title(label+" radius: SD<2 Nvox = %d" %(np.sum(df['scale_xy_dist']<2)),fontsize=6)
        ax.axis('off')
    
    else:
        # Plot the ctr-sur betas
        ax2 = floc.add_subplot(int(np.ceil(np.sqrt(len(datasets)))),int(np.ceil(np.sqrt(len(datasets)))),iR+1)
        cmap2 = plt.cm.get_cmap('plasma')
        pcm = ax2.scatter(df['x'],df['y'],c=df['ctr-sur_unwarp'],s=2,cmap=cmap2,vmin=-1.57,vmax=1.57)
        cbar2 = plt.colorbar(pcm,ax=ax2)
        ax2.add_patch(ellipse)
        ax2.patch.set_facecolor('r')
        ax2.set_title(subjID+"_"+hemi+"\n radius: SD<2 Nvox = %d" %(np.sum(df['scale_xy_dist']<2)),fontsize=6)
        ax2.axis('off')
    
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
    dmap = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [2,5], fsize, fname = 'dmap', pad=0.0)
        
    #plot thresholded map
    dmap_thresh = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [2,5], fsize, fname='dmap_thresh', mask=mnv_mask, pad=0.0)
        
    # Plot voxel loss at each depth after masking
    fdepth_hist = plot_depth_voxel_loss(z, mnv_mask, nDepths, NROIs, key, k_i, fsize)
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(mnv_mask),np.size(mnv),key))
    superficial_mask = (z>=depth_groups['superficial'][0])
    middle_mask = (z>=depth_groups['middle'][0])*(z<depth_groups['middle'][1])
    deep_mask = (z<depth_groups['deep'][1])
    print("\t %d/%d Voxels Survive for superficial %s" %(np.sum(mnv_mask*superficial_mask),np.sum(superficial_mask),key))
    print("\t %d/%d Voxels Survive for middle %s" %(np.sum(mnv_mask*middle_mask),np.sum(middle_mask),key))
    print("\t %d/%d Voxels Survive for deep %s" %(np.sum(mnv_mask*deep_mask),np.sum(deep_mask),key))
    
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

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin.%s' %(fig_format)))
    
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
        
#%% Dimensionality Reduction

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def standardize(data):
    mean = np.mean(data,axis=0)
    std = np.mean(data,axis=0)
    
    return ((data-mean)/std)

# Set up relevant conditions
statDetails = {'labels': ['sur', 'iso0', 'iso90', 'orth', 'ctr_unwarp', 'sur_unwarp', 'ctr-sur_unwarp'],
                'colors': [[.7, .7, .7], 'red', 'darkviolet', 'orange', 'gold', 'purple', 'coral']}
diffDetails = {}
diffDetails['statIDs'] = {'odss': ['orth','iso90'],
                          'fgm': ['iso90','iso0'],
                          'dsi': ['orth','iso0'],
                          'iso-sur': ['iso0','sur'],
                          'ctr-sur': ['ctr_unwarp','sur_unwarp'],
                          'iso90-sur': ['iso90','sur'],
                          'orth-sur': ['orth','sur']}
diffDetails['colors'] = {'odss': 'green',
                         'fgm': 'magenta',
                         'dsi': 'cyan',
                         'iso-sur': 'black',
                         'ctr-sur': 'coral',
                         'iso90-sur': 'darkmagenta',
                         'orth-sur': 'darkturquoise'}

# Compute contrasts
for subjID in all_data.keys():
    for diff in diffDetails['statIDs'].keys():
        all_data[subjID][diff] = all_data[subjID][diffDetails['statIDs'][diff][0]] - all_data[subjID][diffDetails['statIDs'][diff][1]]

taskDiffs = ['odss','fgm','ctr-sur']
for subjID in all_data.keys():
    fig = plt.figure(figsize=(10, 8))
    for c_i, cond in enumerate(taskDiffs):
        mask = (all_data[subjID]['no_vein'] & all_data[subjID]['sig'] & (all_data[subjID]['scale_xy_dist'] < roiRad))
        responses = all_data[subjID][cond][mask].values
        responses = np.reshape(responses,(len(responses),1))
        responses_standardized = standardize(responses)
        cov = (1/(len(responses)-1))*np.dot(responses,responses.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        sorted_inds = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_inds]
        eigenvectors = eigenvectors[sorted_inds]
        norm_eigenvalues = eigenvalues/np.sum(eigenvalues)
        
        ax1 = fig.add_subplot(2, len(taskDiffs), c_i+1)
        vLim = np.std(cov)
        ax1.imshow(cov)
        ax1.set_title(subjID+" "+cond+" covariance")
        ax1.set_xlabel("Voxel ID")
        ax1.set_ylabel("Voxel ID")
        cax = ax1.imshow(cov, cmap='viridis', vmin=-vLim,vmax=vLim)
        fig.colorbar(cax, ax=ax1, orientation='vertical')
        
        if c_i == 0:
            ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(np.arange(2,len(norm_eigenvalues)+1),norm_eigenvalues[1:],color=diffDetails['colors'][cond],label=cond)
        ax2.set_title(subjID+" eigenspetrum")
        ax2.set_xlabel("Rank")
        ax2.set_ylabel("Eigenvalue")
        ax2.set_xlim([0,100])
        ax2.set_ylim([0,None])
    ax2.legend()
        
    fig.tight_layout()
        
    if savefigs:
        fig.savefig(os.path.join(figDir,'scree_%s.%s' %(subjID,fig_format)))
