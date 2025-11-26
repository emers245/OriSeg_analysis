#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:41:56 2024

@author: joe

This code combines work from Cheryl and Joe for creating laminar and surface
profiles from the OriSeg data. This code specifically focuses on the analysis 
of contextual conditions.

This version uses a mixed linear model to determine effects across and within
subjects/hemispheres/depth bins.

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
import statsmodels.formula.api as smf

#Import custom functions
from oriseg_funcs import *

plt.close('all')
    
# Set Plotting Params
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]
savefigs = True #True #if true save all figures
fig_format = 'svg'

# Set Paths
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '.'
figDir = mainDir+'/figs/mixedLM/'

# Set Multiple Comparisons Correction
statCorrType = 'fdr_bh' #'bonferroni'

#%% Import Datasets and Check Stria of Gennari

mainDir = '.'
datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual', 'pnr???_??_???_??.csv'))
#datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_filled', 'pnr???_??_???_??.csv'))
#or exclude
exclude_initial = ['pnr143_V1_tgt_rh','pnr161_V1_tgt_lh','pnr161_V1_tgt_rh','pnr352_V1_tgt_lh','pnr352_V1_tgt_rh','pnr579_V1_tgt_lh','pnr668_V1_tgt_rh']
#exclude_initial = ['pnr352_V1_tgt_lh_rad10']
for e_i, excl in enumerate(exclude_initial):
    datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_roi_manual',excl+'.csv'))
    #datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_filled',excl+'.csv'))        
datasets.sort()

# Set ROI radius: This is the radius (in sigma) that marks the boundary between
#     target- and surround-selective regions.
roiRad = 2 #1.

# Organize data into dictionary
import pandas as pd
all_data = {}
for dataset in datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    all_data[f] = pd.read_csv(dataset, sep=',', index_col=False)
    
## Remove d = 0 voxels: Some voxels do not receive a depth label. We only 
#       analyze voxels with valid depth labels.
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    df = df.drop(df[df['d'] == 0].index)
    
    all_data[label] = df

# Check what the Stria profile looks like in each ROI
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
                      zorder=100, alpha=1., edgecolor='r', facecolor='None')
    
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
        pcm = ax2.scatter(df['x'],df['y'],c=df['ctr-sur_unwarp'],s=2,cmap=cmap2,vmin=-3,vmax=3)
        cbar2 = plt.colorbar(pcm,ax=ax2)
        ax2.add_patch(ellipse)
        ax2.patch.set_facecolor('r')
        ax2.set_title(subjID+"_"+hemi+"\n radius: SD<2 Nvox = %d" %(np.sum(df['scale_xy_dist']<2)),fontsize=6)
        ax2.axis('off')
    
if savefigs:
    frad.savefig(os.path.join(figDir,'xy_map_rad.%s' %(fig_format)))
    floc.savefig(os.path.join(figDir,'xy_map_loc.%s' %(fig_format)))
    
#%% Histograms of p-values
#       Check that enough voxels are responsive to the stimulus
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
#       Check coverage across depth for each ROI
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
    
#%% Devein
#       Use the deepest layer as a proxy for non-vein contaminated voxels
#       Then define the threshold based on this distribution

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

#%% Compare devein thresholds between subjects

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
    
#%% Apply full model p-val mask
#       Only analyze voxels with significant responses to the stimuli

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
        
#pick out ROIs where we're sure of localization
roi_dict = {}
for key in all_data.keys():
    df = all_data[key]
    roi_dict[key] = df['scale_xy_dist'] < roiRad
    all_data[key]['in_tgt'] = df['scale_xy_dist'] < roiRad
    
#%% Compute average depth profiles

#information about what conditions to look at
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
diffDetails['colors'] = ['green','magenta','cyan','black','coral','darkmagenta','darkturquoise']

#store masks for each ROI
masks = {}
masks['tgt'] = {roi:all_data[roi]['in_tgt']*all_data[roi]['sig']*all_data[roi]['no_vein'] for roi in all_data.keys()}
profile_method = 'bin' # bin or smooth
useSI = False #if True, use suppression index as contrast (cond1 - cond2)/(cond1 + cond2)
prop_err = False # do error propagation?
use_decon = True

depthProfiles = {}
diffProfiles = {}
#calculate average profiles
depthProfiles['tgt'] = compute_all_depth_profiles(all_data,statDetails,profile_method,nDepths,masks['tgt'],depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
diffProfiles['tgt'] = compute_diff_profiles(all_data,statDetails,diffDetails['statIDs'],profile_method,nDepths,useSI,masks['tgt'],depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')

#%% Depth Deconvolution

#reformat data to fit decon_rois specs
keep_rois = np.zeros((NROIs,len(statDetails['labels']),nDepths))
for iR, roiID in enumerate(all_data.keys()):
    for iStat, stat in enumerate(statDetails['labels']):
        keep_rois[iR,iStat,:] = depthProfiles['tgt'][stat]['avg'][iR]
        
keep_diffs = np.zeros((NROIs,len(diffDetails['statIDs'].keys()),nDepths))
for iR, roiID in enumerate(all_data.keys()):
    for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
        keep_diffs[iR,iDiff,:] = diffProfiles['tgt'][diff]['avg'][iR]

#define point spread function
p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
Nbins = nDepths #number of bins to use in this analysis

normalize_psf = False #True if you want to normalize the psf by the deepest layer  

decon_rois = depth_deconv(keep_rois,p2t_model,Nbins_model,Nbins,normalize_psf)
decon_diffs = depth_deconv(keep_diffs,p2t_model,Nbins_model,Nbins,normalize_psf)

#now put back in dictionary
for iStat, stat in enumerate(statDetails['labels']):
    depthProfiles['tgt'][stat]['avg_decon'] = np.squeeze(np.array(decon_rois)[:,iStat,:])
    
for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
    diffProfiles['tgt'][diff]['avg_decon'] = np.squeeze(np.array(decon_diffs)[:,iDiff,:])
        
#%% Subregions

centerRad = 1.7 #center of ROI
borderRad = [1.7,2.3] #range for border
surRad = [3.5] #[3, 3.5] #outside of this range will be considered the surround
ring_rads = {'ctr':centerRad,'border':borderRad,'sur':surRad}
rings = ['ctr','border','sur']
nDepths_rings = 3 #number of depth bins to use for rings analysis

for r_i, ring in enumerate(rings):
    frad = plt.figure(figsize=(6.5,8))
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
        
        # show localizer data
        minx = np.min(df['x'].values)
        miny = np.min(df['y'].values)
        ax = frad.add_subplot(int(np.ceil(len(datasets)/2)),2,iR+1)
        
        # Plot the radius determined by the normalized uv coordinates (this should be in SD of a 2D Gaussian fitted to the loc data)
        cmap = plt.cm.get_cmap('plasma')
        pcm = ax.scatter(df['x'],df['y'],c=df['ctr-sur'],s=1,cmap=cmap,vmin=-1,vmax=1)
        plt.colorbar(pcm,ax=ax)
        
        if ring == 'ctr':
            ellipse = Ellipse(com,
                          width=ring_rads[ring]*2*np.sqrt(a),
                          height=ring_rads[ring]*2*np.sqrt(b),
                          angle=180*theta/np.pi,
                          zorder=100, alpha=1, edgecolor='cyan', facecolor='None')
            ax.add_patch(ellipse)
        elif ring == 'sur':
            if len(ring_rads[ring]) == 1:
                ellipse = Ellipse(com,
                              width=ring_rads[ring][0]*2*np.sqrt(a),
                              height=ring_rads[ring][0]*2*np.sqrt(b),
                              angle=180*theta/np.pi,
                              zorder=100, alpha=1, edgecolor='cyan', facecolor='None')
                ax.add_patch(ellipse)
            elif len(ring_rads[ring]) > 1:
                ellipse1 = Ellipse(com,
                              width=ring_rads[ring][0]*2*np.sqrt(a),
                              height=ring_rads[ring][0]*2*np.sqrt(b),
                              angle=180*theta/np.pi,
                              zorder=100, alpha=1., edgecolor='cyan', facecolor='None')
                ellipse2 = Ellipse(com,
                              width=ring_rads[ring][1]*2*np.sqrt(a),
                              height=ring_rads[ring][1]*2*np.sqrt(b),
                              angle=180*theta/np.pi,
                              zorder=100, alpha=1., edgecolor='cyan', facecolor='None')
                ax.add_patch(ellipse1)
                ax.add_patch(ellipse2)
        elif ring == 'border':
            ellipse1 = Ellipse(com,
                          width=ring_rads[ring][0]*2*np.sqrt(a),
                          height=ring_rads[ring][0]*2*np.sqrt(b),
                          angle=180*theta/np.pi,
                          zorder=100, alpha=1., edgecolor='cyan', facecolor='None')
            ellipse2 = Ellipse(com,
                          width=ring_rads[ring][1]*2*np.sqrt(a),
                          height=ring_rads[ring][1]*2*np.sqrt(b),
                          angle=180*theta/np.pi,
                          zorder=100, alpha=1., edgecolor='cyan', facecolor='None')
            ax.add_patch(ellipse1)
            ax.add_patch(ellipse2)
        ax.patch.set_facecolor('r')
        ax.set_title(label+" radius: SD<2 Nvox = %d" %(np.sum(df['scale_xy_dist']<2)),fontsize=6)
        ax.set_aspect('equal')
        ax.axis('off')
        
    if savefigs:
        frad.savefig(os.path.join(figDir,'rings_graphics','rings_map_%s.%s' %(ring,fig_format)))

# Add columns to specify subregions
for k_i, key in enumerate(all_data.keys()):
    
    df = all_data[key]
    
    #subregion assignment
    df['in_ctr'] = df['scale_xy_dist'] <= centerRad
    df['in_border'] = (df['scale_xy_dist'] > borderRad[0]) & (df['scale_xy_dist'] <= borderRad[1])
    if len(surRad) > 1:
        df['in_sur'] = (df['scale_xy_dist'] > surRad[0]) & (df['scale_xy_dist'] <= surRad[1])
    else:
        df['in_sur'] = df['scale_xy_dist'] >= surRad[0]
    
    #depth reassignment
    d_bins_rings = bin_depths(df['d'].values,nDepths_rings,np.min(df['d'].values),np.max(df['d'].values),True)
    df['d_bin_rings'] = d_bins_rings
    
    all_data[key] = df
    
#%% Compare depth profiles at different radii

#pick out ROIs where we're sure of localization
masks['ctr'] = {}
masks['border'] = {}
masks['sur'] = {}
for key in all_data.keys():
    df = all_data[key]
    masks['ctr'][key] = df['in_ctr']*df['sig']*df['no_vein']
    masks['border'][key] = df['in_border']*df['sig']*df['no_vein']
    masks['sur'][key] = df['in_sur']*df['sig']*df['no_vein']
    
for r_i, ring in enumerate(rings):
    depthProfiles[ring] = compute_all_depth_profiles(all_data,statDetails,profile_method,nDepths_rings,masks[ring],depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
    diffProfiles[ring] = compute_diff_profiles(all_data,statDetails,diffDetails['statIDs'],profile_method,nDepths_rings,useSI,masks[ring],depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')

    
#%% Deconvolution for subregions

#reformat data to fit decon_rois specs
keep_rois = {}
keep_diffs = {}
for r_i,ring in enumerate(rings):
    
    keep_rois[ring] = np.zeros((NROIs,len(statDetails['labels']),nDepths_rings))
    for iR, roiID in enumerate(all_data.keys()):
        for iStat, stat in enumerate(statDetails['labels']):
            keep_rois[ring][iR,iStat,:] = depthProfiles[ring][stat]['avg'][iR]
            
    keep_diffs[ring] = np.zeros((NROIs,len(diffDetails['statIDs'].keys()),nDepths_rings))
    for iR, roiID in enumerate(all_data.keys()):
        for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
            keep_diffs[ring][iR,iDiff,:] = diffProfiles[ring][diff]['avg'][iR]
    
    #define point spread function
    p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
    Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
    Nbins = nDepths_rings #number of bins to use in this analysis
    
    normalize_psf = False #True if you want to normalize the psf by the deepest layer  
    
    decon_rois = depth_deconv(keep_rois[ring],p2t_model,Nbins_model,Nbins,normalize_psf)
    decon_diffs = depth_deconv(keep_diffs[ring],p2t_model,Nbins_model,Nbins,normalize_psf)
    
    #now put back in dictionary
    for iStat, stat in enumerate(statDetails['labels']):
        depthProfiles[ring][stat]['avg_decon'] = np.squeeze(np.array(decon_rois)[:,iStat,:])
        
    for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
        diffProfiles[ring][diff]['avg_decon'] = np.squeeze(np.array(decon_diffs)[:,iDiff,:])
      
#%% Restructure datafile to fit formatting for mixedlm
# New data should be organized with each category lised as a column in a 
# dataframe. The categories are nested:
#     subject
    # |
    # `-- hemisphere
    #        |
    #         `-- depth bin
    #                 |
    #                 `-- condition
    
# First, concatenate all of the subjects and ROIs and add labels for each
for k_i, key in enumerate(all_data.keys()):
    
    df = all_data[key]
    
    #subject ID
    subjID = key[key.find('pnr'):key.find('pnr')+6]
    
    #hemisphere
    if 'rh' in key:
        hemi = 'rh'
    elif 'lh' in key:
        hemi = 'lh'
    else:
        hemi = None
        print('No hemisphere label!!!')
        
    #depth bin
    d_bins = bin_depths(df['d'].values,nDepths,np.min(df['d'].values),np.max(df['d'].values),True)
    
    #contrasts
    for c_i, contrast in enumerate(diffDetails['statIDs'].keys()):
        df[contrast] = df[diffDetails['statIDs'][contrast][0]]-df[diffDetails['statIDs'][contrast][1]]
      
    #add new columns
    df['subjID'] = subjID
    df['hemi'] = hemi
    df['d_bin'] = d_bins
    
    # Now, to do statistics on the deconvolved profiles, recenter the voxel
    #   distributions in each depth bin on the deconvolved average. First calculate
    #   the difference in average activation before and after deconvolution, then
    #   apply difference to each voxel.

    # for target ROI
    for iStat, stat in enumerate(statDetails['labels']):
        depthProfiles['tgt'][stat]['decon_diff'] = depthProfiles['tgt'][stat]['avg'] - depthProfiles['tgt'][stat]['avg_decon']
        df[stat+'_decon'] = np.zeros(len(df))
        for d_i in range(nDepths):
            df.loc[df['d_bin']==d_i, stat+'_decon'] = df.loc[df['d_bin']==d_i, stat] - depthProfiles['tgt'][stat]['decon_diff'][k_i][d_i]
        df.loc[~masks['tgt'][key], stat+'_decon'] = None #voxels that were not included in the deconvolution should not have an associated deonvolved value
    for c_i, contrast in enumerate(diffDetails['statIDs'].keys()):
        df[contrast+'_decon'] = df[diffDetails['statIDs'][contrast][0]+'_decon']-df[diffDetails['statIDs'][contrast][1]+'_decon']
    
    # for subregions
    for r_i, r in enumerate(rings):
        for iStat, stat in enumerate(statDetails['labels']):
            depthProfiles[r][stat]['decon_diff'] = depthProfiles[r][stat]['avg'] - depthProfiles[r][stat]['avg_decon']
            df[r+'_'+stat+'_decon'] = np.zeros(len(df))
            for d_i in range(nDepths_rings):
                df.loc[df['d_bin_rings']==d_i, r+'_'+stat+'_decon'] = df.loc[df['d_bin_rings']==d_i, stat] - depthProfiles[r][stat]['decon_diff'][k_i][d_i]
            df.loc[~masks[r][key], stat+'_decon'] = None #voxels that were not included in the deconvolution should not have an associated deonvolved value
        for c_i, contrast in enumerate(diffDetails['statIDs'].keys()):
            df[r+'_'+contrast+'_decon'] = df[r+'_'+diffDetails['statIDs'][contrast][0]+'_decon']-df[r+'_'+diffDetails['statIDs'][contrast][1]+'_decon']
        
    
    #add to big dataframe
    if k_i == 0:
        all_df = df
    else:
        all_df = pd.concat((all_df,df)) 
        
# Mask out unwanted voxels
masked_df = all_df[all_df['in_tgt'] & all_df['sig'] & all_df['no_vein']]
        
# Now rearrange dataframe so that each condition has an index in a new column called condition
cond =  ['odss_decon','fgm_decon','iso-sur_decon','ctr-sur_decon'] #['odss','fgm','iso-sur','ctr-sur'] #['iso0','iso90','orth','sur','ctr','sur.1'] #list of conditions to test
masked_df = masked_df.reset_index()
df_mixedLM = pd.melt(masked_df,id_vars=['subjID','hemi','d_bin','d','index','scale_xy_dist', 'd_bin_rings', 'in_ctr', 'in_border', 'in_sur'],value_vars=cond,var_name='condition',value_name='beta')
        
# Example DataFrame
# df = pd.DataFrame({
#     'subjID': [...],
#     'hemi': [...],
#     'd_bin': [...],
#     'condition': [...],
#     'beta': [...]
# })

# Define the model formula
model_formula = 'beta ~ hemi + C(d_bin) + condition + C(d_bin):condition'
#'beta ~ hemi + d + condition + scale_xy_dist + d:condition + d:scale_xy_dist + scale_xy_dist:condition + d:scale_xy_dist:condition'
model = smf.mixedlm(model_formula, data=df_mixedLM, groups=df_mixedLM['subjID'], re_formula="1")
result = model.fit()

print(result.summary())

#%% Single sample tests
# For each depth bin in each condition contrast, construct an intercept-only
# mixed effects model that returns both the statistical significance of the 
# difference of the mean from zero and the fraction of the variance attributable
# to subject.

import re
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

profiles = {} #create a dictionary to save profiles in

def custom_warning_filter(message):
    # Define specific warning messages to ignore
    ignore_messages = [
        "The MLE may be on the boundary of the parameter space."
    ]
    return any(msg in str(message) for msg in ignore_messages)

def check_convergence(result):
    if result == None:
        return False
    else:
        summary_str = result.summary().as_text()
        m = re.search(r"Converged:\s*(Yes|No)", summary_str)
        isnan = np.isnan(result.pvalues).any()
        check = False
        if m:
            check = True
        
        return(check & ~isnan)

def try_model_fit(model, method=None, maxiter=100000):
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always', ConvergenceWarning)
            if method:
                result = model.fit(method=method, maxiter=maxiter)
            else:
                result = model.fit(maxiter=maxiter)
            
            # Filter warnings
            filtered_warnings = [w for w in caught_warnings if not custom_warning_filter(str(w.message))]
            if filtered_warnings:
                for w in filtered_warnings:
                    print(f"ConvergenceWarning: {w.message}")
                return None, False
        
        if result is None:
            return None, False
        return result, check_convergence(result)
    except np.linalg.LinAlgError:
        print("LinAlgError: Singular matrix.")
        return None, False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, False
    
def compute_icc(result):
    #compute intraclass correlation coefficient
    # icc = group var / (group var + residual var)
    
    icc = result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale)
    
    return icc
    

all_pvals = np.array([]) #initialize a 1D array that will hold all pvals
all_fsv = np.array([]) #initialize a 1D array that will hold all fraction subject variance estimates
pvals_lookup = {} #initialize a dictionary that will keep track of which p-vals correspond to which comparison

# single-condition mixed-effects models
cond =  ['odss_decon','fgm_decon','iso-sur_decon','ctr-sur_decon']
model_formula = 'beta ~ 1'
profiles['tgt'] = {} # set up profiles
for c_i, c in enumerate(cond):
    profiles['tgt'][c] = {}
    profiles['tgt'][c]['norm_depths'] = np.linspace(0,1,nDepths)
    profiles['tgt'][c]['coef'] = np.zeros((nDepths,))
    profiles['tgt'][c]['sterr'] = np.zeros((nDepths,))
    profiles['tgt'][c]['converged'] = np.zeros((nDepths,))
    profiles['tgt'][c]['icc'] = np.zeros((nDepths,))
    for d_i in range(nDepths):
        
        # fit mixed effects model
        temp_df = df_mixedLM[(df_mixedLM['d_bin'] == d_i) & (df_mixedLM['condition'] == c)]
        #temp_df['subjID'] = temp_df['subjID'].astype('category')
        model = smf.mixedlm(model_formula, data=temp_df, groups=temp_df['subjID'])
        
        # Converged?
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            try:
                result, converged = try_model_fit(model, maxiter=10000)
            except Warning as w:
                result, converged = None, False
                
        if not converged:
            print("MixedLM did not converge. Trying more iterations.")
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                result, converged = try_model_fit(model)
                if not converged:
                    print("\t Trying bfgs method.")
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', ConvergenceWarning)
                        result, converged = try_model_fit(model, method='bfgs')
                    if not converged:
                        print("\t \tTrying nm method.")
                        with warnings.catch_warnings():
                            warnings.simplefilter('error', ConvergenceWarning)
                            result, converged = try_model_fit(model, method='nm')
                        if not converged:
                            print("\t \t \t MixedLM failed to converge for region tgt condition %s depth %s. Check data." % (c, d_i))
        
        if converged:
            print("Model converged successfully.")
        else:
            print("Model did not converge.")

        # find the percent variance attributed to subject-level variability
        summary_table = result.summary().tables[1]
        subj_var = float(summary_table['Coef.'].loc['Group Var'])
        total_var = subj_var + result.scale
        frac_subj_var = subj_var / total_var
        
        # save results
        all_pvals = np.append(all_pvals,result.pvalues['Intercept'])
        all_fsv = np.append(all_fsv,frac_subj_var)
        profiles['tgt'][c]['coef'][d_i] = float(summary_table['Coef.'].loc['Intercept'])
        profiles['tgt'][c]['sterr'][d_i] = float(summary_table['Std.Err.'].loc['Intercept'])
        profiles['tgt'][c]['converged'][d_i] = converged
        profiles['tgt'][c]['icc'][d_i] = compute_icc(result)
    pvals_lookup[c] = [c_i*nDepths,c_i*nDepths+nDepths]
    
# single-condition mixed-effects subregions models
masked_df_rings = all_df[all_df['sig'] & all_df['no_vein']]
masked_df_rings = masked_df_rings.reset_index()
nCond = len(cond)
pos_ind = 0
for r_i, r in enumerate(rings):
    cond = [r+'_odss_decon',r+'_fgm_decon',r+'_iso-sur_decon',r+'_ctr-sur_decon']
    df_mixedLM_rings = pd.melt(masked_df_rings,id_vars=['subjID','hemi','d_bin','d','index','scale_xy_dist', 'd_bin_rings', 'in_ctr', 'in_border', 'in_sur'],value_vars=cond,var_name='condition',value_name='beta')
    #pvals_lookup[r] = {}
    profiles[r] = {}
    for c_i, c in enumerate(cond):
        profiles[r][c] = {}
        profiles[r][c]['norm_depths'] = np.linspace(0,1,nDepths_rings)
        profiles[r][c]['coef'] = np.zeros((nDepths_rings,))
        profiles[r][c]['sterr'] = np.zeros((nDepths_rings,))
        profiles[r][c]['converged'] = np.zeros((nDepths_rings,))
        profiles[r][c]['icc'] = np.zeros((nDepths_rings,))
        for d_i in range(nDepths_rings):
            # fit mixed effects model
            temp_df = df_mixedLM_rings[(df_mixedLM_rings['d_bin_rings'] == d_i) & (df_mixedLM_rings['condition'] == c) & (df_mixedLM_rings['in_'+r])]
            #temp_df['subjID'] = temp_df['subjID'].astype('category')
            model = smf.mixedlm(model_formula, data=temp_df, groups=temp_df['subjID'])
            result = model.fit()
            
            # Converged?
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                try:
                    result, converged = try_model_fit(model, maxiter=10000)
                except Warning as w:
                    result, converged = None, False
                    
            if not converged:
                print("MixedLM did not converge. Trying more iterations.")
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    result, converged = try_model_fit(model)
                    if not converged:
                        print("\t Trying bfgs method.")
                        with warnings.catch_warnings():
                            warnings.simplefilter('error', ConvergenceWarning)
                            result, converged = try_model_fit(model, method='bfgs')
                        if not converged:
                            print("\t \tTrying nm method.")
                            with warnings.catch_warnings():
                                warnings.simplefilter('error', ConvergenceWarning)
                                result, converged = try_model_fit(model, method='nm')
                            if not converged:
                                print("\t \t \t MixedLM failed to converge for region tgt condition %s depth %s. Check data." % (c, d_i))
            
            if converged:
                print("Model converged successfully.")
            else:
                print("Model did not converge.")
                
            # find the percent variance attributed to subject-level variability
            summary_table = result.summary().tables[1]
            subj_var = float(summary_table['Coef.'].loc['Group Var'])
            total_var = subj_var + result.scale
            frac_subj_var = subj_var / total_var
            
            # save results
            all_pvals = np.append(all_pvals,result.pvalues['Intercept'])
            all_fsv = np.append(all_fsv,frac_subj_var)
            profiles[r][c]['coef'][d_i] = float(summary_table['Coef.'].loc['Intercept'])
            profiles[r][c]['sterr'][d_i] = float(summary_table['Std.Err.'].loc['Intercept'])
            profiles[r][c]['converged'][d_i] = converged
            profiles[r][c]['icc'][d_i] = compute_icc(result)
        pvals_lookup[c] = [nDepths*nCond + pos_ind*nDepths_rings,nDepths*nCond + pos_ind*nDepths_rings+nDepths_rings]
        pos_ind += 1
        
# Report if all tests converged
print("\n \n Convergence Summary: \n")
for r in profiles.keys():
    print("r: %s" %r)
    for c in profiles[r].keys():
        print("\t c: %s" %c)
        if np.sum(profiles[r][c]['converged']) == len(profiles[r][c]['converged']):
            print("\t \t All converged")
        else:
            for i in range(len(profiles[r][c]['converged'])):
                print("\t \t d %d: %d" %(i, profiles[r][c]['converged'][i]))
        
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
        
# Place in profiles dictionary
for r_i, r in enumerate(profiles.keys()):
    if r == 'tgt':
        cond =  ['odss_decon','fgm_decon','iso-sur_decon','ctr-sur_decon']
    else:
        cond = [r+'_odss_decon',r+'_fgm_decon',r+'_iso-sur_decon',r+'_ctr-sur_decon']
    for c_i, c in enumerate(cond):
        profiles[r][c]['pvals'] = all_pvals_dict[c]
        
#compute averages
Stats = statDetails['labels'][:]
Colors = statDetails['colors'][:]
Diffs = list(diffDetails['statIDs'].keys())[:]

avgProfiles = {}
avgDiffs = {}
for r_i, r in enumerate(profiles.keys()):
    [avgProfiles[r], avgDiffs[r]] = compute_avg_depth_profile(depthProfiles[r],statDetails,diffDetails['statIDs'],Stats,Diffs,use_decon,prop_err,useSI)
    if r == 'tgt':
        cond =  ['odss_decon','fgm_decon','iso-sur_decon','ctr-sur_decon']
    else:
        cond = [r+'_odss_decon',r+'_fgm_decon',r+'_iso-sur_decon',r+'_ctr-sur_decon']
    for c_i, c in enumerate(cond):
        for k_i, k in enumerate(diffProfiles[r]):
            if k in c:
                profiles[r][c]['avg'] = avgDiffs[r][k]['avg']
                profiles[r][c]['stdev'] = avgDiffs[r][k]['stdev']
        
#%% Plot each context modulation effect separately

def plot_profile(p2,x,depths,errorbar,pvals,pval_thresh,color,xlim=[-0.8,1.5],ylim=[-0.02,1.02],showSig=True):
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
    # Plot profile
    p2.plot(x,depths,color=color)
    p2.fill_betweenx(depths,
                    x - errorbar,
                    x + errorbar,
                    linewidth=0.,
                    alpha=0.4,
                    color=color)
    p2.plot([0,0],ylim,'--k', alpha = 0.5)
    
    # Show significance
    top = x + errorbar
    p2.plot(top[pvals <= pval_thresh] + 0.1,depths[pvals <= pval_thresh],color='k',marker='$*$',linestyle='None')

    # Formatting
    p2.text(dx, dy + iStat*.07, stat,
            color=Colors[iStat],
            fontsize=fsize-2)
    p2.set_ylim(ylim)
    p2.set_xlim(xlim)
    p2.set_xlabel(r'$\Delta$ BOLD % change', fontsize=fsize, color=lcolor)
    p2.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
    
pthresh = 0.05 #pvalue significance threshold
if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7

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
plot_avg_depth_profile(p1,avgProfiles['tgt'],['sur','iso0'],[[0.7,0.7,0.7],'red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#iso-sur
xlim = [-0.8,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
# Plot MixedLM Coefficient
plot_profile(p2, profiles['tgt']['iso-sur_decon']['coef'], profiles['tgt']['iso-sur_decon']['norm_depths'], profiles['tgt']['iso-sur_decon']['sterr'], profiles['tgt']['iso-sur_decon']['pvals'], pthresh, color='k')
# Plot Average
#plot_profile(p2, profiles['tgt']['iso-sur_decon']['avg'], profiles['tgt']['iso-sur_decon']['norm_depths'], profiles['tgt']['iso-sur_decon']['stdev']/np.sqrt(avgDiffs['tgt']['iso-sur']['Nsamp']), profiles['tgt']['iso-sur_decon']['pvals'], pthresh, color='k')

if savefigs:
    if use_decon:
        fig.savefig(os.path.join(figDir,'avg_profiles_iso_sur_deconv.%s' %(fig_format)))
    else:
        fig.savefig(os.path.join(figDir,'avg_profiles_iso_sur.%s' %(fig_format)))

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
plot_avg_depth_profile(p1,avgProfiles['tgt'],['iso90','iso0'],['darkviolet','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#FGM
xlim = [-0.8,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
# Plot MixedLM Coefficient
plot_profile(p2, profiles['tgt']['fgm_decon']['coef'], profiles['tgt']['fgm_decon']['norm_depths'], profiles['tgt']['fgm_decon']['sterr'], profiles['tgt']['fgm_decon']['pvals'], pthresh, color='magenta')
# Plot Average
#plot_profile(p2, profiles['tgt']['fgm_decon']['avg'], profiles['tgt']['fgm_decon']['norm_depths'], profiles['tgt']['fgm_decon']['stdev']/np.sqrt(avgDiffs['tgt']['fgm']['Nsamp']), profiles['tgt']['fgm_decon']['pvals'], pthresh, color='magenta')

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
plot_avg_depth_profile(p1,avgProfiles['tgt'],['iso90','orth'],['darkviolet','orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#OTSS
xlim = [-0.8,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
# Plot MixedLM Coefficient
plot_profile(p2, profiles['tgt']['odss_decon']['coef'], profiles['tgt']['odss_decon']['norm_depths'], profiles['tgt']['odss_decon']['sterr'], profiles['tgt']['odss_decon']['pvals'], pthresh, color='green')
# Plot Average
#plot_profile(p2, profiles['tgt']['odss_decon']['avg'], profiles['tgt']['odss_decon']['norm_depths'], profiles['tgt']['odss_decon']['stdev']/np.sqrt(avgDiffs['tgt']['odss']['Nsamp']), profiles['tgt']['odss_decon']['pvals'], pthresh, color='green')

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
plot_avg_depth_profile(p1,avgProfiles['tgt'],['ctr_unwarp','sur_unwarp'],['gold','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#ctr-sur
xlim = [-0.8,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
# Plot MixedLM Coefficient
plot_profile(p2, profiles['tgt']['ctr-sur_decon']['coef'], profiles['tgt']['ctr-sur_decon']['norm_depths'], profiles['tgt']['ctr-sur_decon']['sterr'], profiles['tgt']['ctr-sur_decon']['pvals'], pthresh, color='coral')
# Plot Average
#plot_profile(p2, profiles['tgt']['ctr-sur_decon']['avg'], profiles['tgt']['ctr-sur_decon']['norm_depths'], profiles['tgt']['ctr-sur_decon']['stdev']/np.sqrt(avgDiffs['tgt']['ctr-sur']['Nsamp']), profiles['tgt']['ctr-sur_decon']['pvals'], pthresh, color='coral')

if savefigs:
    if use_decon:
        fig.savefig(os.path.join(figDir,'avg_profiles_ctr_sur_deconv.%s' %(fig_format)))
    else:
        fig.savefig(os.path.join(figDir,'avg_profiles_ctr_sur.%s' %(fig_format)))
        
#%% Rings Plot

pthresh = 0.05 #pvalue significance threshold
if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7

for r in ['ctr','border','sur']:
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
    plot_avg_depth_profile(p1,avgProfiles[r],['sur','iso0'],[[0.7,0.7,0.7],'red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)
    
    #iso-sur
    xlim = [-0.8,1.5]
    p2 = fig.add_axes([.7, .2, .25, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    # Plot MixedLM Coefficient
    plot_profile(p2, profiles[r][r+'_iso-sur_decon']['coef'], profiles[r][r+'_iso-sur_decon']['norm_depths'], profiles[r][r+'_iso-sur_decon']['sterr'], profiles[r][r+'_iso-sur_decon']['pvals'], pthresh, color='k')
    # Plot Average
    #plot_profile(p2, profiles['tgt']['iso-sur_decon']['avg'], profiles['tgt']['iso-sur_decon']['norm_depths'], profiles['tgt']['iso-sur_decon']['stdev']/np.sqrt(avgDiffs['tgt']['iso-sur']['Nsamp']), profiles['tgt']['iso-sur_decon']['pvals'], pthresh, color='k')
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'rings/%s_avg_profiles_iso_sur_deconv.%s' %(r,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'rings/%s_avg_profiles_iso_sur.%s' %(r,fig_format)))
    
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
    plot_avg_depth_profile(p1,avgProfiles[r],['iso90','iso0'],['darkviolet','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)
    
    #FGM
    xlim = [-0.8,1.5]
    p2 = fig.add_axes([.7, .2, .25, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    # Plot MixedLM Coefficient
    plot_profile(p2, profiles[r][r+'_fgm_decon']['coef'], profiles[r][r+'_fgm_decon']['norm_depths'], profiles[r][r+'_fgm_decon']['sterr'], profiles[r][r+'_fgm_decon']['pvals'], pthresh, color='magenta')
    # Plot Average
    #plot_profile(p2, profiles[r]['fgm_decon']['avg'], profiles[r]['fgm_decon']['norm_depths'], profiles[r]['fgm_decon']['stdev']/np.sqrt(avgDiffs[r]['fgm']['Nsamp']), profiles[r]['fgm_decon']['pvals'], pthresh, color='magenta')
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'rings/%s_avg_profiles_fgm_deconv.%s' %(r,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'rings/%s_avg_profiles_fgm.%s' %(r,fig_format)))
    
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
    plot_avg_depth_profile(p1,avgProfiles[r],['iso90','orth'],['darkviolet','orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)
    
    #OTSS
    xlim = [-0.8,1.5]
    p2 = fig.add_axes([.7, .2, .25, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    # Plot MixedLM Coefficient
    plot_profile(p2, profiles[r][r+'_odss_decon']['coef'], profiles[r][r+'_odss_decon']['norm_depths'], profiles[r][r+'_odss_decon']['sterr'], profiles[r][r+'_odss_decon']['pvals'], pthresh, color='green')
    # Plot Average
    #plot_profile(p2, profiles[r]['odss_decon']['avg'], profiles[r]['odss_decon']['norm_depths'], profiles[r]['odss_decon']['stdev']/np.sqrt(avgDiffs[r]['odss']['Nsamp']), profiles[r]['odss_decon']['pvals'], pthresh, color='green')
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'rings/%s_avg_profiles_otss_deconv.%s' %(r,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'rings/%s_avg_profiles_otss.%s' %(r,fig_format)))
    
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
    plot_avg_depth_profile(p1,avgProfiles[r],['ctr_unwarp','sur_unwarp'],['gold','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)
    
    #ctr-sur
    xlim = [-0.8,1.5]
    p2 = fig.add_axes([.7, .2, .25, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    # Plot MixedLM Coefficient
    plot_profile(p2, profiles[r][r+'_ctr-sur_decon']['coef'], profiles[r][r+'_ctr-sur_decon']['norm_depths'], profiles[r][r+'_ctr-sur_decon']['sterr'], profiles[r][r+'_ctr-sur_decon']['pvals'], pthresh, color='coral')
    # Plot Average
    #plot_profile(p2, profiles[r]['ctr-sur_decon']['avg'], profiles[r]['ctr-sur_decon']['norm_depths'], profiles[r]['ctr-sur_decon']['stdev']/np.sqrt(avgDiffs[r]['ctr-sur']['Nsamp']), profiles[r]['ctr-sur_decon']['pvals'], pthresh, color='coral')
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'rings/%s_avg_profiles_ctr_sur_deconv.%s' %(r,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'rings/%s_avg_profiles_ctr_sur.%s' %(r,fig_format)))

#%% Scatter plots

#Comparing OTSS and FGM
fig, ax = plt.subplots()
ax.plot(np.linspace(-5,20),np.linspace(-5,20),'k',label=None)
ax.scatter(masked_df['iso0'],masked_df['orth'],s=0.5,c='orange',alpha=0.7,label='Orth')
ax.scatter(masked_df['iso0'],masked_df['iso90'],s=0.5,c='darkviolet',alpha=0.7,label='Iso90')
ax.set_xlabel("iso0")
ax.set_title("Orth and Iso90")
fig.legend()

#Iso and Sur
fig, ax = plt.subplots()
ax.plot(np.linspace(-5,20),np.linspace(-5,20),'k',label=None)
ax.scatter(masked_df['sur'],masked_df['iso0'],s=0.5,c='gray',alpha=0.7,label='Orth')
ax.set_xlabel("sur")
ax.set_title("Iso and Sur")
fig.legend()

#OTSS and FGM separated by depth
fig, ax = plt.subplots(1,nDepths)
for d_i in range(nDepths):
    ax[d_i].plot(np.linspace(-5,20),np.linspace(-5,20),'k',label=None)
    ax[d_i].scatter(masked_df[masked_df['d_bin'] == d_i]['iso0'],masked_df[masked_df['d_bin'] == d_i]['orth'],s=0.5,c='orange',alpha=0.7,label='Orth')
    ax[d_i].scatter(masked_df[masked_df['d_bin'] == d_i]['iso0'],masked_df[masked_df['d_bin'] == d_i]['iso90'],s=0.5,c='darkviolet',alpha=0.7,label='Iso90')
    ax[d_i].set_title("depth = %d" %d_i)
ax[d_i].set_xlabel("iso0")
#fig.legend()

#OTSS and FGM separated by subject
Nsubj = len(masked_df['subjID'].unique())
fig, ax = plt.subplots(1,Nsubj)
for s_i, sID in enumerate(masked_df['subjID'].unique()):
    ax[s_i].plot(np.linspace(-5,20),np.linspace(-5,20),'k',label=None)
    ax[s_i].scatter(masked_df[masked_df['subjID'] == sID]['iso0'],masked_df[masked_df['subjID'] == sID]['orth'],s=0.5,c='orange',alpha=0.7,label='Orth')
    ax[s_i].scatter(masked_df[masked_df['subjID'] == sID]['iso0'],masked_df[masked_df['subjID'] == sID]['iso90'],s=0.5,c='darkviolet',alpha=0.7,label='Iso90')
    ax[s_i].set_title("Subj %s" %sID)
ax[s_i].set_xlabel("iso0")
#fig.legend()

#OTSS and FGM separated by depth and subject
fig, ax = plt.subplots(Nsubj,nDepths)
for s_i, sID in enumerate(masked_df['subjID'].unique()):
    for d_i in range(nDepths):
        ax[s_i,d_i].plot(np.linspace(-5,20),np.linspace(-5,20),'k',label=None)
        ax[s_i,d_i].scatter(masked_df[(masked_df['subjID'] == sID) & (masked_df['d_bin'] == d_i)]['iso0'],masked_df[(masked_df['subjID'] == sID) & (masked_df['d_bin'] == d_i)]['orth'],s=0.5,c='orange',alpha=0.7,label='Orth')
        ax[s_i,d_i].scatter(masked_df[(masked_df['subjID'] == sID) & (masked_df['d_bin'] == d_i)]['iso0'],masked_df[(masked_df['subjID'] == sID) & (masked_df['d_bin'] == d_i)]['iso90'],s=0.5,c='darkviolet',alpha=0.7,label='Iso90')
        ax[s_i,d_i].set_title("Subj %s depth = %d" %(sID,d_i),fontsize=8)
    ax[s_i,d_i].set_xlabel("iso0",fontsize=6)
fig.tight_layout(pad=0)
#fig.legend()
