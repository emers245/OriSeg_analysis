#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 12:42:00 2024

@author: Joe

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
savefigs = True #if true save all figures
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '.'
figDir = mainDir+'/figs/individual_ROIs/'
fig_format = 'svg'
statCorrType = 'fdr_bh' #'bonferroni'

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset

datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual', 'pnr???_??_???_??.csv'))
#datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_filled', 'pnr???_??_???_??.csv'))
#or exclude
exclude_initial = ['pnr143_V1_tgt_rh','pnr143_V1_tgt_lh','pnr161_V1_tgt_lh','pnr161_V1_tgt_rh','pnr352_V1_tgt_lh','pnr352_V1_tgt_rh','pnr579_V1_tgt_lh','pnr579_V1_tgt_rh','pnr668_V1_tgt_rh', 'pnr668_V1_tgt_lh']
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
    fthresh = plot_mnv_histograms(lmnv, lmnv[deep], mnv_mask, deep_pct, key, k_i, NROIs, fsize, pad=0.0, figsize=(15,3))
    
    # Plot depth maps
    dmap = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [2,5], fsize, fname = 'dmap', pad=0.0)
        
    #plot thresholded map
    dmap_thresh = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [2,5], fsize, fname='dmap_thresh', mask=mnv_mask, pad=0.0)
        
    # Plot voxel loss at each depth after masking
    fdepth_hist = plot_depth_voxel_loss(z, mnv_mask, nDepths, NROIs, key, k_i, fsize)
    
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

#%% Plot Depth Profiles
    
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
      
#%% Recalculate Voxels per bin
fdhist = plt.figure(figsize=(15,4))
vox_per_bin = np.zeros((nDepths,len(all_data)))

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[masks[label]]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,int(np.ceil(len(datasets)/2)),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.title(label)
    plt.xlabel("Normalize Depth WM -> GM")
    plt.ylabel("Num. Voxels")
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
    
    vox_per_bin[:,iR] = np.histogram(roi['d'],bins=nDepths)[0]
fdhist.tight_layout(pad=0.0)

favg = plt.figure(figsize=(6,4))
plt.bar(np.linspace(0,1,nDepths),np.mean(vox_per_bin,axis=1),1/nDepths,alpha=0.6)
plt.errorbar(np.linspace(0,1,nDepths),np.mean(vox_per_bin,axis=1),np.std(vox_per_bin,axis=1),linestyle='')
all_bins = np.tile(np.linspace(0,1,nDepths),(len(all_data),1))
plt.scatter(all_bins,vox_per_bin,s=1,color='b')
plt.xlabel("Depth")
plt.ylabel("Average Voxel Count")

if savefigs:
    fdhist.savefig(os.path.join(figDir,'depth_hist_masked.%s' %(fig_format)))

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
    
#Now subjtract deconvolved mean at each depth from every voxel and put it back in the dataset
for iR, roiID in enumerate(all_data.keys()):
    for iStat, stat in enumerate(depthProfiles):
        all_data[roiID]['d_bin'] = pd.cut(all_data[roiID]['d'], bins=nDepths, labels=False)
        adjustment = depthProfiles[stat]['avg'][iR] - depthProfiles[stat]['avg_decon'][iR]
        all_data[roiID][stat+'_decon'] = np.zeros(len(all_data[roiID]))
        for d_i in range(nDepths):
            d_mask = (all_data[roiID]['d_bin'] == d_i)
            all_data[roiID][stat+'_decon'][d_mask] = all_data[roiID][stat][d_mask] - adjustment[d_i]
    
#%% Plot Individual ROI Profiles

def plot_conditions(roi_data, cond1, cond2, cond1_color = 'blue', cond2_color = 'orange', diff_color = 'gray', lcolor='black', fcolor='white', roi_label = None, diff_label = "Difference", fsize = 14, left_xlim = [0,15], right_xlim = [-3,3], err = 'std', plot_scatter=False):
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)

    fig.clf()

    p1 = fig.add_axes([.15, .2, .3, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

    # Normalize Depth
    roi_data['d_norm'] = (roi_data['d'] - roi_data['d'].min()) / (roi_data['d'].max() - roi_data['d'].min())
    
    # Left subplot: Conditions vs. depth
    ax1 = p1
    offset = 0
    color_list = [cond1_color, cond2_color]
    for cond_i, cond in enumerate([cond1, cond2]):
        if plot_scatter:
            ax1.scatter(roi_data[cond], roi_data['d_norm'], s=1, c=color_list[cond_i], label=cond, alpha=0.5)
        mean_values = roi_data.groupby('d_bin')[cond].mean()
        if err == 'std':
            error_values = roi_data.groupby('d_bin')[cond].std()
        elif err == 'sem':
            error_values = roi_data.groupby('d_bin')[cond].sem()
        else:
            raise Exception("plot_conditions: Invalid option 'err'")
        bins = mean_values.index
        bins_norm = (bins - np.min(bins)) / (np.max(bins) - np.min(bins))
        #ax1.errorbar(mean_values, bins_norm+offset, xerr=std_values, color=color_list[cond_i], label=f'{cond} Mean', fmt='-', capsize=5)
        ax1.plot(mean_values, bins_norm, color=color_list[cond_i], label=f'{cond} Mean')
        ax1.fill_betweenx(bins_norm,
                        mean_values - error_values,
                        mean_values + error_values,
                        linewidth=0.,
                        alpha=0.4,
                        color=color_list[cond_i])
        #offset+=0.01

    ax1.set_xlim(left_xlim)
    ax1.set_xlabel('BOLD % change')
    ax1.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)')
    ax1.legend(fontsize=8)
    ax1.set_title(f'{roi_label}: {cond1} and {cond2}')

    # Right subplot: Difference between conditions vs. depth
    p2 = fig.add_axes([.7, .2, .25, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
    ax2 = p2
    roi_data['Difference'] = roi_data[cond1] - roi_data[cond2]
    if plot_scatter:
        ax2.scatter(roi_data['Difference'], roi_data['d_norm'], s=1, c=diff_color, label=diff_label, alpha=0.5)
    mean_diff = roi_data.groupby('d_bin')['Difference'].mean()
    if err == 'std':
        error_diff = roi_data.groupby('d_bin')['Difference'].std()
    elif err == 'sem':
        error_diff = roi_data.groupby('d_bin')['Difference'].sem()
    else:
        raise Exception("plot_conditions: Invalid option 'err'")
    ax2.plot(mean_diff, bins_norm, color=diff_color, label=diff_label)
    ax2.fill_betweenx(bins_norm,
                    mean_diff - error_diff,
                    mean_diff + error_diff,
                    linewidth=0.,
                    alpha=0.4,
                    color=diff_color)
    ax2.set_xlim(right_xlim)

    ax2.set_xlabel(r'$\Delta$ BOLD %')
    ax2.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)')
    ax2.legend(fontsize=8)
    ax2.set_title(f'{roi_label}: {cond1} - {cond2}')

    plt.tight_layout()
    plt.show()
    
    return fig

# OTSS
for label in all_data.keys():
    roi = all_data[label]
    roi = roi[roi['in_tgt'] & roi['sig'] & roi['no_vein']]
    roi['d_bin'] = pd.cut(roi['d'], bins=nDepths, labels=False)
    fig = plot_conditions(roi, 'orth', 'iso90', cond1_color='orange', cond2_color='darkviolet', diff_color='green', roi_label = label, diff_label = 'OTSS')
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_otss.%s' %(label,fig_format)))
        
# FGM
for label in all_data.keys():
    roi = all_data[label]
    roi = roi[roi['in_tgt'] & roi['sig'] & roi['no_vein']]
    roi['d_bin'] = pd.cut(roi['d'], bins=nDepths, labels=False)
    fig = plot_conditions(roi, 'iso90', 'iso0', cond1_color='darkviolet', cond2_color='red', diff_color='magenta', roi_label = label, diff_label = 'FGM')
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_fgm.%s' %(label,fig_format)))
        
# iso-sur
for label in all_data.keys():
    roi = all_data[label]
    roi = roi[roi['in_tgt'] & roi['sig'] & roi['no_vein']]
    roi['d_bin'] = pd.cut(roi['d'], bins=nDepths, labels=False)
    fig = plot_conditions(roi, 'iso0', 'sur', cond1_color='red', cond2_color='gray', diff_color='black', roi_label = label, diff_label = 'iso-sur')
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_iso-sur.%s' %(label,fig_format)))
        
# ctr-sur
for label in all_data.keys():
    roi = all_data[label]
    roi = roi[roi['in_tgt'] & roi['sig'] & roi['no_vein']]
    roi['d_bin'] = pd.cut(roi['d'], bins=nDepths, labels=False)
    fig = plot_conditions(roi, 'ctr_unwarp', 'sur_unwarp', cond1_color='gold', cond2_color='purple', diff_color='coral', roi_label = label, diff_label = 'ctr-sur', left_xlim = [-5,15], right_xlim = [-3,7])
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_ctr-sur.%s' %(label,fig_format)))
        
#%% Plot Individual ROI Profile Conditions Separately

use_decon = True #use deconvolved profiles?
use_veinmask = True

def plot_one_condition(roi_data, cond1, cond1_color = 'blue', lcolor='black', fcolor='white', roi_label = None, fsize = 14, xlim = [0,15], err = 'std', plot_scatter=False):
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)

    fig.clf()

    p1 = fig.add_axes([.15, .2, .3, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

    # Normalize Depth
    roi_data['d_norm'] = (roi_data['d'] - roi_data['d'].min()) / (roi_data['d'].max() - roi_data['d'].min())
    
    # Left subplot: Conditions vs. depth
    ax1 = p1
    offset = 0
    if plot_scatter:
        ax1.scatter(roi_data[cond1], roi_data['d_norm'], s=1, c=cond1_color, label=cond1, alpha=0.5)
    mean_values = roi_data.groupby('d_bin')[cond1].mean()
    if err == 'std':
        error_values = roi_data.groupby('d_bin')[cond1].std()
    elif err == 'sem':
        error_values = roi_data.groupby('d_bin')[cond1].sem()
    else:
        raise Exception("plot_conditions: Invalid option 'err'")
    bins = mean_values.index
    bins_norm = (bins - np.min(bins)) / (np.max(bins) - np.min(bins))
    #ax1.errorbar(mean_values, bins_norm+offset, xerr=std_values, color=color_list[cond_i], label=f'{cond} Mean', fmt='-', capsize=5)
    ax1.plot(mean_values, bins_norm, color=cond1_color, label=f'{cond1} Mean')
    ax1.fill_betweenx(bins_norm,
                        mean_values - error_values,
                        mean_values + error_values,
                        linewidth=0.,
                        alpha=0.4,
                        color=cond1_color)
        #offset+=0.01

    ax1.set_xlim(xlim)
    ax1.set_xlabel('BOLD % change')
    ax1.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)')
    ax1.legend(fontsize=8)
    ax1.set_title(f'{roi_label}: {cond1}')

    plt.tight_layout()
    plt.show()
    
    return fig

# Ctr
for label in all_data.keys():
    roi = all_data[label]
    if use_veinmask:
        roi = roi[roi['in_tgt'] & roi['sig'] & roi['no_vein']]
    else:
        roi = roi[roi['in_tgt'] & roi['sig']]
    roi['d_bin'] = pd.cut(roi['d'], bins=nDepths, labels=False)
    if use_decon:
        fig = plot_one_condition(roi, 'ctr_unwarp_decon', cond1_color='gold', roi_label = label, xlim = [-5,15])
    else:
        fig = plot_one_condition(roi, 'ctr_unwarp', cond1_color='gold', roi_label = label, xlim = [-5,15])
    
    if savefigs:
        if use_decon:
            if use_veinmask:
                fig.savefig(os.path.join(figDir,'%s_ctr_unwarp_decon_veinmask.%s' %(label,fig_format)))
            else:
                fig.savefig(os.path.join(figDir,'%s_ctr_unwarp_decon.%s' %(label,fig_format)))
        elif use_veinmask:
            fig.savefig(os.path.join(figDir,'%s_ctr_unwarp_veinmask.%s' %(label,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_ctr_unwarp.%s' %(label,fig_format)))
        
# Sur
for label in all_data.keys():
    roi = all_data[label]
    if use_veinmask:
        roi = roi[roi['in_tgt'] & roi['sig'] & roi['no_vein']]
    else:
        roi = roi[roi['in_tgt'] & roi['sig']]
    roi['d_bin'] = pd.cut(roi['d'], bins=nDepths, labels=False)
    if use_decon:
        fig = plot_one_condition(roi, 'sur_unwarp_decon', cond1_color='purple', roi_label = label, xlim = [-5,15])
    else:
        fig = plot_one_condition(roi, 'sur_unwarp', cond1_color='purple', roi_label = label, xlim = [-5,15])
    
    if savefigs:
        if use_decon:
            if use_veinmask:
                fig.savefig(os.path.join(figDir,'%s_sur_unwarp_decon_veinmask.%s' %(label,fig_format)))
            else:
                fig.savefig(os.path.join(figDir,'%s_sur_unwarp_decon.%s' %(label,fig_format)))
        elif use_veinmask:
            fig.savefig(os.path.join(figDir,'%s_sur_unwarp_veinmask.%s' %(label,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_sur_unwarp.%s' %(label,fig_format)))

# Orth
for label in all_data.keys():
    roi = all_data[label]
    if use_veinmask:
        roi = roi[roi['in_tgt'] & roi['sig'] & roi['no_vein']]
    else:
        roi = roi[roi['in_tgt'] & roi['sig']]
    roi['d_bin'] = pd.cut(roi['d'], bins=nDepths, labels=False)
    if use_decon:
        fig = plot_one_condition(roi, 'orth_decon', cond1_color='orange', roi_label = label)
    else:
        fig = plot_one_condition(roi, 'orth', cond1_color='orange', roi_label = label)
    
    if savefigs:
        if use_decon:
            if use_veinmask:
                fig.savefig(os.path.join(figDir,'%s_orth_decon_veinmask.%s' %(label,fig_format)))
            else:
                fig.savefig(os.path.join(figDir,'%s_orth_decon.%s' %(label,fig_format)))
        elif use_veinmask:
            fig.savefig(os.path.join(figDir,'%s_orth_veinmask.%s' %(label,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_orth.%s' %(label,fig_format)))
        
# Iso
for label in all_data.keys():
    roi = all_data[label]
    if use_veinmask:
        roi = roi[roi['in_tgt'] & roi['sig'] & roi['no_vein']]
    else:
        roi = roi[roi['in_tgt'] & roi['sig']]
    roi['d_bin'] = pd.cut(roi['d'], bins=nDepths, labels=False)
    if use_decon:
        fig = plot_one_condition(roi, 'iso0_decon', cond1_color='red', roi_label = label)
    else:
        fig = plot_one_condition(roi, 'iso0', cond1_color='red', roi_label = label)
    
    if savefigs:
        if use_decon:
            if use_veinmask:
                fig.savefig(os.path.join(figDir,'%s_iso_decon_veinmask.%s' %(label,fig_format)))
            else:
                fig.savefig(os.path.join(figDir,'%s_iso_decon.%s' %(label,fig_format)))
        elif use_veinmask:
            fig.savefig(os.path.join(figDir,'%s_iso_veinmask.%s' %(label,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_iso.%s' %(label,fig_format)))
        
# Iso90
for label in all_data.keys():
    roi = all_data[label]
    if use_veinmask:
        roi = roi[roi['in_tgt'] & roi['sig'] & roi['no_vein']]
    else:
        roi = roi[roi['in_tgt'] & roi['sig']]
    roi['d_bin'] = pd.cut(roi['d'], bins=nDepths, labels=False)
    if use_decon:
        fig = plot_one_condition(roi, 'iso90_decon', cond1_color='darkviolet', roi_label = label)
    else:
        fig = plot_one_condition(roi, 'iso90', cond1_color='darkviolet', roi_label = label)
    
    if savefigs:
        if use_decon:
            if use_veinmask:
                fig.savefig(os.path.join(figDir,'%s_iso90_decon_veinmask.%s' %(label,fig_format)))
            else:
                fig.savefig(os.path.join(figDir,'%s_iso90_decon.%s' %(label,fig_format)))
        elif use_veinmask:
            fig.savefig(os.path.join(figDir,'%s_iso90_veinmask.%s' %(label,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_iso90.%s' %(label,fig_format)))
        
#%% Loc Profiles across surface!
# smoothing according to line 441 in analysisROI_dev
#with smoothing
plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(14, 7))
fontsize = 8
fig1.patch.set_facecolor(fcolor)
locStatDetails = {'labels':['ctr-sur_unwarp'], 'colors':['black']}


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
            ax.scatter(x_smooth, coef_smooth, c=coef_smooth, cmap='plasma', alpha=alpha, label=locStatDetails['labels'][iStat])
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
                    ax.legend(fontsize = 6)
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

#%% Average Stat Porfiles with individual profiles overlaid

fig = plt.figure(figsize=(5, 6))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12
colors = ['black']
plotStats = ['ctr-sur_unwarp']
plot_indiv = True #if true, overlay individiual traces

for iD, depth_label in enumerate(['deep','middle','superficial']):
    print(depth_label)
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    fix_axes(p, lcolor=lcolor, fcolor=fcolor)
    for iStat, label in enumerate(plotStats):
        nProfiles = len(all_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [-6, 6], '--', color=(.3, .3, .3))
        p.plot([0, 4], [0, 0], '--', color=(.3, .3 ,.3))

        p.plot(x_smooth, stat_avg, color=colors[iStat], label=label)
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
            p.legend()
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-5, 7])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
if savefigs:
    fig.savefig(os.path.join(figDir,"average_indiv_radial_profiles_loc.%s" %(fig_format)))
        
#%% Stats across surface!
# smoothing according to line 441 in analysisROI_dev
#with smoothing
plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(14, 7))
fontsize = 8
fig1.patch.set_facecolor(fcolor)
locStatDetails = {'labels':['ctr-sur_unwarp','iso0','iso90','orth','sur'], 'colors':['black','red','purple','orange','gray']}

radMax = 4
nRadii = 20
ymax = 10
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
        # ax2 = ax.twinx()
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

            ax.plot(x_smooth, coef_smooth, color=locStatDetails['colors'][iStat], alpha=alpha, label=locStatDetails['labels'][iStat])
            #ax.scatter(x_smooth, coef_smooth, c=coef_smooth, cmap='plasma', alpha=alpha)
            ax.set_ylim([ymin, ymax])
            ax.set_xticks([0, 1, 2, 3])
            if iD == 2:
                title = label.split('_')
                ax.set_title(title[0]+' '+title[-1], fontsize=8, color=lcolor) 
            ax.plot([2, 2], [ymin, ymax], '--', color='gray', alpha=0.15)
            ax.plot([np.min(x_smooth), np.max(x_smooth)], [0, 0], '--', color='gray', alpha=0.15)
            fix_axes(ax, lcolor=lcolor, fcolor=fcolor)
            if iR > 0:
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
            else:
                if iD == 0:
                    ax.set_xlabel("$\sigma$ from ROI center", fontsize = 8, color=lcolor)
                    ax.legend(fontsize=6)
            if iD == 0:
                ax.set_xticklabels(['0', '', '2', ''], fontsize=8, color=lcolor)
            else:
                ax.set_xticklabels([])
               
            # # Now plot ctr - sur Fstat across surface
            # if plot_Fstat:
            #     Fstat = df['ctr-sur F'].values
            #     x = df['scale_xy_dist'].values
            #     #smooth data
            #     Fstat_smooth, x_smooth = smoothen(Fstat, x)
    
            #     ax2.plot(x_smooth, Fstat_smooth, color='black', alpha=alpha)
            #     ax2.set_ylim([0, 40])
            #     ax2.set_xticks([0, 1, 2, 3])
            #     if iD == 3:
            #         ax2.set_title(label, fontsize=8, color=lcolor) 
            #     ax2.plot([2, 2], [0, ymax], '--', color='gray', alpha=0.15)
            #     fix_axes(ax2, lcolor=lcolor, fcolor=fcolor)
            #     if iR > 0:
            #         ax2.yaxis.set_visible(False)
            #         ax2.spines['right'].set_visible(False)
                    
if savefigs:
    fig1.savefig(os.path.join(figDir,"radial_profiles_task.%s" %(fig_format)))
#%% Compute differences and put them back in all_data
for diff in diffDetails['statIDs'].keys():
    for iR, label in enumerate(all_data.keys()):
        stat1 = diffDetails['statIDs'][diff][0]
        stat2 = diffDetails['statIDs'][diff][1]
        all_data[label][diff] = all_data[label][stat1] - all_data[label][stat2]
      
#%% Difference profiles across radial distance

plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(14, 7))
fontsize = 8
#fig1.patch.set_facecolor(fcolor)
diffStatDetails = {'labels':['fgm','odss','iso-sur'], 'colors':['magenta','green','gray']}

radMax = 4
nRadii = 20
ymax = 3
ymin = -3
highlight = False #['ctr-sur'] #['orth', 'iso90']#'orth'
depth_labels = ['deep', 'middle', 'superficial']
depthBoundaries = np.array([[0,0.3],[0.3,0.6],[0.6,1]])

# Add to all_profiles
for d in depth_labels:
    for iStat, stat in enumerate(diffStatDetails['labels']):
        all_profiles[depth_labels[iD]][stat] = []

for iR, label in enumerate(all_data.keys()):
    for iD in range(3):
        df = all_data[label][mask_dict[label]]
        df = df[(df['d'] >=depthBoundaries[iD,0]) & (df['d'] <depthBoundaries[iD,1])]
        aw = .8/len(all_data)
        plt.figure(1)
        ax = plt.axes([.07 + iR*1.1*aw, .14 + iD*.3,  aw, .2])
        #ax2 = ax.twinx()
        for iStat in range(len(diffStatDetails['labels'])):
            alpha = 1.
            if highlight:
                if diffStatDetails['labels'][iStat] in highlight:
                    alpha = 1.
                else:
                    alpha = 0.2
            coef = df[diffStatDetails['labels'][iStat]].values
            x = df['scale_xy_dist'].values
            #smooth data
            coef_smooth, x_smooth = smoothen(coef, x)
            all_profiles[depth_labels[iD]][diffStatDetails['labels'][iStat]].append(coef_smooth)
            # dataDict = makeProfile1D(x, nRadii, coef)
            # plt.plot(dataDict['profile']['depth'],
            #          dataDict['profile']['avg'][0],
            #          '--',
            #          color=statDetails['colors'][iStat])

            ax.plot(x_smooth, coef_smooth, color=diffStatDetails['colors'][iStat], alpha=alpha, label=diffStatDetails['labels'][iStat])
            #ax.scatter(x_smooth, coef_smooth, c=coef_smooth, cmap='plasma', alpha=alpha)
            ax.set_ylim([ymin, ymax])
            ax.set_xticks([0, 1, 2, 3])
            if iD == 2:
                title = label.split('_')
                ax.set_title(title[0]+' '+title[-1], fontsize=8, color=lcolor) 
            ax.plot([2, 2], [ymin, ymax], '--', color='gray', alpha=0.15)
            ax.plot([np.min(x_smooth), np.max(x_smooth)], [0, 0], '--', color='gray', alpha=0.15)
            fix_axes(ax, lcolor=lcolor, fcolor=fcolor)
            if iR > 0:
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
            else:
                if iD == 0:
                    ax.set_xlabel("$\sigma$ from ROI center", fontsize = 8, color=lcolor)
                    ax.legend(fontsize=6)
            if iD == 0:
                ax.set_xticklabels(['0', '', '2', ''], fontsize=8, color=lcolor)
            else:
                ax.set_xticklabels([])
                    
if savefigs:
    fig1.savefig(os.path.join(figDir,"radial_profiles_diff.%s" %(fig_format)))
    
#%% Average Loc Porfiles with individual profiles overlaid

fig = plt.figure(figsize=(5, 6))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12
colors = ['orange']
plotStats = ['iso90']
plot_indiv = True #if true, overlay individiual traces
ymin = -3
ymax = 12

for iD, depth_label in enumerate(['deep','middle','superficial']):
    print(depth_label)
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    fix_axes(p, lcolor=lcolor, fcolor=fcolor)
    for iStat, label in enumerate(plotStats):
        nProfiles = len(all_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [-6, 6], '--', color=(.3, .3, .3))
        p.plot([0, 4], [0, 0], '--', color=(.3, .3 ,.3))

        p.plot(x_smooth, stat_avg, color=colors[iStat], label=label)
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
            p.legend()
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([ymin, ymax])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
if savefigs:
    fig.savefig(os.path.join(figDir,"average_indiv_radial_profiles_%s.%s" %(plotStats,fig_format)))

