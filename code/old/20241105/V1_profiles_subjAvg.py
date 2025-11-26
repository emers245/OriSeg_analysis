#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cheryl

This code combines work from Cheryl and Joe for creating laminar and surface
profiles from the OriSeg data. This code specifically focuses on the analysis 
of contextual conditions.

Update (02/21/2024): This version treats each subject as if they have a single
ROI that spans both hemispheres (except for subjects that have one exluded 
hemisphere).

Update (09/20/2024): Adding in gPPI analysis
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
figDir = mainDir+'/figs/subjAvg/'
fig_format = 'svg'
statCorrType = 'fdr_bh' #'bonferroni'

#Set random seed
np.random.seed(68752)

#%%###########################################################################

mainDir = '.'
datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual', 'pnr???_??_???_??.csv'))
#datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_filled', 'pnr???_??_???_??.csv'))
#or exclude
exclude_initial = ['pnr143_V1_tgt_lh','pnr143_V1_tgt_rh','pnr161_V1_tgt_lh','pnr161_V1_tgt_rh','pnr352_V1_tgt_lh','pnr352_V1_tgt_rh','pnr579_V1_tgt_lh','pnr668_V1_tgt_rh']
# exclude_initial = ['pnr143_V1_tgt_lh','pnr143_V1_tgt_rh',
#                    'pnr161_V1_tgt_lh','pnr161_V1_tgt_rh',
#                    'pnr352_V1_tgt_lh','pnr352_V1_tgt_rh',
#                    'pnr495_V1_tgt_lh','pnr495_V1_tgt_rh',
#                    'pnr579_V1_tgt_lh','pnr579_V1_tgt_rh',
#                    'pnr668_V1_tgt_lh','pnr668_V1_tgt_rh',
#                    'pnr685_V1_tgt_lh','pnr685_V1_tgt_rh',
#                    'pnr713_V1_tgt_lh','pnr713_V1_tgt_rh',
#                    'pnr822_V1_tgt_lh','pnr822_V1_tgt_rh']
for e_i, excl in enumerate(exclude_initial):
    datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_roi_manual',excl+'.csv'))
    #datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_filled',excl+'.csv'))        
datasets.sort()

# datasets = [#'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_lh_target_laynii.csv',
#     #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_rh_target_laynii.csv',
#     mainDir+'/roi_data_manualSeg/target_filled/pnr256_V1_tgt_lh_rad10.csv',
#     mainDir+'/roi_data_manualSeg/target_filled/pnr256_V1_tgt_rh_rad10.csv',
#     mainDir+'/roi_data_manualSeg/target_filled/pnr328_V1_tgt_lh_rad10.csv',
#     mainDir+'/roi_data_manualSeg/target_filled/pnr328_V1_tgt_rh_rad10.csv',
#     mainDir+'/roi_data_manualSeg/target_filled/pnr510_V1_tgt_lh_rad10.csv',
#     mainDir+'/roi_data_manualSeg/target_filled/pnr510_V1_tgt_rh_rad10.csv',
#     mainDir+'/roi_data_manualSeg/target_filled/pnr739_V1_tgt_lh_rad10.csv',
#     mainDir+'/roi_data_manualSeg/target_filled/pnr739_V1_tgt_rh_rad10.csv',
# #   mainDir+'/roi_data_manualSeg/target_filled/pnr756_V1_tgt_lh_rad10.csv',
#     mainDir+'/roi_data_manualSeg/target_filled/pnr756_V1_tgt_rh_rad10.csv'
#   ]
datasets.sort()
Ndsets = len(datasets)

#make all_data dataframe
roiRad = 1. #radius for target ROI
centerRad = 1.7 #radius for center of ROI
borderRad = [1.7,2.3] #radius range for border
surRad = [3.5] #[3, 3.5] #outside of this radius will be considered the surround
import pandas as pd
all_data = {}
subjIDs = []
for dataset in datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    sID = f[f.find('pnr'):f.find('pnr')+6]
    if sID not in subjIDs:
        all_data[f[:-9]] = pd.read_csv(dataset, sep=',', index_col=False)
        subjIDs.append(sID)
        if 'lh' in f:
            all_data[f[:-9]]['hemi'] = ['lh' for i in range(len(all_data[f[:-9]]))]
        elif 'rh' in f:
            all_data[f[:-9]]['hemi'] = ['rh' for i in range(len(all_data[f[:-9]]))]
        else:
            print('No hemi label!!!')
    else:
        new_hemi = pd.read_csv(dataset, sep=',', index_col=False)
        if 'lh' in f:
            new_hemi['hemi'] = ['lh' for i in range(len(new_hemi))]
        elif 'rh' in f:
            new_hemi['hemi'] = ['rh' for i in range(len(new_hemi))]
        else:
            print('No hemi label!!!')
        all_data[f[:-9]] = pd.concat((all_data[f[:-9]],new_hemi),ignore_index=True)
    
## THIS IS A HACKY FIX TO GET RID OF DEPTH = 0 VOXELS; SHOULD REMOVE THIS IN THE FUTURE
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    df = df.drop(df[df['d'] == 0].index)
    
    all_data[label] = df

# check and see what the Stria profile looks like in each ROI
nDepths = 7
nDepths_rings = 3
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
iR = 0
for iS, label in enumerate(all_data.keys()):
    df_all = all_data[label]
    
    for iH, hemi in enumerate(df_all['hemi'].unique()):
        
        df = df_all[df_all['hemi']==hemi]

        # recapitulate the fitting, which is a bit of overkill, but it gets us 
        # an accurate ellipse
        tgt_df = df[df['ctr-sur'] > 0]
        #tgt_df = df[df['scale_xy_dist'] <= 2]
        cov = np.cov(tgt_df['x'][df['scale_xy_dist'] < 2.2],
                     tgt_df['y'][df['scale_xy_dist'] < 2.2])
        com = (np.mean(tgt_df['x'][df['scale_xy_dist'] < 2.2]),
               np.mean(tgt_df['y'][df['scale_xy_dist'] < 2.2]))
        
        # Get an xy-distance measure in mm in addition to the normalized version
        df['xy_dist'] = np.sqrt((df['x']-com[0])**2 + (df['y']-com[1])**2)
        # Update the 'xy_dist' in the original dataframe 'df_all'
        df_all.loc[df_all['hemi'] == hemi, 'xy_dist'] = df['xy_dist']
        
        # Recompute the major and minor axes of the ellipse
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
        
        iR += 1
    
    #save back to main dataframe
    all_data[label] = df_all
    
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
    plt.text(0.8,5,'< 0.01 = %d %%' %(100*np.sum(roi['loc p-val'] <= 0.01)/len(roi['loc p-val'])))
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
    plt.text(0.8,5,'< 0.01 = %d %%' %(100*np.sum(roi['task p-val'] <= 0.01)/len(roi['task p-val'])))
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
NROIs = len(datasets)
dropout = {'superficial': np.zeros(NROIs),
           'middle': np.zeros(NROIs),
           'deep': np.zeros(NROIs),
           'total': np.zeros(NROIs)} #dropout rates

k_i = 0
for S_i, key in enumerate(all_data.keys()):
    df_all = all_data[key]
    mnv_mask_all = np.array([])
    for h_i, hemi in enumerate(df_all['hemi'].unique()):

        # calculate log(MNV)
        df = df_all[df_all['hemi']==hemi]
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
        
        # Plot distributions
        fthresh = plot_mnv_histograms(lmnv, lmnv[deep], mnv_mask, deep_pct, key, k_i, Ndsets, fsize, pad=0.0, figsize=(15,3))
        
        # Plot depth maps
        dmap = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, Ndsets, [2,5], fsize, fname = 'dmap', pad=0.0)
            
        #plot thresholded map
        dmap_thresh = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, Ndsets, [2,5], fsize, fname='dmap_thresh', mask=mnv_mask, pad=0.0)
            
        # Plot voxel loss at each depth after masking
        fdepth_hist = plot_depth_voxel_loss(z, mnv_mask, nDepths, Ndsets, key, k_i, fsize)
        
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

        mnv_mask_all = np.append(mnv_mask_all,mnv_mask.tolist())
        k_i += 1
        
    all_data[key]['no_vein'] = mnv_mask_all==1
    mask_dict[key] = mnv_mask_all
    
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

#try violin plots
spreadF = 2
f = plt.figure()
k_i = 0
for S_i, key in enumerate(all_data.keys()):
    df_all = all_data[key]
    
    for h_i, hemi in enumerate(df_all['hemi'].unique()):

        # calculate log(MNV)
        df = df_all[df_all['hemi']==hemi]
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
        
        k_i += 1
plt.xticks(np.arange(0,2*spreadF*len(all_data.keys()),2*spreadF),all_data.keys(),rotation=15,fontsize=6)
plt.ylabel("log(MNV)")

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin.%s' %(fig_format)))
    
#%% Apply full model p-val mask if desired

use_fullmodel_mask = True
use_loc_mask = False #True #if true, mask out voxels with non-significant tgt-sur contrast
pthresh_fullmodel = 0.01
pthresh_loc = 0.01

#initialize significance mask
for k_i, key in enumerate(all_data.keys()):
    all_data[key]['sig'] = (np.ones(len(all_data[key])) == 1)
    
#full model mask
if use_fullmodel_mask:
    for k_i, key in enumerate(all_data.keys()):
        df = all_data[key]
        pvals = df['task p-val']
        pval_mask = pvals < pthresh_fullmodel
        print("%d/%d voxels survive full model p-val mask" %(np.sum(pval_mask),np.size(pval_mask)))
        mask_dict[key] = mask_dict[key] & pval_mask   
        all_data[key]['sig'] = (all_data[key]['sig'] & pval_mask)

#loc mask
if use_loc_mask:
    for k_i, key in enumerate(all_data.keys()):
        df = all_data[key]
        pvals = df['loc p-val']
        pval_mask = pvals < pthresh_loc
        print("%d/%d voxels survive loc p-val mask" %(np.sum(pval_mask),np.size(pval_mask)))
        mask_dict[key] = mask_dict[key] & pval_mask   
        all_data[key]['sig'] = (all_data[key]['sig'] & pval_mask)

#%% Plot Depth Profiles
    
statDetails = {'labels': ['sur', 'iso0', 'iso90', 'orth', 
                          'ctr_unwarp', 'sur_unwarp', 'ctr-sur_unwarp',
                          'V23_superficial_deveined_orth','V23_middle_deveined_orth', 'V23_deep_deveined_orth',
                          'V23_superficial_deveined_iso90', 'V23_middle_deveined_iso90', 'V23_deep_deveined_iso90',
                          'V23_superficial_deveined_iso0', 'V23_middle_deveined_iso0', 'V23_deep_deveined_iso0',
                          'V23_superficial_deveined_sur', 'V23_middle_deveined_sur', 'V23_deep_deveined_sur'],
                'colors': [[.7, .7, .7], 'red', 'darkviolet', 'orange',
                          'gold', 'purple', 'coral', 
                          'orange', 'orange', 'orange', 
                          'darkviolet', 'darkviolet', 'darkviolet', 
                          'red', 'red', 'red', 
                          'gray', 'gray', 'gray']}
diffDetails = {}
diffDetails['statIDs'] = {'odss': ['orth','iso90'],
                          'fgm': ['iso90','iso0'],
                          'dsi': ['orth','iso0'],
                          'iso-sur': ['iso0','sur'],
                          'ctr-sur': ['ctr_unwarp','sur_unwarp'],
                          'iso90-sur': ['iso90','sur'],
                          'orth-sur': ['orth','sur'],
                          'odss_gPPI_superficial_V23':['V23_superficial_deveined_orth', 'V23_superficial_deveined_iso90'],
                          'odss_gPPI_middle_V23':['V23_middle_deveined_orth','V23_middle_deveined_iso90'],
                          'odss_gPPI_deep_V23':['V23_deep_deveined_orth','V23_deep_deveined_iso90'],
                          'fgm_gPPI_superficial_V23':['V23_superficial_deveined_iso90','V23_superficial_deveined_iso0'],
                          'fgm_gPPI_middle_V23':['V23_middle_deveined_iso90','V23_middle_deveined_iso0'],
                          'fgm_gPPI_deep_V23':['V23_deep_deveined_iso90','V23_deep_deveined_iso0'],
                          'iso-sur_gPPI_superficial_V23':['V23_superficial_deveined_iso0','V23_superficial_deveined_sur'],
                          'iso-sur_gPPI_middle_V23':['V23_middle_deveined_iso0','V23_middle_deveined_sur'],
                          'iso-sur_gPPI_deep_V23':['V23_deep_deveined_iso0','V23_deep_deveined_sur']
                          }
diffDetails['colors'] = ['green','magenta','cyan','black','coral','darkmagenta','darkturquoise',[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0],[1,0,1],[0.7,0,0.7],[0.5,0,0.5],[0.7,0.7,0.7],[0.5,0.5,0.5],[0.3,0.3,0.3]]
profile_method = 'bin' # bin or smooth
#pick out ROIs where we're sure of localization
for key in all_data.keys():
    df = all_data[key]
    all_data[key]['in_tgt'] = df['scale_xy_dist'] < roiRad
    all_data[key]['in_ctr'] = df['scale_xy_dist'] < centerRad
    all_data[key]['in_bor'] = (df['scale_xy_dist'] >= borderRad[0]) & (df['scale_xy_dist'] < borderRad[1])
    if len(surRad) == 1:
        sur_mask = df['scale_xy_dist'] > surRad[0]
    elif len(surRad) > 1:
        sur_mask = ((df['scale_xy_dist'] >= surRad[0]) & (df['scale_xy_dist'] <= surRad[1]))
    all_data[key]['in_sur'] = sur_mask
    
useSI = False #use suppression index rather than differences (cond1 - cond2 / cond1 + cond2)

#create full masks
masks = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}}
for roi in masks.keys():
    masks[roi] = {key:all_data[key][roi]*all_data[key]['sig']*all_data[key]['no_vein'] for key in all_data.keys()}
    
#now compute depth profiles
depthProfiles = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}}
diffProfiles = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}}
for roi in depthProfiles.keys():
    if roi == 'in_tgt':
        nD = nDepths
    else:
        nD = nDepths_rings
    depthProfiles[roi] = compute_all_depth_profiles(all_data,statDetails,profile_method,nD,masks[roi],depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
    diffProfiles[roi] = compute_diff_profiles(all_data,statDetails,diffDetails['statIDs'],profile_method,nD,useSI,masks[roi],depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
      
#%% Centroid plots
# Let's take a look at raw voxel betas across depth by condition

Nsubj = len(all_data.keys())

#plot centroids for each condition and ROI
plot_centroids(all_data, masks['in_tgt'], statDetails, roiRad, nDepths=nDepths)
        
#calculate difference profiles
plot_centroids_diff(all_data, masks['in_tgt'], statDetails, diffDetails, roiRad, nDepths)
          
if savefigs:
    for l in statDetails['labels']:
        plt.figure(l)
        plt.savefig(os.path.join(figDir,'centroids_%s.%s' %(l,fig_format)))
    for l in diffDetails['statIDs'].keys():
        plt.figure(l)
        plt.savefig(os.path.join(figDir,'centroids_%s.%s' %(l,fig_format)))
        
#%% Individual Subjects Histogram

radParam = 'scale_xy_dist'
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

for roi in depthProfiles.keys():
    
    if roi == 'in_tgt':
        nD = nDepths
    else:
        nD = nDepths_rings
        
    dP = depthProfiles[roi]
    diffP = diffProfiles[roi]
    
    #reformat data to fit decon_rois specs
    keep_rois = np.zeros((Nsubj,len(statDetails['labels']),nD))
    for iR, roiID in enumerate(all_data.keys()):
        for iStat, stat in enumerate(statDetails['labels']):
            keep_rois[iR,iStat,:] = dP[stat]['avg'][iR]
            
    keep_diffs = np.zeros((Nsubj,len(diffDetails['statIDs'].keys()),nD))
    for iR, roiID in enumerate(all_data.keys()):
        for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
            keep_diffs[iR,iDiff,:] = diffP[diff]['avg'][iR]
    
    #define point spread function
    p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
    Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
    Nbins = nD #number of bins to use in this analysis
    
    normalize_psf = False #True if you want to normalize the psf by the deepest layer  
    
    decon_rois = depth_deconv(keep_rois,p2t_model,Nbins_model,Nbins,normalize_psf)
    decon_diffs = depth_deconv(keep_diffs,p2t_model,Nbins_model,Nbins,normalize_psf)
    
    #now put back in dictionary
    for iStat, stat in enumerate(statDetails['labels']):
        depthProfiles[roi][stat]['avg_decon'] = np.squeeze(np.array(decon_rois)[:,iStat,:])
        
    for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
        diffProfiles[roi][diff]['avg_decon'] = np.squeeze(np.array(decon_diffs)[:,iDiff,:])

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

# Different stats to use
taskStats = statDetails['labels'][0:4]
taskColors = statDetails['colors'][0:4]
taskDiffs = list(diffDetails['statIDs'].keys())[0:4]
locStats = statDetails['labels'][4:]
locColors = statDetails['colors'][4:]
locDiffs = list(diffDetails['statIDs'].keys())[4:5]
#otherDiffs = list(diffDetails['statIDs'].keys())[5:]
otherDiffs = list(diffDetails['statIDs'].keys())[5:7]
gPPIStats = statDetails['labels'][7:]
gPPIColors = statDetails['colors'][7:]
gPPIDiffs = list(diffDetails['statIDs'].keys())[7:]

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
avgGPPIRadialProfiles = {}
avgGPPIRadialDiff = {}
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
    avgGPPIRadialProfiles[l], avgGPPIRadialDiff[l] = compute_avg_rad_profile(radialProfiles[l],statDetails,diffDetails['statIDs'],gPPIStats,gPPIDiffs,prop_err,useSI)

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
use_decon = True
useSI = False

all_pvals = np.array([]) #initialize a 1D array that will contain all pvals
pvals_lookup = {} #initialize a dictionary that will keep track of which p-vals correspond to which comparison
iter_var = 0

for roi_type in depthProfiles.keys():
    pvals_lookup[roi_type] = {}
    
    if roi_type == 'in_tgt':
        nD = nDepths
    else:
        nD = nDepths_rings
    
    [avgTaskProfiles, avgTaskDiffs] = compute_avg_depth_profile(depthProfiles[roi_type],statDetails,diffDetails['statIDs'],taskStats,taskDiffs,use_decon,prop_err,useSI)  
    [avgLocProfiles, avgLocDiffs] = compute_avg_depth_profile(depthProfiles[roi_type],statDetails,diffDetails['statIDs'],locStats,locDiffs,use_decon,prop_err,useSI)
    [tmp , avgOtherDiffs] = compute_avg_depth_profile(depthProfiles[roi_type],statDetails,diffDetails['statIDs'],taskStats,otherDiffs,use_decon,prop_err,useSI)
    
    # 1-sample t-tests
    # Depth-dependent task contrasts
    for c_i, c in enumerate(avgTaskDiffs.keys()):
        all_pvals = np.append(all_pvals,avgTaskDiffs[c]['p-vals'].pvalue)
        n_pvals = len(avgTaskDiffs[c]['p-vals'].pvalue)
        pvals_lookup[roi_type][c] = [iter_var,iter_var+n_pvals]
        iter_var += n_pvals
    # Depth-dependent localizer contrasts
    for c_i, c in enumerate(avgLocDiffs.keys()):
        all_pvals = np.append(all_pvals,avgLocDiffs[c]['p-vals'].pvalue)
        pvals_lookup[roi_type][c] = [iter_var,iter_var+n_pvals]
        iter_var += n_pvals
    # for c_i, c in enumerate(avgOtherDiffs.keys()):
    #     all_pvals = np.append(all_pvals,avgOtherDiffs[c]['p-vals'].pvalue)
    #     pvals_lookup[c] = [cond_iter*nD,cond_iter*nD+nD]
    #     iter_var += 1
        
    # 2-way ANOVA
    diffProfiles_list = {}
    for cond in diffProfiles[roi_type].keys():
        diffProfiles_list[cond] = diffProfiles[roi_type][cond]['avg_decon']
        
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
    
# Do the same thing for radial profiles
for roi_type in avgTaskRadialDiff.keys():
    pvals_lookup[roi_type] = {}
    
    # Radial task contrasts
    for c_i, c in enumerate(avgTaskRadialDiff[roi_type].keys()):
        all_pvals = np.append(all_pvals,avgTaskRadialDiff[roi_type][c]['p-vals'].pvalue)
        n_pvals = len(avgTaskRadialDiff[roi_type][c]['p-vals'].pvalue)
        pvals_lookup[roi_type][c] = [iter_var,iter_var+n_pvals]
        iter_var += n_pvals
    for c_i, c in enumerate(avgLocRadialDiff[roi_type].keys()):
        all_pvals = np.append(all_pvals,avgLocRadialDiff[roi_type][c]['p-vals'].pvalue)
        n_pvals = len(avgLocRadialDiff[roi_type][c]['p-vals'].pvalue)
        pvals_lookup[roi_type][c] = [iter_var,iter_var+n_pvals]
        iter_var += n_pvals
    for c_i, c in enumerate(avgGPPIRadialDiff[roi_type].keys()):
        all_pvals = np.append(all_pvals,avgGPPIRadialDiff[roi_type][c]['p-vals'].pvalue)
        n_pvals = len(avgGPPIRadialDiff[roi_type][c]['p-vals'].pvalue)
        pvals_lookup[roi_type][c] = [iter_var,iter_var+n_pvals]
        iter_var += n_pvals

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
all_roiTypes = list(depthProfiles.keys()) + list(avgTaskRadialDiff.keys())
all_pvals_dict = {roi_type: {} for roi_type in all_roiTypes}

# Repackage all of the corrected p-values into a new dictionary
for roi_type in pvals_lookup:
    for key in pvals_lookup[roi_type]:
        if isinstance(pvals_lookup[roi_type][key],dict):
            all_pvals_dict[roi_type][key] = {}
            for combo in pvals_lookup[key]:
                all_pvals_dict[roi_type][key][combo] = all_pvals_corrected[pvals_lookup[roi_type][key][combo]]
        else:
            all_pvals_dict[roi_type][key] = all_pvals_corrected[pvals_lookup[roi_type][key][0]:pvals_lookup[roi_type][key][1]]


#%% now make some average plots

roi_type = 'in_tgt' #the type of ROI to examine

[avgTaskProfiles, avgTaskDiffs] = compute_avg_depth_profile(depthProfiles[roi_type],statDetails,diffDetails['statIDs'],taskStats,taskDiffs,use_decon,prop_err,useSI)  
[avgLocProfiles, avgLocDiffs] = compute_avg_depth_profile(depthProfiles[roi_type],statDetails,diffDetails['statIDs'],locStats,locDiffs,use_decon,prop_err,useSI)
#[tmp , avgOtherDiffs] = compute_avg_depth_profile(depthProfiles[roi_type],statDetails,diffDetails['statIDs'],taskStats,otherDiffs,use_decon,prop_err,useSI)

# Save Task Profiles to CSV
for diff, data in avgTaskDiffs.items():
    depth_bins = np.arange(len(data['avg']))  # Create depth bins
    df = pd.DataFrame({
        'depth bin': depth_bins,
        'avg': data['avg'],
        'stdev': data['stdev'],
        'norm_depths': data['norm_depths'],
        't-statistic': data['p-vals'].statistic,
        'p-value': data['p-vals'].pvalue,
        'p-value corrected': all_pvals_dict[roi_type][diff],
        'df': data['p-vals'].df,
        'N': [data['Nsamp']] * len(data['avg'])  # Repeat the Nsamp value
    })
    df.to_csv(os.path.join(figDir,f"{diff}_{roi_type}.csv"), index=False)
    
for diff, data in avgLocDiffs.items():
    depth_bins = np.arange(len(data['avg']))  # Create depth bins
    df = pd.DataFrame({
        'depth bin': depth_bins,
        'avg': data['avg'],
        'stdev': data['stdev'],
        'norm_depths': data['norm_depths'],
        't-statistic': data['p-vals'].statistic,
        'p-value': data['p-vals'].pvalue,
        'p-value corrected': all_pvals_dict[roi_type][diff],
        'df': data['p-vals'].df,
        'N': [data['Nsamp']] * len(data['avg'])  # Repeat the Nsamp value
    })
    df.to_csv(os.path.join(figDir,f"{diff}_{roi_type}.csv"), index=False)

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
xlim = [-0.3,1.8]
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
xlim = [-0.3,1.8]
    
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
pvals_task = {cond:all_pvals_dict[roi_type][cond] for cond in avgTaskDiffs.keys()}
pvals_loc = {cond:all_pvals_dict[roi_type][cond] for cond in avgLocDiffs.keys()}
# pvals_other = {cond:all_pvals_dict[cond] for cond in avgOtherDiffs.keys()}

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
xlim = [-0.5,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task)

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
xlim = [-0.5,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['dsi'],['tab:cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task)

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
xlim = [-0.5,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task)

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
xlim = [-0.5,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['odss'],['tab:green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task)

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
xlim = [-0.5,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgLocDiffs,['ctr-sur'],['coral'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_loc)


if savefigs:
    if use_decon:
        fig.savefig(os.path.join(figDir,'avg_profiles_ctr_sur_deconv.%s' %(fig_format)))
    else:
        fig.savefig(os.path.join(figDir,'avg_profiles_ctr_sur.%s' %(fig_format)))
        
# #iso90 and sur  
# fig = plt.figure(figsize=(6, 4))
# fig.set_size_inches((6,4))
# fig.patch.set_facecolor(fcolor)

# fig.clf()
# fsize = 14

# p1 = fig.add_axes([.15, .2, .3, .7])
# fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
# ylim = [-0.02,1.02]
# xlim = [0,6]
# Ntext = [4,0.05]
# plot_avg_depth_profile(p1,avgTaskProfiles,['iso90','sur'],['purple','gray'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

# #iso90-sur
# xlim = [-0.8,1.5]
# p2 = fig.add_axes([.7, .2, .25, .7])
# fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
# plot_avg_diff_profile(p2,avgOtherDiffs,['iso90-sur'],['darkmagenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_other)

# if savefigs:
#     if use_decon:
#         fig.savefig(os.path.join(figDir,'avg_profiles_iso90-sur_deconv.%s' %(fig_format)))
#     else:
#         fig.savefig(os.path.join(figDir,'avg_profiles_iso90-sur.%s' %(fig_format)))

# #orth and sur  
# fig = plt.figure(figsize=(6, 4))
# fig.set_size_inches((6,4))
# fig.patch.set_facecolor(fcolor)

# fig.clf()
# fsize = 14

# p1 = fig.add_axes([.15, .2, .3, .7])
# fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
# ylim = [-0.02,1.02]
# xlim = [0,6]
# Ntext = [4,0.05]
# plot_avg_depth_profile(p1,avgTaskProfiles,['orth','sur'],['orange','gray'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

# #orth-sur
# xlim = [-0.8,1.5]
# p2 = fig.add_axes([.7, .2, .25, .7])
# fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
# plot_avg_diff_profile(p2,avgOtherDiffs,['orth-sur'],['darkturquoise'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_other)

# if savefigs:
#     if use_decon:
#         fig.savefig(os.path.join(figDir,'avg_profiles_orth-sur_deconv.%s' %(fig_format)))
#     else:
#         fig.savefig(os.path.join(figDir,'avg_profiles_orth-sur.%s' %(fig_format)))
    
#%% Make condition depth profiles with individual subject data overlaid

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [0,7]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['sur'],['gray'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles['in_tgt'])

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
xlim = [0,7]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['iso0'],['red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles['in_tgt'])

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
xlim = [0,7]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['iso90'],['darkviolet'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles['in_tgt'])

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
xlim = [0,7]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgTaskProfiles,['orth'],['orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles['in_tgt'])

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
xlim = [-3,7]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgLocProfiles,['ctr_unwarp'],['gold'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles['in_tgt'])

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
xlim = [-3,7]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,avgLocProfiles,['sur_unwarp'],['purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles['in_tgt'])

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
plot_avg_diff_profile(p2,avgLocDiffs,['ctr-sur'],['coral'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles['in_tgt'],showSig=True,pthresh=pthresh,statCorrType=pvals_loc)

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
plot_avg_diff_profile(p2,avgTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles['in_tgt'],showSig=True,pthresh=pthresh,statCorrType=pvals_task)

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
plot_avg_diff_profile(p2,avgTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles['in_tgt'],showSig=True,pthresh=pthresh,statCorrType=pvals_task)

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
plot_avg_diff_profile(p2,avgTaskDiffs,['odss'],['green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles['in_tgt'],showSig=True,pthresh=pthresh,statCorrType=pvals_task)

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
plot_avg_diff_profile(p2,avgTaskDiffs,['dsi'],['cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles['in_tgt'],showSig=True,pthresh=pthresh,statCorrType=pvals_task)

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_dsi.%s' %(fig_format)))
    
#%% Does FGM Appear at Different Layers than OTSS?

# 2-sample paired t-test
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
plot_avg_diff_profile(p2,avgTaskDiffs,['fgm','odss'],['magenta','green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI)

# Show significance
top = avgTaskDiffs['fgm']['avg'] + avgTaskDiffs['fgm']['stdev']/np.sqrt(np.shape(avgTaskDiffs['fgm']['avg'])[0])
corrected_pvalues = all_pvals_dict['in_tgt']['fgmvsodss']
p2.plot(top[corrected_pvalues <= pthresh] + 0.1,avgTaskDiffs['fgm']['norm_depths'][corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None')
p2.plot()
   
#%% Loc Profiles across surface!
# smoothing according to line 441 in analysisROI_dev

# Define Gaussian smoothing kernel
def gaussian_kernel(x, k, xloc):
    ''' A Gaussian kernel
    
    Inputs:
        x (array): coordinates
        k (float): standard dev.
        xloc (array): coordinates to center kernel
    
    Outputs:
        kernel (array): normalized exponential kernel
        
    Dependencies:
        numpy
    '''
    kernel = (1/(np.sqrt(2*np.pi)*k))*np.exp(-((x-xloc)**2)/(2*k**2))
    return kernel/np.sum(kernel)

#with smoothing
plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(14, 7))
fontsize = 8
fig1.patch.set_facecolor(fcolor)
locStatDetails = {'labels':['ctr-sur_unwarp'], 'colors':['black']}

kernel = smooth_kernel
#smooth_factor = 0.4/(2*np.sqrt(2*np.log(2))) #This is the st. dev. for a FWHM = 0.4 sigma which is about 0.5 mm for an ROI with a diameter of 5 mm
smooth_factor = 0.3 #This is the smooth factor for the exponential kernel. It roughly corresponds to a 1 mm FWHM on the cortical surface for an ROI with a diameter of 5 mm.
radMax = 4
nRadii = 20
ymax = 5
ymin = -5
highlight = False #['ctr-sur'] #['orth', 'iso90']#'orth'
depth_labels = ['deep', 'middle', 'superficial']
depthBoundaries = np.array([[0,0.333],[0.333,0.666],[0.666,1]])
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
            x = df['scale_xy_dist'].values #use distance in mm for smoothing
            
            #smooth data
            coef_smooth, x_smooth = smoothen(coef, x, kernel=kernel, smooth_factor=smooth_factor, radMax=radMax)
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

_#%% Average Stat Porfiles with individual profiles overlaid

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
locStatDetails = {'labels':['ctr-sur_unwarp','iso0','iso90','orth','sur'],
                  'colors':['black','red','purple','orange','gray',
                            'green', 'green', 'green',
                            'magenta','magenta','magenta',
                            'gray','gray','gray']}

radMax = 4
nRadii = 20
ymax = 14
ymin = -6
highlight = False #['ctr-sur'] #['orth', 'iso90']#'orth'
depth_labels = ['deep', 'middle', 'superficial']
depthBoundaries = np.array([[0,0.333],[0.333,0.666],[0.666,1]])
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
            coef_smooth, x_smooth = smoothen(coef, x, kernel=kernel, smooth_factor=smooth_factor, radMax=radMax)
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
#diffStatDetails = {'labels':['fgm','odss','iso-sur'], 'colors':['magenta','green','gray']}
diffStatDetails = {'labels':['fgm','odss','iso-sur',
                        'odss_gPPI_superficial_V23','odss_gPPI_middle_V23','odss_gPPI_deep_V23',
                        'fgm_gPPI_superficial_V23','fgm_gPPI_middle_V23','fgm_gPPI_deep_V23',
                        'iso-sur_gPPI_superficial_V23','iso-sur_gPPI_middle_V23','iso-sur_gPPI_deep_V23'],
                   'colors':['magenta','green','gray',
                             [0.1,1,0],[0.1,0.7,0],[0.1,0.5,0],
                             [1,0,1],[0.7,0,0.7],[0.5,0,0.5],
                             [0.7,0.7,0.7],[0.5,0.5,0.5],[0.3,0.3,0.3]]
                   }

radMax = 4
nRadii = 20
ymax = 10#3
ymin = -10#-3
highlight = False #['ctr-sur'] #['orth', 'iso90']#'orth'

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
            coef_smooth, x_smooth = smoothen(coef, x, kernel=kernel, smooth_factor=smooth_factor, radMax=radMax)
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
    
#%% Average Loc Profiles with individual profiles overlaid

fig = plt.figure(figsize=(5, 8))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12
colors = ['green']
plotStats = ['odss_gPPI_deep_V23']
plot_indiv = True #if true, overlay individiual traces
ymin = -12
ymax = 12
ymin_bar = -3.0
ymax_bar = 5.0

#radialDiff = avgTaskRadialDiff
radialDiff = avgGPPIRadialDiff

for iD, depth_label in enumerate(['deep','middle','superficial']):
    print(depth_label)
    p = fig.add_axes([.15, .22 + iD*.3, .7, .15])
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
        p.set_title(depth_label)
        
        if plot_indiv:
            p.plot(np.tile(x_smooth,[np.shape(all_profiles[depth_label][label])[0],1]).T, np.array(all_profiles[depth_label][label]).T, color=colors[iStat], alpha=0.2)
        

        if iD == 0 and not showSig:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
            p.legend()
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([ymin, ymax])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
        if showSig:
            #also plot bins to show size
            p.vlines(radBins,-1*np.ones([nRad+1,]),2*np.ones([nRad+1,]),linestyle='dotted',alpha=0.3)
            
            p = fig.add_axes([.15, .12 + iD*.3, .7, .05])
            fix_axes(p, lcolor=lcolor, fcolor=fcolor)
            
            if statCorrType == 'none':
                top = 1.7*np.ones(nRad)
                p.plot(radBinCtrs[radialDiff[depth_label][label]['p-vals'].pvalue <= pthresh],top[radialDiff[depth_label][label]['p-vals'].pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
            else:
                top = 1.7*np.ones(nRad)
                # corrected_pvalues = multipletests(radialDiff[depth_label][label]['p-vals'].pvalue,method=statCorrType)[1]
                corrected_pvalues = all_pvals_dict[depth_label][label]
                p.plot(radBinCtrs[corrected_pvalues <= pthresh],top[corrected_pvalues <= pthresh],color='k',marker='$*$',linestyle='None')
            
            #plot average in each bin
            p.bar(radBinCtrs,radialDiff[depth_label][label]['avg'],width=0.9,facecolor=colors[iStat])
            p.errorbar(radBinCtrs,radialDiff[depth_label][label]['avg'],radialDiff[depth_label][label]['stdev']/np.sqrt(radialDiff[depth_label][label]['Nsamp']),marker='None',linestyle='None',color='k')
            p.set_ylim([ymin_bar,ymax_bar])
            
            if iD == 0:
                p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
            else:
                p.set_xticklabels([])
            
        if savefigs:
            fig.savefig(os.path.join(figDir,"%s_average_radial_difference_profiles_wStats.%s" %(label,fig_format)))
        
if savefigs:
    fig.savefig(os.path.join(figDir,"average_indiv_radial_profiles_%s.%s" %(plotStats,fig_format)))
  
#%% For each condition, create nDepthsxnDepths grid

nDepths_gPPI = 3
roi = 'in_tgt'
use_decon = False
depthProfiles_gPPI = compute_all_depth_profiles(all_data,statDetails,profile_method,nDepths_gPPI,masks[roi],depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
[avgGPPIProfiles, avgGPPIDiffs] = compute_avg_depth_profile(depthProfiles_gPPI,statDetails,diffDetails['statIDs'],gPPIStats,gPPIDiffs,use_decon,prop_err,useSI)  

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
    fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(1,5)
else:
    fig, ((ax2, ax3, ax4, ax5)) = plt.subplots(1,4)
fig.set_figwidth(14)
fig.set_figheight(6)
fontsize=12

#Get all corrected p-values
all_corrected_gPPI_pvalues = {}
if use_all_meas:
    if plot_baseline:
        V1_condition_list = gPPIStats + gPPIDiffs
    else:
        V1_condition_list = ['V23_deep_deveined_iso0',
               'V23_middle_deveined_iso0', 'V23_superficial_deveined_iso0',
               'V23_deep_deveined_iso90', 'V23_middle_deveined_iso90',
               'V23_superficial_deveined_iso90', 'V23_deep_deveined_orth',
               'V23_middle_deveined_orth', 'V23_superficial_deveined_orth',
               'V23_deep_deveined_sur', 'V23_middle_deveined_sur',
               'V23_superficial_deveined_sur']
        V1_diff_list = ['odss_gPPI_superficial_V23',
                 'odss_gPPI_middle_V23',
                 'odss_gPPI_deep_V23',
                 'fgm_gPPI_superficial_V23',
                 'fgm_gPPI_middle_V23',
                 'fgm_gPPI_deep_V23',
                 'iso-sur_gPPI_superficial_V23',
                 'iso-sur_gPPI_middle_V23',
                 'iso-sur_gPPI_deep_V23']
    all_gPPI_pvalues = np.concatenate([
        np.array([avgGPPIProfiles[c]['p-vals'].pvalue for c in V1_condition_list]),
        np.array([avgGPPIDiffs[c]['p-vals'].pvalue for c in V1_diff_list])
    ])
    condition_list = V1_condition_list
    diff_list = V1_diff_list
    corrected_list = multipletests(all_gPPI_pvalues.flatten(),method=statCorrType)[1]
    for c_i, c in enumerate(condition_list):
        all_corrected_gPPI_pvalues[c] = corrected_list[c_i*nDepths_gPPI:c_i*nDepths_gPPI+nDepths_gPPI]
    for c_i, c in enumerate(diff_list):
        all_corrected_gPPI_pvalues[c] = corrected_list[c_i*nDepths_gPPI:c_i*nDepths_gPPI+nDepths_gPPI]

if plot_baseline:
    # V2/3 -> V1
    data_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
    p_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
    p_thresh = 0.05
    condition_list = ['V23_deep_deveined','V23_middle_deveined','V23_superficial_deveined']
    if use_all_meas:
        for c_i, c in enumerate(condition_list):
            data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
            p_mat[:,c_i] = all_corrected_gPPI_pvalues[c]
    else:
        for c_i, c in enumerate(condition_list):
            data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
            corrected_pvalues = multipletests(avgGPPIProfiles[c]['p-vals'].pvalue,method=statCorrType)[1]
            p_mat[:,c_i] = corrected_pvalues
        
    plot_gPPI_mat(data_mat,p_mat,'V2/3','V1',seed='V2/3',target='V1',fig=fig,ax=ax1,cbar=False)

    # if savefigs:
    #     fig.savefig(os.path.join(figDir,'V23->V1_mat.%s' %figType),format=figType)
    
# V2/3 -> V1 iso0
data_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_iso0','V23_middle_deveined_iso0','V23_superficial_deveined_iso0']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
        p_mat[:,c_i] = all_corrected_gPPI_pvalues[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
        corrected_pvalues = multipletests(avgGPPIProfiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso0','V1 iso0',seed='V2/3 iso0',target='V1 iso0',fig=fig,ax=ax2,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_iso0_mat'))

# V2/3 -> V1 iso90
data_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_iso90','V23_middle_deveined_iso90','V23_superficial_deveined_iso90']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
        p_mat[:,c_i] = all_corrected_gPPI_pvalues[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
        corrected_pvalues = multipletests(avgGPPIProfiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso90','V1 iso90',seed='V2/3 iso90',target='V1 iso90',fig=fig,ax=ax3,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_iso90_mat'))

# V2/3 -> V1 orth
data_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_orth','V23_middle_deveined_orth','V23_superficial_deveined_orth']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
        p_mat[:,c_i] = all_corrected_gPPI_pvalues[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
        corrected_pvalues = multipletests(avgGPPIProfiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 orth','V1 orth',seed='V2/3 orth',target='V1 orth',fig=fig,ax=ax4,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_orth_mat'))

# V2/3 -> V1 sur
data_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_thresh = 0.05
condition_list = ['V23_deep_deveined_sur','V23_middle_deveined_sur','V23_superficial_deveined_sur']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
        p_mat[:,c_i] = all_corrected_gPPI_pvalues[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIProfiles[c]['avg']
        corrected_pvalues = multipletests(avgGPPIProfiles[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
heatmap, fig, ax = plot_gPPI_mat(data_mat,p_mat,'V2/3 sur','V1 sur',seed='V2/3 sur',target='V1 sur',fig=fig,ax=ax5,cbar=False)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_sur_mat'))

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(heatmap, cax=cbar_ax)
cbar.set_label('Colorbar Label')
cbar.set_label("% BOLD Change",fontsize=0.6*fontsize)
cbar.ax.tick_params(labelsize=0.6*fontsize)

plt.tight_layout(rect=(0,0,0.9,1))

if savefigs:
    fig.savefig(os.path.join(figDir,'gPPI_conditions_mat.%s' %fig_format),format=fig_format)

#%% Now do the same thing for contrast conditions

cbar_lims = [-4,4]

#create plot
use_all_meas = True #True if you would like to use all measurements in your correction for multiple comparisons


fig, ((ax1, ax2, ax3)) = plt.subplots(1,3)
fig.set_figwidth(14)
fig.set_figheight(6)
fontsize=12

#Get all corrected p-values
# all_corrected_diff_pvalues = {"V1":{},"V23":{}}
# if use_all_meas:
#     V1_condition_list = list(diffDetails_V1tgt['statIDs'].keys())
#     V23_condition_list = list(diffDetails_V23tgt['statIDs'].keys())
#     all_diff_pvalues = np.concatenate(([avgGPPIDiffs[c]['p-vals'].pvalue for c in V1_condition_list],[avgV23Diffs[c]['p-vals'].pvalue for c in V23_condition_list])) #get all p-values for all measurements in one list
#     condition_list = np.concatenate((V1_condition_list,V23_condition_list))
#     corrected_list = multipletests(all_diff_pvalues.flatten(),method=statCorrType)[1]
#     for c_i, c in enumerate(condition_list):
#         if 'V1' in c:
#             all_corrected_diff_pvalues['V1'][c] = corrected_list[c_i:c_i+nDepths]
#         if 'V23' in c:
#             all_corrected_diff_pvalues['V23'][c] = corrected_list[c_i:c_i+nDepths]

# V2/3 -> V1 iso0 - sur
data_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_thresh = 0.05
condition_list = ['iso-sur_gPPI_deep_V23','iso-sur_gPPI_middle_V23','iso-sur_gPPI_superficial_V23']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIDiffs[c]['avg']
        p_mat[:,c_i] = all_corrected_gPPI_pvalues[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIDiffs[c]['avg']
        top = avgGPPIDiffs[c]['avg'] + avgGPPIDiffs[c]['stdev']/np.sqrt(np.shape(avgGPPIDiffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgGPPIDiffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 iso-sur','V1 iso-sur',seed='V2/3 iso-sur',target='V1 iso-sur',fig=fig,ax=ax1,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_iso-sur_mat.%s' %figType),format=figType)

# V2/3 -> V1 odss
data_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_thresh = 0.05
condition_list = ['odss_gPPI_deep_V23','odss_gPPI_middle_V23','odss_gPPI_superficial_V23']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIDiffs[c]['avg']
        p_mat[:,c_i] = all_corrected_gPPI_pvalues[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIDiffs[c]['avg']
        top = avgGPPIDiffs[c]['avg'] + avgGPPIDiffs[c]['stdev']/np.sqrt(np.shape(avgGPPIDiffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgGPPIDiffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
plot_gPPI_mat(data_mat,p_mat,'V2/3 ODSS','V1 ODSS',seed='V2/3 ODSS',target='V1 ODSS',fig=fig,ax=ax2,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_odss_mat.%s' %figType),format=figType)

# V2/3 -> V1 fgm
data_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_mat = np.zeros([nDepths_gPPI,nDepths_gPPI])
p_thresh = 0.05
condition_list = ['fgm_gPPI_deep_V23','fgm_gPPI_middle_V23','fgm_gPPI_superficial_V23']
if use_all_meas:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIDiffs[c]['avg']
        p_mat[:,c_i] = all_corrected_gPPI_pvalues[c]
else:
    for c_i, c in enumerate(condition_list):
        data_mat[:,c_i] = avgGPPIDiffs[c]['avg']
        top = avgGPPIDiffs[c]['avg'] + avgGPPIDiffs[c]['stdev']/np.sqrt(np.shape(avgGPPIDiffs[c]['avg'])[0])
        corrected_pvalues = multipletests(avgGPPIDiffs[c]['p-vals'].pvalue,method=statCorrType)[1]
        p_mat[:,c_i] = corrected_pvalues
    
heatmap, fig, ax = plot_gPPI_mat(data_mat,p_mat,'V2/3 FGM','V1 FGM',seed='V2/3 FGM',target='V1 FGM',fig=fig,ax=ax3,cbar=False,cbar_lims=cbar_lims)

# if savefigs:
#     fig.savefig(os.path.join(figDir,'V23->V1_fgm_mat.%s' %figType),format=figType)
    
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
cbar = fig.colorbar(heatmap, cax=cbar_ax)
cbar.set_label('Colorbar Label')
cbar.set_label("% BOLD Change",fontsize=0.6*fontsize)
cbar.ax.tick_params(labelsize=0.6*fontsize)

plt.tight_layout(rect=(0,0,0.9,1))

if savefigs:
    fig.savefig(os.path.join(figDir,'gPPI_diffs_mat.%s' %fig_format),format=fig_format)
    
#%% Plot each difference profile separately and show statistics
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
    
#%% RINGS Depth Histograms
ring_labels = ['in_ctr', 'in_bor', 'in_sur']
figDir = os.path.join(mainDir,'figs/subjAvg/rings')

for ring in ring_labels:
    cond = list(depthProfiles[ring].keys())[0]
    fdhist = plt.figure(figsize=(15,5))
    for iR, label in enumerate(all_data.keys()):
        plt.subplot(2,int(np.ceil(len(datasets)/2)),iR+1)
        plt.bar(np.linspace(0,1,nDepths_rings),height=depthProfiles[ring][cond]['N'][iR],width=(1+1/nDepths_rings)/nDepths_rings)
        plt.title(label,fontsize=10)
        plt.xlabel("Normalize Depth WM -> GM")
        plt.ylabel("Num. Voxels")
        plt.legend(['N='+str(len(roi)),], fontsize = 6)
    plt.suptitle(ring)
    fdhist.tight_layout(pad=0.0)

#%% RINGS now make some average plots

prop_err = False # do error propagation?
use_decon = True
useSI = False

for roi_type in ring_labels:
    taskStats = statDetails['labels'][0:4]
    taskColors = statDetails['colors'][0:4]
    taskDiffs = list(diffDetails['statIDs'].keys())[0:4]
    locStats = statDetails['labels'][4:]
    locColors = statDetails['colors'][4:]
    locDiffs = list(diffDetails['statIDs'].keys())[4:5]
    otherDiffs = list(diffDetails['statIDs'].keys())[5:]
    
    [avgTaskProfiles, avgTaskDiffs] = compute_avg_depth_profile(depthProfiles[roi_type],statDetails,diffDetails['statIDs'],taskStats,taskDiffs,use_decon,prop_err,useSI)  
    [avgLocProfiles, avgLocDiffs] = compute_avg_depth_profile(depthProfiles[roi_type],statDetails,diffDetails['statIDs'],locStats,locDiffs,use_decon,prop_err,useSI)
    [tmp , avgOtherDiffs] = compute_avg_depth_profile(depthProfiles[roi_type],statDetails,diffDetails['statIDs'],taskStats,otherDiffs,use_decon,prop_err,useSI)
    
    # Save Task Profiles to CSV
    for diff, data in avgTaskDiffs.items():
        depth_bins = np.arange(len(data['avg']))  # Create depth bins
        df = pd.DataFrame({
            'depth bin': depth_bins,
            'avg': data['avg'],
            'stdev': data['stdev'],
            'norm_depths': data['norm_depths'],
            't-statistic': data['p-vals'].statistic,
            'p-value': data['p-vals'].pvalue,
            'p-value corrected': all_pvals_dict[roi_type][diff],
            'df': data['p-vals'].df,
            'N': [data['Nsamp']] * len(data['avg'])  # Repeat the Nsamp value
        })
        df.to_csv(os.path.join(figDir,f"{diff}_{roi_type}.csv"), index=False)

    
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
    xlim = [-0.3,1.8]
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
    xlim = [-0.3,1.8]
        
    plot_avg_diff_profile(p2,avgLocDiffs,locDiffs,diffDetails['colors'][4:],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI)
    
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_task_deconv.%s' %(roi_type,fig_format)))
            fig2.savefig(os.path.join(figDir,'%s_avg_profiles_loc_deconv.%s' %(roi_type,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'avg_profiles_task.%s' %(roi_type,fig_format)))
            fig2.savefig(os.path.join(figDir,'avg_profiles_loc.%s' %(roi_type,fig_format)))
            
    #%% Plot each context modulation effect separately
    
    pthresh = 0.05 #pvalue significance threshold
    pvals_task = {cond:all_pvals_dict[roi_type][cond] for cond in avgTaskDiffs.keys()}
    pvals_loc = {cond:all_pvals_dict[roi_type][cond] for cond in avgLocDiffs.keys()}
    # pvals_other = {cond:all_pvals_dict[cond] for cond in avgOtherDiffs.keys()}
    
    #iso and surround only
    fig = plt.figure(figsize=(10, 4))
    fig.set_size_inches((10,4))
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
    xlim = [-0.5,1.5]
    p2 = fig.add_axes([.5, .2, .35, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    plot_avg_diff_profile(p2,avgTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task)
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_iso_sur_deconv.%s' %(roi_type,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_iso_sur.%s' %(roi_type,fig_format)))
    
    #iso and orth
    fig = plt.figure(figsize=(10, 4))
    fig.set_size_inches((10,4))
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
    xlim = [-0.5,1.5]
    p2 = fig.add_axes([.5, .2, .35, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    plot_avg_diff_profile(p2,avgTaskDiffs,['dsi'],['tab:cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task)
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_dsi_deconv.%s' %(roi_type,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_dsi.%s' %(roi_type,fig_format)))
    
    #iso and iso90
    fig = plt.figure(figsize=(10, 4))
    fig.set_size_inches((10,4))
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
    xlim = [-0.5,1.5]
    p2 = fig.add_axes([.5, .2, .35, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    plot_avg_diff_profile(p2,avgTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task)
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_fgm_deconv.%s' %(roi_type,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_fgm.%s' %(roi_type,fig_format)))
    
    #iso90 and orth    
    fig = plt.figure(figsize=(10, 4))
    fig.set_size_inches((10,4))
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
    xlim = [-0.5,1.5]
    p2 = fig.add_axes([.5, .2, .35, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    plot_avg_diff_profile(p2,avgTaskDiffs,['odss'],['tab:green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task)
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_otss_deconv.%s' %(roi_type,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_otss.%s' %(roi_type,fig_format)))
    
    #ctr and sur
    fig = plt.figure(figsize=(10, 4))
    fig.set_size_inches((10,4))
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
    xlim = [-0.5,1.5]
    p2 = fig.add_axes([.5, .2, .35, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    plot_avg_diff_profile(p2,avgLocDiffs,['ctr-sur'],['coral'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_loc)
    
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_ctr_sur_deconv.%s' %(roi_type,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'%s_avg_profiles_ctr_sur.%s' %(roi_type,fig_format)))
            
    # #iso90 and sur  
    # fig = plt.figure(figsize=(6, 4))
    # fig.set_size_inches((6,4))
    # fig.patch.set_facecolor(fcolor)
    
    # fig.clf()
    # fsize = 14
    
    # p1 = fig.add_axes([.15, .2, .3, .7])
    # fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
        
    # ylim = [-0.02,1.02]
    # xlim = [0,6]
    # Ntext = [4,0.05]
    # plot_avg_depth_profile(p1,avgTaskProfiles,['iso90','sur'],['purple','gray'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)
    
    # #iso90-sur
    # xlim = [-0.8,1.5]
    # p2 = fig.add_axes([.7, .2, .25, .7])
    # fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    # plot_avg_diff_profile(p2,avgOtherDiffs,['iso90-sur'],['darkmagenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_other)
    
    # if savefigs:
    #     if use_decon:
    #         fig.savefig(os.path.join(figDir,'avg_profiles_iso90-sur_deconv.%s' %(fig_format)))
    #     else:
    #         fig.savefig(os.path.join(figDir,'avg_profiles_iso90-sur.%s' %(fig_format)))
    
    # #orth and sur  
    # fig = plt.figure(figsize=(6, 4))
    # fig.set_size_inches((6,4))
    # fig.patch.set_facecolor(fcolor)
    
    # fig.clf()
    # fsize = 14
    
    # p1 = fig.add_axes([.15, .2, .3, .7])
    # fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
        
    # ylim = [-0.02,1.02]
    # xlim = [0,6]
    # Ntext = [4,0.05]
    # plot_avg_depth_profile(p1,avgTaskProfiles,['orth','sur'],['orange','gray'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)
    
    # #orth-sur
    # xlim = [-0.8,1.5]
    # p2 = fig.add_axes([.7, .2, .25, .7])
    # fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    # plot_avg_diff_profile(p2,avgOtherDiffs,['orth-sur'],['darkturquoise'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_other)
    
    # if savefigs:
    #     if use_decon:
    #         fig.savefig(os.path.join(figDir,'avg_profiles_orth-sur_deconv.%s' %(fig_format)))
    #     else:
    #         fig.savefig(os.path.join(figDir,'avg_profiles_orth-sur.%s' %(fig_format)))
        
    #%% Make condition depth profiles with individual subject data overlaid
    
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 14
    
    p1 = fig.add_axes([.15, .2, .3, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
        
    ylim = [-0.02,1.02]
    xlim = [0,7]
    Ntext = [4,0.05]
    plot_avg_depth_profile(p1,avgTaskProfiles,['sur'],['gray'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles[roi_type])
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_profiles_sur.%s' %(roi_type,fig_format)))
    
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 14
    
    p1 = fig.add_axes([.15, .2, .3, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
        
    ylim = [-0.02,1.02]
    xlim = [0,7]
    Ntext = [4,0.05]
    plot_avg_depth_profile(p1,avgTaskProfiles,['iso0'],['red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles[roi_type])
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_profiles_iso0.%s' %(roi_type,fig_format)))
    
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 14
    
    p1 = fig.add_axes([.15, .2, .3, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
        
    ylim = [-0.02,1.02]
    xlim = [0,7]
    Ntext = [4,0.05]
    plot_avg_depth_profile(p1,avgTaskProfiles,['iso90'],['darkviolet'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles[roi_type])
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_profiles_iso90.%s' %(roi_type,fig_format)))
    
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 14
    
    p1 = fig.add_axes([.15, .2, .3, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
        
    ylim = [-0.02,1.02]
    xlim = [0,7]
    Ntext = [4,0.05]
    plot_avg_depth_profile(p1,avgTaskProfiles,['orth'],['orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles[roi_type])
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_profiles_orth.%s' %(roi_type,fig_format)))
        
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 14
    
    p1 = fig.add_axes([.15, .2, .3, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
        
    ylim = [-0.02,1.02]
    xlim = [-3,7]
    Ntext = [4,0.05]
    plot_avg_depth_profile(p1,avgLocProfiles,['ctr_unwarp'],['gold'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles[roi_type])
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_profiles_ctr.%s' %(roi_type,fig_format)))
        
    fig = plt.figure(figsize=(6, 4))
    fig.set_size_inches((6,4))
    fig.patch.set_facecolor(fcolor)
    
    fig.clf()
    fsize = 14
    
    p1 = fig.add_axes([.15, .2, .3, .7])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
        
    ylim = [-0.02,1.02]
    xlim = [-3,7]
    Ntext = [4,0.05]
    plot_avg_depth_profile(p1,avgLocProfiles,['sur_unwarp'],['purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles[roi_type])
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_profiles_sur1.%s' %(roi_type,fig_format)))
        
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
    plot_avg_diff_profile(p2,avgLocDiffs,['ctr-sur'],['coral'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles[roi_type],showSig=True,pthresh=pthresh,statCorrType=pvals_loc)
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_diffs_ctr-sur.%s' %(roi_type,fig_format)))
    
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
    plot_avg_diff_profile(p2,avgTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles[roi_type],showSig=True,pthresh=pthresh,statCorrType=pvals_task)
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_diffs_iso-sur.%s' %(roi_type,fig_format)))
    
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
    plot_avg_diff_profile(p2,avgTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles[roi_type],showSig=True,pthresh=pthresh,statCorrType=pvals_task)
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_diffs_fgm.%s' %(roi_type,fig_format)))
    
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
    plot_avg_diff_profile(p2,avgTaskDiffs,['odss'],['green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles[roi_type],showSig=True,pthresh=pthresh,statCorrType=pvals_task)
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_diffs_odss.%s' %(roi_type,fig_format)))
        
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
    plot_avg_diff_profile(p2,avgTaskDiffs,['dsi'],['cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles[roi_type],showSig=True,pthresh=pthresh,statCorrType=pvals_task)
    
    if savefigs:
        fig.savefig(os.path.join(figDir,'%s_avg_diffs_dsi.%s' %(roi_type,fig_format)))