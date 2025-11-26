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
fig_format = 'png'
statCorrType = 'bonferroni'

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code'
#datasets = glob.glob(os.path.join(mainDir, 'roi_data', 'pnr???_??_??_?????.csv'))
datasets = [#'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_lh_target_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr102_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr256_V23_tgt_lh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr256_V23_tgt_rh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr328_V23_tgt_lh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr328_V23_tgt_rh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr510_V23_tgt_lh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr510_V23_tgt_rh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr739_V23_tgt_lh.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/V23_filled/pnr739_V23_tgt_rh.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target/pnr756_V1_tgt_lh_rad10.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target/pnr756_V1_tgt_rh_rad10.csv'
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
nDepths = 6
fig = plt.figure(num=1)
fig.clf()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df
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
    fig.savefig(os.path.join(figDir,'t1w_profiles_V23.%s' %(fig_format)))

#%% Histograms of p-values
floc = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df
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
ftask.tight_layout(pad=0.1)

if savefigs:
    floc.savefig(os.path.join(figDir,'pvals_loc_V23.%s' %(fig_format)))
    ftask.savefig(os.path.join(figDir,'pvals_task_V23.%s' %(fig_format)))

#%% Depth Histograms
# I want to see how much coverage we are getting through depth.
fdhist = plt.figure(figsize=(15,4))

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
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
    fdhist.savefig(os.path.join(figDir,'depth_hist_V23.%s' %(fig_format)))
    
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
             
    # Plot voxel loss at each depth after masking
    fdepth_hist = plot_depth_voxel_loss(z, mnv_mask, nDepths, NROIs, key, k_i, fsize)
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(mnv_mask),np.size(mnv),key))
    
if savefigs:
    fthresh.savefig(os.path.join(figDir,'mnv_hist_V23.%s' %(fig_format)))
    fdepth_hist.savefig(os.path.join(figDir,'mnv_depth_hist_V23.%s' %(fig_format)))

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
    plt.plot(spreadF*(k_i+0.4),lmnv_dict[key]['thresh'],color='r',marker='s')
plt.xticks(np.arange(0,spreadF*len(all_data.keys()),spreadF),all_data.keys(),rotation=15,fontsize=6)
plt.ylabel("log(MNV)")

if savefigs:
    f.savefig(os.path.join(figDir,'mnv_summary_violin_V23.%s' %(fig_format)))
    
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
    
statDetails = {'labels': ['sur', 'iso0', 'iso90', 'orth', 'ctr', 'sur.1', 'ctr-sur'],
                'colors': [[.7, .7, .7], 'red', 'darkviolet', 'orange', 'gold', 'purple', 'coral']}
diffDetails = {}
diffDetails['statIDs'] = {'odss': ['orth','iso90'],
                          'fgm': ['iso90','iso0'],
                          'dsi': ['orth','iso0'],
                          'iso-sur': ['iso0','sur'],
                          'ctr-sur': ['ctr','sur.1']}
diffDetails['colors'] = ['green','magenta','cyan','black','coral']
profile_method = 'bin' # bin or smooth

useSI = False #use suppression index rather than differences (cond1 - cond2 / cond1 + cond2)
#create full masks
masks = {roi:all_data[roi]['sig']*all_data[roi]['no_vein'] for roi in all_data.keys()}
depthProfiles = compute_all_depth_profiles(all_data,statDetails,profile_method,nDepths,masks,depthParam='d',radialParam=None,spec_Drange='MinMax')
diffProfiles = compute_diff_profiles(all_data,statDetails,diffDetails['statIDs'],profile_method,nDepths,useSI,masks,depthParam='d',radialParam=None,spec_Drange='MinMax')
      
#%% Centroid plots
# Let's take a look at raw voxel betas across depth by condition

Nsubj = len(all_data.keys())

#plot centroids for each condition and ROI
plot_centroids(all_data, masks, statDetails, roiRad, nDepths, radParam=None)
        
#calculate difference profiles
plot_centroids_diff(all_data, masks, statDetails, diffDetails, roiRad, nDepths, radParam=None)
          
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
    dx = 1.
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
    fig.savefig(os.path.join(figDir,'avg_profiles_task_V23.%s' %(fig_format)))
    fig2.savefig(os.path.join(figDir,'avg_profiles_loc_V23.%s' %(fig_format)))
    
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
xlim = [-1.2,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_iso_sur_V23.%s' %(fig_format)))

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
xlim = [-1.2,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['dsi'],['tab:cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_dsi_V23.%s' %(fig_format)))

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
xlim = [-1.2,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_fgm_V23.%s' %(fig_format)))

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
xlim = [-1.2,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['odss'],['tab:green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_otss_V23.%s' %(fig_format)))

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
plot_avg_depth_profile(p1,avgLocProfiles,['ctr','sur.1'],['gold','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#ctr - sur
xlim = [-1.2,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgLocDiffs,['ctr-sur'],['coral'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType='bonferroni')


if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_ctr_sur_V23.%s' %(fig_format)))
    
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
    fig.savefig(os.path.join(figDir,'avg_profiles_sur_V23.%s' %(fig_format)))

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
    fig.savefig(os.path.join(figDir,'avg_profiles_iso90_V23.%s' %(fig_format)))

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
    fig.savefig(os.path.join(figDir,'avg_profiles_orth_V23.%s' %(fig_format)))
    
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
plot_avg_depth_profile(p1,avgLocProfiles,['ctr'],['gold'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles)

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_ctr_V23.%s' %(fig_format)))
    
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
plot_avg_depth_profile(p1,avgLocProfiles,['sur.1'],['purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles)

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_profiles_sur1_V23.%s' %(fig_format)))
    
#%% Make difference depth profiles with individual subject data overlaid

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.6,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgLocDiffs,['ctr-sur'],['coral'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_ctr-sur_V23.%s' %(fig_format)))

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.6,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_iso-sur_V23.%s' %(fig_format)))

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.6,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_fgm_V23.%s' %(fig_format)))

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.6,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgTaskDiffs,['odss'],['green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_odss_V23.%s' %(fig_format)))
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    
ylim = [-0.02,1.02]
xlim = [-1.6,2]
Ntext = [4,0.05]
plot_avg_diff_profile(p2,avgTaskDiffs,['dsi'],['cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles,showSig=True,pthresh=pthresh,statCorrType='bonferroni')

if savefigs:
    fig.savefig(os.path.join(figDir,'avg_diffs_dsi_V23.%s' %(fig_format)))
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)