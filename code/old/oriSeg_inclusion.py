#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cheryl

This code combines work from Cheryl and Joe. This code tests out inclusion 
criteria for datasets.
"""
#import nibabel
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json

#Import custom functions
from oriseg_funcs import *

plt.close('all')
    
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]
savefigs = False

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '.'
datasets = glob.glob(os.path.join(mainDir, 'roi_data/target_filled', 'pnr???_??_???_??_?????.csv'))
#or exclude pnr102 for bad ellipse fit, pnr161 for missing superfical data, and rh of pn510 for no stria
exclude_initial = ['pnr352_V1_tgt_lh_rad10']
for e_i, excl in enumerate(exclude_initial):
    datasets.remove(os.path.join(mainDir,'roi_data/target_filled',excl+'.csv'))               
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

# # to get pnr739_lh on right to use as demo subject
# DoF_file = os.path.join(mainDir,'DoF_V1.txt') #degrees of freedom file for F-stat calculation
# #import degrees of freedom
# with open(DoF_file,'r') as infile:
#         DoF_old = json.load(infile)
# DoF = {}
# roiKeys = DoF_old.keys()
# for key in roiKeys: #clean up key names
#     subjKeys = DoF_old[key].keys()
#     DoF[key] = {}
#     for subjDir in subjKeys:
#         newlabel = subjDir[:6]
#         DoF[key][newlabel] = DoF_old[key][subjDir]

#%% T1w data
roiRad = 2 #1.
import pandas as pd
all_data = {}
for dataset in datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    all_data[f] = pd.read_csv(dataset, sep=',', index_col=False)

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

#%% Histograms of p-values

# The voxels we are analyzing need to be selective for the center over the 
# surround. I'll use the p-values from the localizer ctr - sur regressor to 
# make that determination. I'll consider voxels with a ctr - sur p-val of less
# than 0.05 to be selective. I'll restrict my search to only within the first 
# st. dev. of the target selective radius, because these voxels are the ones
# that should be most driven by the center and have minimal overlapping 
# receptive fields with the surround. I'll include an ROI if >50% of the 
# voxels within the first st. dev. of the radius of the target-selective 
# region have a ctr - sur p-val of <= 0.05.
roiRad = 1
locPthresh = 0.01 #p-value threshold
locPercSigThresh = 50 #% of voxels significantly modulated threshold
excludeROIs = []
floc = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    
    if 'loc pval' in df.keys(): #some special logic for a notation error, this should be fixed in extract_ROIs.py in the future
            df = df.rename(columns={'loc pval':'loc p-val'})
            all_data[label] = df
    roi = df[df['scale_xy_dist'] < roiRad]
        
    percSig = 100*np.sum(roi['loc p-val'] <= 0.05)/len(roi['loc p-val'])
    if percSig < locPercSigThresh:
        excludeROIs.append(label)
    
    plt.subplot(int(np.ceil(len(datasets)/2)),2,iR+1)
    plt.hist(roi['loc p-val'].values,bins=20,density=True)
    plt.title(label+" loc p-val",fontsize=8)
    plt.xlabel("pval",fontsize=6)
    plt.text(0.7,5,'< %.2f = %d %%' %(locPthresh,percSig),fontsize=6)
    plt.ylim([0,10])
    plt.xlim([0,1])
    if iR not in [int(2*np.ceil(len(datasets)/2)-2),int(2*np.ceil(int(len(datasets)/2))-1)]:
        plt.xticks([],fontsize=6)
    else:
        plt.xticks(fontsize=6)
    if np.mod(iR,2) == 1:
        plt.yticks([],fontsize=6)
    else:
        plt.yticks(fontsize=6)
floc.tight_layout(pad=0.1)
floc.savefig("locPvals.png")


# I also need to know that the voxels in the ROI were significantly modulated
# by the task stimuli. Here, I will just be looking at the p-vals for the 
# combined regressor (all conditions). Since I'll be looking at responses in 
# the target- and surround-selective regions, I'll look at all the voxels in 
# the ROI. I'll include an ROI if 50% or more of the voxels had a task p-val 
# <= 0.05.
taskPthresh = 0.01 #p-value threshold
taskPercSigThresh = 50 #% of voxels significantly modulated threshold
ftask = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    if 'task pval' in df.keys(): #some special logic for a notation error, this should be fixed in extract_ROIs.py in the future
        df = df.rename(columns={'task pval':'task p-val'})
        all_data[label] = df
    roi = df
        
    percSig = 100*np.sum(roi['task p-val'] <= 0.05)/len(roi['task p-val'])
    if percSig < taskPercSigThresh:
        excludeROIs.append(label)
    
    plt.subplot(int(np.ceil(len(datasets)/2)),2,iR+1)
    plt.hist(roi['task p-val'].values,bins=20,density=True)
    plt.title(label+" task p-val",fontsize=8)
    plt.xlabel("pval",fontsize=6)
    plt.text(0.7,5,'< %.2f = %d %%' %(taskPthresh,percSig),fontsize=6)
    plt.ylim([0,20])
    plt.xlim([0,1])
    if iR not in [int(2*np.ceil(len(datasets)/2)-2),int(2*np.ceil(len(datasets)/2)-1)]:
        plt.xticks([],fontsize=6)
    else:
        plt.xticks(fontsize=6)
    if np.mod(iR,2) == 1:
        plt.yticks([],fontsize=6)
    else:
        plt.yticks(fontsize=6)
ftask.tight_layout(pad=0.1)
ftask.savefig("locPvals.png")

#%% Superficial Vein Removal
# Use mean-normalized variance as a way to estimate voxels that are located 
# where large veins are. Larger veins will tend to have high variance and will
# show up as voxels with unually high mean-normalized variance.
#
# I need to make sure that each depth bin has enough voxels to be confident in
# the results. I'll set a criterion of at least 20% of voxels included after 
# deveining at each depth.

fmap = plt.figure("map")
fhist = plt.figure("hist")
fdepth = plt.figure("depth")
conditions = ['iso0','iso90','orth','sur']
tickFont = 6
labelFont = 6
mnv_thresh = 50
Nvpd_thresh = 0.2;  #min fraction of voxels included per depth
mask_dict = {} #create a mask dictionary
for k_i, key in enumerate(all_data.keys()):
    tmp = all_data[key]['iso0'].values
    betas = np.zeros([len(tmp),len(conditions)])
    for c_i,con in enumerate(conditions):
        tmp = all_data[key][con].values
        betas[:,c_i] = tmp
    avg_betas = np.mean(betas,1)
    mnv = all_data[key]['stdev_xerrts'].values**2 #the variance from the xerrts are already normalized by the mean
    mnv_mask = np.abs(mnv) < mnv_thresh
    
    #plot maps
    plt.figure("map")
    plt.subplot(3,len(all_data.keys()),(k_i+1))
    plt.scatter(all_data[key]['x'],all_data[key]['y'],s = 1,c = mnv,cmap='Reds')
    if k_i == len(all_data.keys())-1:
        plt.colorbar()
    plt.title(key+" Var/Mean",fontsize=labelFont)
    plt.xticks(ticks = [], labels = [])
    plt.yticks(ticks = [], labels = [])
    plt.clim([0,100])
    
    plt.subplot(3,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.scatter(all_data[key]['x'],all_data[key]['y'],s = 1,c = avg_betas,cmap='coolwarm')
    if k_i == len(all_data.keys())-1:
        plt.colorbar()
    plt.title(key+" Mean $\\beta$",fontsize=labelFont)
    plt.xticks(ticks = [], labels = [])
    plt.yticks(ticks = [], labels = [])
    plt.clim([-10,10])
    
    plt.subplot(3,len(all_data.keys()),(k_i+1)+2*len(all_data.keys()))
    plt.scatter(all_data[key]['x'][mnv_mask],all_data[key]['y'][mnv_mask],s = 1,c = mnv[mnv_mask],cmap='Reds')
    if k_i == len(all_data.keys())-1:
        plt.colorbar()
    plt.title(key+" Var/Mean th=%.1f" %(mnv_thresh),fontsize=labelFont)
    plt.xticks(ticks = [], labels = [])
    plt.yticks(ticks = [], labels = [])
    plt.clim([0,100])
    
    #plot histograms
    plt.figure("hist")
    plt.subplot(3,len(all_data.keys()),(k_i+1))
    plt.hist(mnv,bins=np.linspace(-100,100,200))
    plt.title(key+" Var/Mean",fontsize=labelFont)
    plt.xticks(ticks = None, labels = None, fontsize=tickFont)
    plt.yticks(ticks = None, labels = None, fontsize=tickFont)
    
    plt.subplot(3,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.hist(avg_betas,bins=np.linspace(-10,10,200))
    plt.title(key+" Mean $\\beta$",fontsize=labelFont)
    plt.xticks(ticks = None, labels = None, fontsize=tickFont)
    plt.yticks(ticks = None, labels = None, fontsize=tickFont)
    
    plt.subplot(3,len(all_data.keys()),(k_i+1)+2*len(all_data.keys()))
    plt.hist(mnv[mnv_mask],bins=np.linspace(-100,100,200))
    plt.title(key+" Var/Mean th=%.1f" %(mnv_thresh),fontsize=labelFont)
    plt.xticks(ticks = None, labels = None, fontsize=tickFont)
    plt.yticks(ticks = None, labels = None, fontsize=tickFont)
    
    #check depth histograms
    nD = np.max(all_data[key]['z'])+1
    frac_included = np.zeros((int(nD),))
    exclude = False
    for di in range(int(nD)):
        dmasked = all_data[key]['z'][mnv_mask]
        frac_included[di] = np.sum(dmasked == di)/np.sum(all_data[key]['z'] == di)
        if np.sum(dmasked==di)/np.sum(all_data[key]['z'] == di) < Nvpd_thresh:
            exclude = True
    if exclude:
        excludeROIs.append(key)
    plt.figure("depth")
    plt.subplot(2,len(all_data.keys()),(k_i+1))
    nD = np.max(all_data[key]['z'])+1
    plt.hist(all_data[key]['z'],bins=np.arange(0,nD))
    plt.title(key+" Depth Hist",fontsize=labelFont)
    plt.xticks(ticks = None, labels = None, fontsize=tickFont)
    plt.yticks(ticks = None, labels = None, fontsize=tickFont)
    plt.hist(all_data[key]['z'][mnv_mask],bins=np.arange(0,nD),alpha=0.7)
    plt.legend(["unmasked","masked"],fontsize=tickFont)
    
    plt.subplot(2,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.bar(np.arange(0,nD),frac_included,width=1,color='tomato')
    plt.xticks(ticks = None, labels = None, fontsize=tickFont)
    plt.yticks(ticks = None, labels = None, fontsize=tickFont)
    plt.xlabel("Depth (WM --> Pia)",fontsize=labelFont)
    plt.ylim([0,1])
    plt.ylabel("Frac Included",fontsize=labelFont)
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(np.abs(mnv) < mnv_thresh),np.size(mnv),key))
    
    mask_dict[key] = mnv_mask #add mask to dictionary
    all_data[key]['mnv'] = mnv #add mnv to all_data
    
fhist.tight_layout(pad=0.5)
fmap.tight_layout(pad=0.5)
fdepth.tight_layout(pad=0.1)
fhist.savefig("residualVar_hist.png")
fmap.savefig("residualVar_map.png")
fdepth.savefig("residualVar_depth.png")

# Across Depth
depth_groups = {'deep': [0,6], 'middle': [7,13], 'superficial': [14,21]}
depth_labels = ['superficial','middle','deep'] #put them in the right order
Ngroups = len(depth_groups.keys())
NROIs = len(all_data.keys())
fdmaps = plt.figure()
for k_i, key in enumerate(all_data.keys()):
    mnv = all_data[key]['mnv'].values
    z = all_data[key]['z'].values
    x = all_data[key]['x'].values
    y = all_data[key]['y'].values
    for d_i, depth_label in enumerate(depth_labels):
        level_mask = (z >= depth_groups[depth_label][0]) & (z <= depth_groups[depth_label][1])
        
        #plot map
        plt.subplot(Ngroups*2,NROIs,(k_i+1)+2*d_i*NROIs)
        plt.scatter(x[level_mask],y[level_mask],s=0.5,c=mnv[level_mask],cmap='Reds')
        plt.clim([0,100])
        if k_i == NROIs-1:
            plt.colorbar()
        key_split = key.split('_')
        key_name = key_split[0]+key_split[2]
        plt.title(key_name+" MNV \n"+depth_label,fontsize=labelFont)
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        
        #plot hist
        plt.subplot(Ngroups*4,NROIs,(k_i+1)+(2*NROIs)+(4*d_i*NROIs))
        plt.hist(mnv[level_mask],bins=np.linspace(0,100,100))
        plt.xticks(ticks = None, labels = None, fontsize=tickFont)
        plt.yticks(ticks = None, labels = None, fontsize=tickFont)
        plt.xlabel("Var/Mean",fontsize=5)
        
fdmaps.savefig("residualVar_maphist.png")