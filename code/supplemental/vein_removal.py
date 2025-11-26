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

#Import custom functions
from oriseg_funcs import *

plt.close('all')
    
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]

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
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target/pnr756_V1_tgt_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/target_filled/pnr756_V1_tgt_rh_rad10.csv'
  ]
datasets.sort()
# to get pnr739_lh on right to use as demo subject
DoF_file = os.path.join(mainDir,'DoF_V1.txt') #degrees of freedom file for F-stat calculation
#import degrees of freedom
with open(DoF_file,'r') as infile:
        DoF_old = json.load(infile)
DoF = {}
roiKeys = DoF_old.keys()
for key in roiKeys: #clean up key names
    subjKeys = DoF_old[key].keys()
    DoF[key] = {}
    for subjDir in subjKeys:
        newlabel = subjDir[:6]
        DoF[key][newlabel] = DoF_old[key][subjDir]

roiRad = 2 #1.
import pandas as pd
all_data = {}
for dataset in datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    all_data[f] = pd.read_csv(dataset, sep=',', index_col=False)

# check and see what the Stria profile looks like in each ROI
nDepths = 7
fig = plt.figure(num=1)
fig.clf()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = roi[roi['scale_xy_dist'] > 0]
    dataDict = makeProfile1D(roi['d'].values,
                             nDepths, #number of depths
                             roi['t1'].values,
                             np.min(roi['d'].values), #min depth value
                             np.max(roi['d'].values), #max depth value
                             True) #Use LayNii values
    
    plt.subplot(int(np.ceil(len(all_data.keys())/2.)), 2, 1 + iR)
    plt.plot(dataDict['profile']['depth'],
             dataDict['profile']['avg'][0])
    plt.title('%s (%d vox)' %(label, len(roi)), fontsize=8)

#%% Depth Histograms
# I want to see how much coverage we are getting through depth.
roiRad = 2
fdhist = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,np.ceil(len(datasets)/2),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.title(label)
    plt.xlabel("Normalize Depth WM -> GM")
    plt.ylabel("Num. Voxels")
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
fdhist.tight_layout(pad=0.1)
    
#%% Superficial Vein Removal
# Use mean-normalized variance as a way to estimate voxels that are located 
# where large veins are. Larger veins will tend to have high variance and will
# show up as voxels with unually high mean-normalized variance.

fmap = plt.figure("map")
fhist = plt.figure("hist")
floghist = plt.figure("loghist")
fdepth = plt.figure("depth")
conditions = ['iso0','iso90','orth','sur']
mnv_thresh = 50
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
    plt.subplot(4,len(all_data.keys()),(k_i+1))
    plt.scatter(all_data[key]['x'],all_data[key]['y'],s = 1,c = mnv,cmap='Reds')
    if k_i == len(all_data.keys())-1:
        plt.colorbar()
    plt.title(key+" Var/Mean",fontsize=5)
    plt.xticks(ticks = [], labels = [])
    plt.yticks(ticks = [], labels = [])
    plt.clim([0,100])
    
    plt.subplot(4,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.scatter(all_data[key]['x'],all_data[key]['y'],s = 1,c = avg_betas,cmap='coolwarm')
    if k_i == len(all_data.keys())-1:
        plt.colorbar()
    plt.title(key+" Mean $\\beta$",fontsize=5)
    plt.xticks(ticks = [], labels = [])
    plt.yticks(ticks = [], labels = [])
    plt.clim([-10,10])
    
    plt.subplot(4,len(all_data.keys()),(k_i+1)+2*len(all_data.keys()))
    plt.scatter(all_data[key]['x'],all_data[key]['y'],s = 1,c = (all_data[key]['stdev_xerrts'].values),cmap='Reds')
    if k_i == len(all_data.keys())-1:
        plt.colorbar()
    plt.title(key+" SD/Mean",fontsize=5)
    plt.xticks(ticks = [], labels = [])
    plt.yticks(ticks = [], labels = [])
    plt.clim([0,10])
    
    plt.subplot(4,len(all_data.keys()),(k_i+1)+3*len(all_data.keys()))
    plt.scatter(all_data[key]['x'][mnv_mask],all_data[key]['y'][mnv_mask],s = 1,c = mnv[mnv_mask],cmap='Reds')
    if k_i == len(all_data.keys())-1:
        plt.colorbar()
    plt.title(key+" Log(Var/Mean) th=%.1f" %(mnv_thresh),fontsize=5)
    plt.xticks(ticks = [], labels = [])
    plt.yticks(ticks = [], labels = [])
    plt.clim([0,100])
    
    #plot histograms
    plt.figure("hist")
    plt.subplot(4,len(all_data.keys()),(k_i+1))
    plt.hist(mnv,bins=np.linspace(-100,100,200))
    plt.title(key+" Log(Var/Mean)",fontsize=6)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    
    plt.subplot(4,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.hist(avg_betas,bins=np.linspace(-10,10,200))
    plt.title(key+" Mean $\\beta$",fontsize=6)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    
    plt.subplot(4,len(all_data.keys()),(k_i+1)+2*len(all_data.keys()))
    plt.hist(all_data[key]['stdev_xerrts'].values,bins=np.linspace(-10,10,200))
    plt.title(key+" Log(SD/Mean)",fontsize=6)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    
    plt.subplot(4,len(all_data.keys()),(k_i+1)+3*len(all_data.keys()))
    plt.hist(mnv[mnv_mask],bins=np.linspace(-100,100,200))
    plt.title(key+" Log(Var/Mean) th=%.1f" %(mnv_thresh),fontsize=5)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    
    #now compare histograms of mnv to log histograms
    plt.figure("loghist")
    plt.subplot(2,len(all_data.keys()),(k_i+1))
    plt.hist(mnv,bins=np.linspace(-100,100,200))
    plt.title(key,fontsize=6)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    plt.xlabel("Var/Mean",fontsize=6)
    
    plt.subplot(2,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.hist(np.log(mnv),bins=np.linspace(0,10,200))
    plt.title(key,fontsize=6)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    plt.xlabel("Log(Var/Mean)",fontsize=6)
    
    
    #check depth histograms
    nD = np.max(all_data[key]['d'])
    frac_included = np.zeros((int(nDepths),))
    binSize = nD/nDepths
    depthBoundaries = np.arange(np.min(all_data[key]['d'])-binSize/2, np.max(all_data[key]['d'])+binSize, binSize)
    for di in np.arange(0,nDepths):
        dmasked = all_data[key]['d'][mnv_mask]
        frac_included[di] = np.sum((dmasked > depthBoundaries[di]) & (dmasked <= depthBoundaries[di+1]))/np.sum((all_data[key]['d'] > depthBoundaries[di]) & (all_data[key]['d'] <= depthBoundaries[di+1]))
    plt.figure("depth")
    plt.subplot(2,len(all_data.keys()),(k_i+1))
    nD = np.max(all_data[key]['d'])
    plt.hist(all_data[key]['d'],bins=np.arange(np.min(all_data[key]['d']),nD,nD/nDepths))
    plt.title(key+" Depth Hist",fontsize=5)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.xlim([0,1])
    plt.yticks(ticks = None, labels = None, fontsize=4)
    plt.hist(all_data[key]['d'][mnv_mask],bins=np.arange(np.min(all_data[key]['d']),nD,nD/nDepths),alpha=0.7)
    plt.legend(["unmasked","masked"],fontsize=4)
    
    plt.subplot(2,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.bar(np.arange(0,nD,nD/nDepths),frac_included,width=nD/nDepths,color='tomato')
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    plt.xlabel("Depth (WM --> Pia)",fontsize=5)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("Frac Included",fontsize=5)
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(np.abs(mnv) < mnv_thresh),np.size(mnv),key))
    
    mask_dict[key] = mnv_mask #add mask to dictionary
    all_data[key]['mnv'] = mnv #add mnv to all_data
    
fhist.tight_layout(pad=0.5)
floghist.tight_layout(pad=0.5)
fmap.tight_layout(pad=0.5)
fdepth.tight_layout(pad=0.5)

# Across Depth
depth_groups = {'deep': [0.2,0.4], 'middle': [0.4,0.6], 'superficial': [0.6,0.8]}
depth_labels = ['superficial','middle','deep'] #put them in the right order
Ngroups = len(depth_groups.keys())
NROIs = len(all_data.keys())
fdmaps = plt.figure()
for k_i, key in enumerate(all_data.keys()):
    mnv = all_data[key]['mnv'].values
    z = all_data[key]['d'].values
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
        plt.title(key_name+" MNV "+depth_label,fontsize=5)
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        
        #plot hist
        plt.subplot(Ngroups*4,NROIs,(k_i+1)+(2*NROIs)+(4*d_i*NROIs))
        plt.hist(mnv[level_mask],bins=np.linspace(0,100,100))
        plt.xticks(ticks = None, labels = None, fontsize=4)
        plt.yticks(ticks = None, labels = None, fontsize=4)
        plt.xlabel("Var/Mean",fontsize=5)
        
#%% Use the deepest layer as a proxy for non-vein contaminated voxels
# Then define the threshold based on this distribution

fthresh = plt.figure("thresh")
fdmap = plt.figure("dmap")
fdmap_thresh = plt.figure("dmap_thresh")
fdepth = plt.figure("depth")
deep_pct = 10 #percentile to call deep layers
conditions = ['iso0','iso90','orth','sur']
depth_groups = {'deep': [0.2,0.4], 'middle': [0.4,0.6], 'superficial': [0.6,0.8]}
depth_labels = ['superficial','middle','deep'] #put them in the right order
depth_var = 'd'
x_var = 'u'
y_var = 'v'
sd_thresh = 2 #how many st. dev. of the deep layer mean to use as the threshold
mask_dict = {} #create a mask dictionary
lmnv_dict = {key:{'mean':0,'std':0,'thresh':0,'deep_mean':0,'deep_std':0} for key in all_data.keys()} #thresh dictionary
fsize=8 #fontsize of title
for k_i, key in enumerate(all_data.keys()):

    # calculate log(MNV)
    df = all_data[key]
    mnv = df['task stdev_xerrts'].values**2 #the variance from the xerrts are already normalized by the mean
    lmnv = np.log(mnv) #log of the mean-normalized variance
    
    # get deep layer distribution
    z = df[depth_var]
    deep = z <= np.percentile(z,deep_pct) #deepest % of voxels
    deep_mean = np.mean(lmnv[deep])
    deep_std = np.std(lmnv[deep])
    lmnv_dict[key]['deep_mean'] = deep_mean
    lmnv_dict[key]['deep_std'] = deep_std
    
    # define threshold based on deep layer distribution
    lmnv_thresh = deep_mean + sd_thresh*deep_std
    lmnv_dict[key]['thresh'] = lmnv_thresh
    lmnv_dict[key]['mean'] = np.mean(lmnv)
    lmnv_dict[key]['std'] = np.std(lmnv)
    mnv_mask = lmnv < lmnv_thresh
    mask_dict[key] = mnv_mask
    
    # Plot distributions
    plt.figure("thresh")
    plt.subplot(2,len(all_data.keys()),(k_i+1))
    plt.hist(lmnv,bins=np.linspace(0,10,200),density=True,alpha=0.5)
    plt.hist(lmnv[deep],bins=np.linspace(0,10,200),density=True,alpha=0.5)
    plt.xlabel("log(MNV)",fontsize=0.7*fsize)
    plt.ylabel("Density (voxels/bin len)",fontsize=0.7*fsize)
    plt.legend(['full','deepest %d%%' %(deep_pct)],fontsize=0.7*fsize)
    plt.xticks(fontsize=0.5*fsize)
    plt.yticks(fontsize=0.5*fsize)
    plt.title(key,fontsize=fsize)
    # Plot thresholded distribution
    plt.subplot(2,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.hist(lmnv[mnv_mask],bins=np.linspace(0,10,200),density=True,alpha=0.5)
    plt.xlabel("log(MNV)",fontsize=0.7*fsize)
    plt.ylabel("Density (voxels/bin len)",fontsize=0.7*fsize)
    plt.legend(['masked'],fontsize=0.7*fsize)
    plt.xticks(fontsize=0.5*fsize)
    plt.yticks(fontsize=0.5*fsize)
    plt.title(key,fontsize=fsize)
    
    # Plot depth maps
    x = df[x_var]
    y = df[y_var]
    for d_i, depth_label in enumerate(depth_labels):
        level_mask = (z >= depth_groups[depth_label][0]) & (z <= depth_groups[depth_label][1])
        
        #plot map
        plt.figure("dmap")
        plt.subplot(Ngroups*2,NROIs,(k_i+1)+2*d_i*NROIs)
        plt.scatter(x[level_mask],y[level_mask],s=0.5,c=mnv[level_mask],cmap='Reds')
        plt.clim([0,100])
        if k_i == NROIs-1:
            plt.colorbar()
        key_split = key.split('_')
        key_name = key_split[0]+key_split[2]
        plt.title(key_name+" MNV \n "+depth_label,fontsize=fsize)
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        
        #plot hist
        plt.subplot(Ngroups*4,NROIs,(k_i+1)+(2*NROIs)+(4*d_i*NROIs))
        plt.hist(mnv[level_mask],bins=np.linspace(0,100,100))
        plt.xticks(ticks = None, labels = None, fontsize=0.5*fsize)
        plt.yticks(ticks = None, labels = None, fontsize=0.5*fsize)
        plt.xlabel("MNV",fontsize=0.7*fsize)
        
        #plot thresholded map
        plt.figure("dmap_thresh")
        plt.subplot(Ngroups*2,NROIs,(k_i+1)+2*d_i*NROIs)
        plt.scatter(x[level_mask*mnv_mask],y[level_mask*mnv_mask],s=0.5,c=mnv[level_mask*mnv_mask],cmap='Reds')
        plt.clim([0,100])
        if k_i == NROIs-1:
            plt.colorbar()
        key_split = key.split('_')
        key_name = key_split[0]+key_split[2]
        plt.title(key_name+" MNV \n "+depth_label,fontsize=fsize)
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        
        #plot hist
        plt.subplot(Ngroups*4,NROIs,(k_i+1)+(2*NROIs)+(4*d_i*NROIs))
        plt.hist(mnv[level_mask*mnv_mask],bins=np.linspace(0,100,100))
        plt.xticks(ticks = None, labels = None, fontsize=0.5*fsize)
        plt.yticks(ticks = None, labels = None, fontsize=0.5*fsize)
        plt.xlabel("MNV",fontsize=0.7*fsize)
        
    # Plot voxel loss at each depth after masking
    nD = np.max(z)
    frac_included = np.zeros((int(nDepths),))
    binSize = nD/nDepths
    depthBoundaries = np.arange(np.min(z)-binSize/2, np.max(z)+binSize, binSize)
    for di in np.arange(0,nDepths):
        dmasked = z[mnv_mask]
        frac_included[di] = np.sum((dmasked > depthBoundaries[di]) & (dmasked <= depthBoundaries[di+1]))/np.sum((z > depthBoundaries[di]) & (z <= depthBoundaries[di+1]))
    plt.figure("depth")
    plt.subplot(2,len(all_data.keys()),(k_i+1))
    nD = np.max(z)
    plt.hist(z,bins=np.arange(np.min(z),nD,nD/nDepths))
    plt.title(key+"\n Depth Hist",fontsize=fsize)
    plt.xticks(ticks = None, labels = None, fontsize=0.5*fsize)
    plt.xlim([0,1])
    plt.ylim([0,1.3*np.max(np.bincount(np.digitize(z,np.arange(np.min(z),nD,nD/nDepths))))])
    plt.yticks(ticks = None, labels = None, fontsize=0.5*fsize)
    plt.hist(z[mnv_mask],bins=np.arange(np.min(z),nD,nD/nDepths),alpha=0.7)
    plt.legend(["unmasked","masked"],fontsize=0.7*fsize)
    
    plt.subplot(2,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.bar(np.arange(0,nD,nD/nDepths),frac_included,width=nD/nDepths,color='tomato')
    plt.xticks(ticks = None, labels = None, fontsize=0.5*fsize)
    plt.yticks(ticks = None, labels = None, fontsize=0.5*fsize)
    plt.xlabel("Depth (WM --> Pia)",fontsize=0.7*fsize)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("Frac Included",fontsize=0.7*fsize)
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(mnv_mask),np.size(mnv),key))
    
fthresh.tight_layout(pad=0.1)
fdepth.tight_layout(pad=0.5)

#%% Compare thresholds between subjects

f = plt.figure()
for k_i, key in enumerate(all_data.keys()):
    plt.errorbar(k_i,lmnv_dict[key]['mean'],lmnv_dict[key]['std'],color='b',linestyle='None',marker='o')
    plt.errorbar(k_i+0.1,lmnv_dict[key]['deep_mean'],lmnv_dict[key]['deep_std'],color='orange',linestyle='None',marker='o')
    plt.plot(k_i+0.2,lmnv_dict[key]['thresh'],color='r',marker='s')
plt.xticks(np.arange(0,len(all_data.keys())),all_data.keys(),rotation=30,fontsize=6)
plt.ylabel("log(MNV)")
#plt.legend(['full','thresh'])
        
#%% Fit Gaussian Mixture model to data

from sklearn.mixture import GaussianMixture


