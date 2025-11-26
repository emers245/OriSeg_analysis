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
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V1_lh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_lh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V1_lh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V1_rh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_lh_target_laynii.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_rh_target_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr756_V1_lh_target_laynii.csv',
   #'/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr756_V1_rh_target_laynii.csv'
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
nDepths = 10
fig = plt.figure(num=1)
fig.clf()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_uv_dist'] < roiRad]
    roi = roi[roi['scale_uv_dist'] > 0]
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

#%% Check on xy to uv mapping
from matplotlib.patches import Ellipse

frad = plt.figure()
floc = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]

    # recapitulate the fitting, which is a bit of overkill, but it gets us 
    # an accurate ellipse
    tgt_df = df[df['ctr-sur'] > 0]
    #tgt_df = df[df['scale_xy_dist'] <= 2]
    cov = np.cov(tgt_df['u'][df['scale_uv_dist'] < 2.2],
                 tgt_df['v'][df['scale_uv_dist'] < 2.2])
    com = (np.mean(tgt_df['u'][df['scale_uv_dist'] < 2.2]),
           np.mean(tgt_df['v'][df['scale_uv_dist'] < 2.2]))
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
    minx = np.min(df['u'].values)
    miny = np.min(df['v'].values)
    ax = frad.add_subplot(np.ceil(len(datasets)/2),2,iR+1)
    
    # Plot the radius determined by the normalized uv coordinates (this should be in SD of a 2D Gaussian fitted to the loc data)
    cmap = plt.cm.get_cmap('viridis')
    pcm = ax.scatter(df['u'],df['v'],c=df['scale_uv_dist'],s=10,cmap=cmap)
    plt.colorbar(pcm,ax=ax)
    ax.add_patch(ellipse)
    ax.patch.set_facecolor('r')
    ax.set_title(label+" radius: SD<2 Nvox = %d" %(np.sum(df['scale_uv_dist']<2)),fontsize=6)
    ax.axis('off')
    
    # Plot the ctr-sur betas
    plt.figure(floc)
    cmap_rev = cmap.reversed()
    plt.subplot(np.ceil(len(datasets)/2),2,iR+1)
    plt.scatter(df['u'],df['v'],c=df['ctr-sur'],s=10,cmap=cmap_rev)
    plt.colorbar()
    plt.title(label+" localizer",fontsize=6)

#%% Histograms of p-values
floc = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_uv_dist'] < roiRad]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(np.ceil(len(datasets)/2),2,iR+1)
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
    
    roi = df[df['scale_uv_dist'] < roiRad]
    if 'task pval' in roi.keys():
        roi = roi.rename(columns={'task pval':'task p-val'})
    plt.subplot(np.ceil(len(datasets)/2),2,iR+1)
    plt.hist(roi['task p-val'].values,bins=20,density=True)
    plt.title(label+" task p-val",fontsize=8)
    plt.xlabel("pval")
    plt.text(0.8,5,'< 0.05 = %d %%' %(100*np.sum(roi['task p-val'] <= 0.05)/len(roi['task p-val'])))
    plt.ylim([0,10])
    plt.xlim([0,1])
ftask.tight_layout(pad=0.1)

#%% Depth Histograms
# I want to see how much coverage we are getting through depth.
roiRad = 2
fdhist = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_uv_dist'] < roiRad]
    roi = df[df['scale_uv_dist'] < roiRad]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,np.ceil(len(datasets)/2),iR+1)
    plt.hist(roi['d'].values,bins=nDepths)
    plt.title(label)
    plt.xlabel("Normalize Depth WM -> GM")
    plt.ylabel("Num. Voxels")
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
fdhist.tight_layout(pad=0.1)
    
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
depth_groups = {'deep': [0.2,0.4], 'middle': [0.4,0.6], 'superficial': [0.6,0.8]}
depth_labels = ['superficial','middle','deep'] #put them in the right order
Ngroups = len(depth_groups.keys())
NROIs = len(all_data.keys())
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
    
#%% Plot Depth Profiles
    
statDetails = {'labels': ['sur', 'iso0', 'iso90', 'orth'],
                'colors': [[.7, .7, .7], 'red', 'darkviolet', 'orange']}
profile_method = 'bin' # bin or smooth
nDepths = 10
roiRad = 2.
#pick out ROIs where we're sure of localization

mask_pval = True #True if you want to mask voxels based on p-values
pthresh_vox = 0.05 #threshold for significance
if mask_pval:
    for k_i, key in enumerate(all_data.keys()):
        pvals = all_data[key]['task p-val']
        pval_mask = pvals < pthresh_vox
        mask_dict[key] = mask_dict[key]*pval_mask

roi_dict = {}
for key in all_data.keys():
    df = all_data[key]
    roi_dict[key] = df['scale_uv_dist'] < roiRad
useSI = False #use suppression index rather than differences (cond1 - cond2 / cond1 + cond2)
[hprofiles, hdiff, keep_depths, keep_rois, keep_std, fgm, odss, dSI] = plot_depth_profiles(all_data, roi_dict, statDetails, profile_method, nDepths, ['profiles','diff'],useSI,mask_dict,True)
      
#%% Centroid plots
# Let's take a look at raw voxel betas across depth by condition

roiRad = 2
nDepths = 10
Nsubj = len(all_data.keys())
for iStat in range(len(statDetails['labels'])):
    plt.figure(statDetails['labels'][iStat])
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = roi_dict[label] & mask_dict[label]
        roi = df[roi_idx]
        roi = df[df['scale_uv_dist'] < roiRad] # only very center, to be sure!
        roi = roi[roi['scale_uv_dist'] >= 0] #don't allow negative radii   
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
        
        #plot
        plt.subplot(2,int(np.ceil(len(all_data.keys())/2.)), 1 + iR)
        plt.scatter(roi[statDetails['labels'][iStat]],normDepths,s=0.5,c=statDetails['colors'][iStat])
        plt.plot(dataDict['profile']['avg'][iStat],avg_normDepths,color=statDetails['colors'][iStat])
        # plt.xlim([0, 7])
        plt.ylim([-0.02, 1.02])
        plt.xlim([-5,30])
        # plt.xlabel('BOLD % change', fontsize=fsize, color=lcolor)
        # plt.ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
        plt.title(label,FontSize = 8)
        plt.tick_params(labelsize = 6)
        
#calculate difference profiles
diffDetails = {}
diffDetails['statIDs'] = {'odss': [np.where(np.array(statDetails['labels']) == 'orth')[0][0],np.where(np.array(statDetails['labels']) == 'iso90')[0][0]],
                          'fgm': [np.where(np.array(statDetails['labels']) == 'iso90')[0][0],np.where(np.array(statDetails['labels']) == 'iso0')[0][0]],
                          'dsi': [np.where(np.array(statDetails['labels']) == 'orth')[0][0],np.where(np.array(statDetails['labels']) == 'iso0')[0][0]]}
diffDetails['colors'] = ['green','magenta','cyan']
                        #get stat IDs needed tocompute each difference profile
for iDiff, Diff in enumerate(diffDetails['statIDs'].keys()):
    plt.figure(iStat+iDiff+2)
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = roi_dict[label] & mask_dict[label]
        roi = df[roi_idx]
        roi = df[df['scale_uv_dist'] < roiRad] # only very center, to be sure!
        roi = roi[roi['scale_uv_dist'] >= 0] #don't allow negative radii    
        maxDepth = np.max(roi['d'])
        minDepth = np.min(roi['d'])
        normDepths = (roi['d']-minDepth)/maxDepth
        dataDict = makeProfile1D(roi['d'].values,
                             nDepths, #number of depths
                             roi[statDetails['labels']].values,
                             np.min(roi['d'].values), #min depth value
                             np.max(roi['d'].values), #max depth value
                             True) #Use LayNii values
        avg_normDepths = (dataDict['profile']['depth'] - np.nanmin(dataDict['profile']['depth']))/(np.nanmax(dataDict['profile']['depth']) - np.nanmin(dataDict['profile']['depth']))
        
        #compute differences
        statIDs = diffDetails['statIDs'][Diff]
        diff = roi[statDetails['labels'][statIDs[0]]] - roi[statDetails['labels'][statIDs[1]]]
        diff_avg = np.array(dataDict['profile']['avg'][statIDs[0]]) - np.array(dataDict['profile']['avg'][statIDs[1]])
        
        #plot
        plt.subplot(2,int(np.ceil(len(all_data.keys())/2.)), 1 + iR)
        plt.scatter(diff,normDepths,s=0.5,c=diffDetails['colors'][iDiff])
        plt.plot(diff_avg,avg_normDepths,color=diffDetails['colors'][iDiff])
        # plt.xlim([0, 7])
        plt.ylim([-0.02, 1.02])
        plt.xlim([-5,10])
        # plt.xlabel('BOLD % change', fontsize=fsize, color=lcolor)
        # plt.ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
        plt.title(label,FontSize = 8)
        plt.tick_params(labelsize = 6)
          
#%% Deconvolution

# measured_betas = mixing_matrix * true_betas
#
# mixing_matrix: nDepths x nDepths
# true_betas: nDepths x 1

#define point spread function
p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
Nbins = 10 #number of bins to use in this analysis

normalize_psf = False #True if you want to normalize the psf by the deepest layer  

decon_rois = depth_deconv(keep_rois,p2t_model,Nbins_model,Nbins,normalize_psf)

#%% now make some average plots

prop_err = False #True # do error propagation?
use_decon = True 
useSI = False
    
[fig, stat_avg, stat_std, depth_avg_norm, dx, dy, odss_avg, odss_std, fgm_avg, fgm_std, dSI_avg, dSI_std, Ttest_odss, Ttest_fgm, Ttest_dSI] = plot_avg_depth_profile(decon_rois, keep_depths, keep_std, statDetails, use_decon, prop_err, useSI, lcolor, fcolor)

#%% Plot each context modulation effect separately

pthresh = 0.05 #pvalue significance threshold

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

for iStat in [0,1]:
    p1.plot(stat_avg[iStat, :], depth_avg_norm, color=statDetails['colors'][iStat])
    p1.fill_betweenx(depth_avg_norm,
                    stat_avg[iStat, :] - stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                    stat_avg[iStat, :] + stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                    linewidth=0.,
                    alpha=0.4,
                    color=statDetails['colors'][iStat])
    p1.text(dx, dy + iStat*.07, statDetails['labels'][iStat],
            color=statDetails['colors'][iStat],
            fontsize=fsize-2)
p1.set_ylim([-0.02, 1.02])
p1.set_xlim([0, 7])
p1.set_xlabel('BOLD % change', fontsize=fsize, color=lcolor)
p1.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
p1.text(4, .05, 'n=%d hemis' %len(all_data), color=lcolor, fontsize=fsize*.5, fontstyle='italic')

if use_decon:
    fig.savefig('task_baselines_decon.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig.savefig('task_baselines.png', facecolor=fig.get_facecolor(), edgecolor='none')
   
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

for iStat in [1,3]:
    p1.plot(stat_avg[iStat, :], depth_avg_norm, color=statDetails['colors'][iStat])
    p1.fill_betweenx(depth_avg_norm,
                    stat_avg[iStat, :] - stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                    stat_avg[iStat, :] + stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                    linewidth=0.,
                    alpha=0.4,
                    color=statDetails['colors'][iStat])
    p1.text(dx, dy + iStat*.07, statDetails['labels'][iStat],
            color=statDetails['colors'][iStat],
            fontsize=fsize-2)
p1.set_ylim([-0.02, 1.02])
p1.set_xlim([0, 7])
p1.set_xlabel('BOLD % change', fontsize=fsize, color=lcolor)
p1.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
p1.text(4, .05, 'n=%d hemis' %len(all_data), color=lcolor, fontsize=fsize*.5, fontstyle='italic')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
p2.plot([0, 0], [0, 1], '--', color='gray')
p2.plot(dSI_avg, depth_avg_norm, color='magenta')
p2.fill_betweenx(depth_avg_norm,
                dSI_avg - dSI_std/np.sqrt(len(keep_depths)),
                dSI_avg + dSI_std/np.sqrt(len(keep_depths)),
                linewidth=0.,
                alpha=0.2,
                color='magenta')
top = dSI_avg + dSI_std/np.sqrt(len(keep_depths))
p2.plot(top[Ttest_dSI.pvalue <= pthresh] + 0.1,depth_avg_norm[Ttest_dSI.pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
p2.set_ylim([-0.02, 1.02])
p2.set_xlim([-.3, 2.2])
p2.text(1.15, .4-.19, '$\Delta$SI', color='magenta', fontsize=fsize-2)
p2.set_yticklabels([])
p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)

if use_decon:
    fig.savefig('task_dSI_decon.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig.savefig('task_dSI.png', facecolor=fig.get_facecolor(), edgecolor='none')
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

for iStat in [1,2]:
    p1.plot(stat_avg[iStat, :], depth_avg_norm, color=statDetails['colors'][iStat])
    p1.fill_betweenx(depth_avg_norm,
                    stat_avg[iStat, :] - stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                    stat_avg[iStat, :] + stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                    linewidth=0.,
                    alpha=0.4,
                    color=statDetails['colors'][iStat])
    p1.text(dx, dy + iStat*.07, statDetails['labels'][iStat],
            color=statDetails['colors'][iStat],
            fontsize=fsize-2)
p1.set_ylim([-0.02, 1.02])
p1.set_xlim([0, 7])
p1.set_xlabel('BOLD % change', fontsize=fsize, color=lcolor)
p1.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
p1.text(4, .05, 'n=%d hemis' %len(all_data), color=lcolor, fontsize=fsize*.5, fontstyle='italic')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
p2.plot([0, 0], [0, 1], '--', color='gray')
p2.plot(fgm_avg, depth_avg_norm, color='cyan')
p2.fill_betweenx(depth_avg_norm,
                fgm_avg - fgm_std/np.sqrt(len(keep_depths)),
                fgm_avg + fgm_std/np.sqrt(len(keep_depths)),
                linewidth=0.,
                alpha=0.2,
                color='cyan')
top = fgm_avg + fgm_std/np.sqrt(len(keep_depths))
p2.plot(top[Ttest_fgm.pvalue <= pthresh] + 0.1,depth_avg_norm[Ttest_fgm.pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
p2.set_ylim([-0.02, 1.02])
p2.set_xlim([-.3, 2.2])
p2.text(1.15, .4-.19, 'FGM', color='cyan', fontsize=fsize-2)
p2.set_yticklabels([])
p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)

if use_decon:
    fig.savefig('task_FGM_decon.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig.savefig('task_FGM.png', facecolor=fig.get_facecolor(), edgecolor='none')
    
fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6,4))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 14

p1 = fig.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    
for iStat in [2,3]:
    p1.plot(stat_avg[iStat, :], depth_avg_norm, color=statDetails['colors'][iStat])
    p1.fill_betweenx(depth_avg_norm,
                    stat_avg[iStat, :] - stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                    stat_avg[iStat, :] + stat_std[iStat, :]/np.sqrt(len(keep_depths)),
                    linewidth=0.,
                    alpha=0.4,
                    color=statDetails['colors'][iStat])
    p1.text(dx, dy + iStat*.07, statDetails['labels'][iStat],
            color=statDetails['colors'][iStat],
            fontsize=fsize-2)
p1.set_ylim([-0.02, 1.02])
p1.set_xlim([0, 7])
p1.set_xlabel('BOLD % change', fontsize=fsize, color=lcolor)
p1.set_ylabel(r'relative depth (WM $\rightarrow$ Pia)', fontsize=fsize, color=lcolor)
p1.text(4, .05, 'n=%d hemis' %len(all_data), color=lcolor, fontsize=fsize*.5, fontstyle='italic')

p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
p2.plot([0, 0], [0, 1], '--', color='gray')
p2.plot(odss_avg, depth_avg_norm, color='tab:green')
p2.fill_betweenx(depth_avg_norm,
                odss_avg - odss_std/np.sqrt(len(keep_depths)),
                odss_avg + odss_std/np.sqrt(len(keep_depths)),
                linewidth=0.,
                alpha=0.2,
                color='tab:green')
top = odss_avg + odss_std/np.sqrt(len(keep_depths))
p2.plot(top[Ttest_odss.pvalue <= pthresh] + 0.1,depth_avg_norm[Ttest_odss.pvalue <= pthresh],color='k',marker='$*$',linestyle='None')
p2.set_ylim([-0.02, 1.02])
p2.set_xlim([-.3, 2.2])
p2.text(1.15, .4-.19, 'ODSS', color='tab:green', fontsize=fsize-2)
p2.set_yticklabels([])
p2.set_xlabel(r'$\Delta$ BOLD %', fontsize=fsize, color=lcolor)

if use_decon:
    fig.savefig('task_ODSS_decon.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig.savefig('task_ODSS.png', facecolor=fig.get_facecolor(), edgecolor='none')
    
#%% Show patches
from matplotlib.patches import Ellipse

purple = np.array([.5, 0., 0.5])
yellow = np.array([1., 1., 0.])
middle = (yellow + purple)/2
# had a hard time deciding what color to use for radial distance
white = np.array([1., 1., 1.])
red = np.array([1., 0., 0.])
blue = np.array([0., 0., 1.])
gray = np.array([.5, .5, .5])
cmax = 5.
fig = plt.figure(figsize=(8,8))
fig.set_size_inches((8, 4))
fontsize = 8
fig.patch.set_facecolor(fcolor)

patchwidth = .13
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    # recapitulate the fitting, which is a bit of overkill, but it gets us 
    # an accurate ellipse
    tgt_df = df[df['ctr-sur'] > 0]
    #tgt_df = df[df['scale_xy_dist'] <= 2]
    cov = np.cov(tgt_df['u'][df['scale_uv_dist'] < 2.2],
                 tgt_df['v'][df['scale_uv_dist'] < 2.2])
    com = (np.mean(tgt_df['u'][df['scale_uv_dist'] < 2.2]),
           np.mean(tgt_df['v'][df['scale_uv_dist'] < 2.2]))
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
    minx = np.min(df['u'].values)
    miny = np.min(df['v'].values)
    aw = .8/len(all_data)
    ax = plt.axes([.07 + iR*1.1*aw, .5,  aw, .26])
    for iV, beta in enumerate(df['ctr-sur']):
        if df['ctr-sur F'][iV] > 6.66:
            weight = np.min((np.max((beta, -cmax)), cmax))/cmax
            if weight > 0:
                color = middle + weight*(yellow - middle)
            else:
                color = middle - weight*(purple - middle)
            ax.plot(df['u'][iV], df['v'][iV], '.', color=color)
    ax.set_xlim([minx, minx+18])
    ax.set_ylim([miny, miny+18])
    ax.add_patch(ellipse)
    ax.patch.set_facecolor(fcolor)
    plt.axis('off')
    ax.set_title(label, fontsize=fontsize, color=lcolor)
    
    # showdistance metric
    ax = plt.axes([.07 + iR*1.1*aw, .1,  aw, .26])
    for iV, beta in enumerate(df['scale_uv_dist']):
        if df['scale_uv_dist'][iV] < 2:
            weight = np.min((np.max((beta, -2)), 2))/2
            color = red + weight*(white-red)
            ax.plot(df['u'][iV], df['v'][iV], '.', color=color)
    ax.set_xlim([minx, minx+15])
    ax.set_ylim([miny, miny+15])
    
    ellipse = Ellipse(com,
                      width=2*2*np.sqrt(a),
                      height=2*2*np.sqrt(b),
                      angle=180*theta/np.pi,
                      zorder=100, alpha=1., edgecolor='r', facecolor='None')
    ax.add_patch(ellipse)
    ax.patch.set_facecolor(fcolor)
    plt.axis('off')
    
#%% Visualize Betas across Surface
import scipy.interpolate as sciInterp

red = np.array([1,0,0]); #color
cmax = 10.; #max beta weight
patchwidth = .13
cond = 'iso0'
pthresh = 0.01
fig = plt.figure(figsize=(8,8))
fig.set_size_inches((8, 4))
fontsize = 8
fig.patch.set_facecolor(fcolor)

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    if 'task pval' in df.columns:
        df['task p-val'] = df['task pval']
        all_data[label] = df
    
    # Plot iso
    minx = np.min(df['u'].values)
    miny = np.min(df['v'].values)
    aw = .8/len(all_data)
    ax = plt.axes([.07 + iR*1.1*aw, .5,  aw, .26])
    for iV, beta in enumerate(df[cond]):
        if df['task p-val'][iV] < pthresh:
            weight = np.min((np.max((beta, -cmax)), cmax))/cmax
            weight = (weight + 1)/2; #make value between 0 and 1
            if weight > 0:
                color = weight*red
            else:
                color = weight*red
            ax.plot(df['u'][iV], df['v'][iV], '.', color=color)
    ax.set_xlim([minx, minx+18])
    ax.set_ylim([miny, miny+18])
    ax.patch.set_facecolor(fcolor)
    plt.axis('off')
    ax.set_title(label, fontsize=fontsize, color=lcolor)
    
#Smooth 2D space with kernel
fig2 = plt.figure()
smooth_factor = 0.5
resamp_factor = 100
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    # Plot iso
    minx = np.min(df['u'].values)
    maxx = np.max(df['v'].values)
    miny = np.min(df['u'].values)
    maxy = np.max(df['v'].values)
    x_interp = np.linspace(minx,maxx,resamp_factor)
    y_interp = np.linspace(miny,maxy,resamp_factor)
    # beta_interp = sciInterp.interp2d(df['x'].values,df['y'].values,df[cond].values,kind='linear')
    # beta_resamp = beta_interp(x_interp,y_interp)
    # beta_resamp[beta_resamp > cmax] = cmax #truncate
    # beta_resamp[beta_resamp < -cmax] = -cmax
    
    # 2D blurring
    betas = df[cond].values[df['task p-val'] < pthresh]
    betas_x = df['u'].values[df['task p-val'] < pthresh]
    betas_y = df['v'].values[df['task p-val'] < pthresh]
    beta_resamp = smoothen2D(betas,np.array([betas_x,betas_y]),np.array([x_interp,y_interp]),smooth_kernel2D,smooth_factor)

    # plot
    plt.subplot(1,len(all_data),iR+1)
    plt.imshow(beta_resamp,cmap='Reds')
    plt.colorbar()
    plt.title(label+': '+cond)
    
#%% Across surface!
# smoothing according to line 441 in analysisROI_dev
#with smoothing
plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(8.75, 4))
fig1.set_size_inches((8, 4))
fontsize = 8
fig1.patch.set_facecolor(fcolor)


radMax = 4
nRadii = 20
ymax = 6
highlight = ['sur', 'iso0'] #['orth', 'iso90']#'orth'
depth_labels = ['deep', 'middle', 'superficial']
depthBoundaries = np.array([[0,0.3],[0.3,0.6],[0.6,1]])
all_profiles = {depth_label: {label: [] for label in statDetails['labels']} for depth_label in depth_labels}

for iR, label in enumerate(all_data.keys()):
    for iD in range(3):
        df = all_data[label][mask_dict[label]]
        df = df[(df['d'] >=depthBoundaries[iD,0]) & (df['d'] <depthBoundaries[iD,1])]
        aw = .8/len(all_data)
        plt.figure(1)
        ax = plt.axes([.07 + iR*1.1*aw, .14 + iD*.3,  aw, .2])
        ax2 = ax.twinx()
        for iStat in range(len(statDetails['labels'])):
            alpha = 1.
            if highlight:
                if statDetails['labels'][iStat] in highlight:
                    alpha = 1.
                else:
                    alpha = 0.2
            coef = df[statDetails['labels'][iStat]].values
            x = df['scale_uv_dist'].values
            #smooth data
            coef_smooth, x_smooth = smoothen(coef, x)
            if iR < 5:
                all_profiles[depth_labels[iD]][statDetails['labels'][iStat]].append(coef_smooth)
            # dataDict = makeProfile1D(x, nRadii, coef)
            # plt.plot(dataDict['profile']['depth'],
            #          dataDict['profile']['avg'][0],
            #          '--',
            #          color=statDetails['colors'][iStat])

            ax.plot(x_smooth, coef_smooth, color=statDetails['colors'][iStat], alpha=alpha)
            ax.set_ylim([0, ymax])
            ax.set_xticks([0, 1, 2, 3])
            if iD == 2:
                ax.set_title(label, fontsize=8, color=lcolor) 
            ax.plot([2, 2], [0, ymax], '--', color='gray', alpha=0.15)
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
            else:
                ax.set_xticklabels([])
                
            # Now plot ctr - sur Fstat across surface
            if plot_Fstat:
                Fstat = df['ctr-sur F'].values
                x = df['scale_uv_dist'].values
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
                    
#%% Loc Profiles across surface!
# smoothing according to line 441 in analysisROI_dev
#with smoothing
plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(8.75, 4))
fig1.set_size_inches((8, 4))
fontsize = 8
fig1.patch.set_facecolor(fcolor)
locStatDetails = {'labels':['ctr-sur'], 'colors':'black'}


radMax = 4
nRadii = 20
ymax = 6
highlight = ['ctr-sur'] #['orth', 'iso90']#'orth'
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
            x = df['scale_uv_dist'].values
            #smooth data
            coef_smooth, x_smooth = smoothen(coef, x)
            if iR < 5:
                all_profiles[depth_labels[iD]][locStatDetails['labels'][iStat]].append(coef_smooth)
            # dataDict = makeProfile1D(x, nRadii, coef)
            # plt.plot(dataDict['profile']['depth'],
            #          dataDict['profile']['avg'][0],
            #          '--',
            #          color=statDetails['colors'][iStat])

            ax.plot(x_smooth, coef_smooth, color=locStatDetails['colors'][iStat], alpha=alpha)
            ax.set_ylim([0, ymax])
            ax.set_xticks([0, 1, 2, 3])
            if iD == 2:
                ax.set_title(label, fontsize=8, color=lcolor) 
            ax.plot([2, 2], [0, ymax], '--', color='gray', alpha=0.15)
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
            else:
                ax.set_xticklabels([])
                
            # Now plot ctr - sur Fstat across surface
            if plot_Fstat:
                Fstat = df['ctr-sur F'].values
                x = df['scale_uv_dist'].values
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
                    
#%% Average surface profiles

fig = plt.figure(figsize=(4, 6))
fig.set_size_inches((4, 6))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12

for iD, depth_label in enumerate(all_profiles.keys()):
    print(depth_label)
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    for iStat, label in enumerate(statDetails['labels']):
        nProfiles = len(all_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [0, 6], '--', color=(.3, .3, .3))

        p.plot(x_smooth, stat_avg, color=statDetails['colors'][iStat])
        p.fill_between(x_smooth,
                        stat_avg - stat_std/np.sqrt(nProfiles),
                        stat_avg + stat_std/np.sqrt(nProfiles),
                        linewidth=0.,
                        alpha=0.4,
                        color=statDetails['colors'][iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, statDetails['labels'][iStat],
                color=statDetails['colors'][iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([0, 6])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)
        
#%% FGM and ODSS Surface Profiles

plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

fig1 = plt.figure(figsize=(8.75, 4))
fig1.set_size_inches((8, 4))
fontsize = 8
fig1.patch.set_facecolor(fcolor)

useSI = False #if true, use suppression index ratherthan difference (cond1 - cond2 / cond1 + cond2)
radMax = 4
nRadii = 20
ymax = 4
highlight = ['odss','fgm','dsi'] #['sur', 'iso0'] #['orth', 'iso90']#'orth'
depth_labels = ['deep', 'middle', 'superficial']
diff_labels = ['odss','fgm','dsi']
diff_colors = ['green','violet','cyan']
diffDetails = {'labels': diff_labels, 'colors': diff_colors}
all_diff_profiles = {depth_label: {label: [] for label in diff_labels} for depth_label in depth_labels}

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    #compute odss, fgm, and dsi
    if useSI:
        df['odss'] = (df['orth'] - df['iso90'])/(df['orth'] + df['iso90'])
        df['fgm'] = (df['iso90'] - df['iso0'])/(df['iso90'] + df['iso0'])
        df['dsi'] = (df['orth'] - df['iso0'])/(df['orth'] + df['iso0'])
    else:
        df['odss'] = df['orth'] - df['iso90']
        df['fgm'] = df['iso90'] - df['iso0']
        df['dsi'] = df['orth'] - df['iso0']
    
    all_data[label] = df
    
    for iD in range(3):
        df = all_data[label][mask_dict[label]]
        if iD == 0:
            df = df[(df['d'] >=0) & (df['d'] <0.3)]
        elif iD == 1:
            df = df[(df['d'] >=0.3) & (df['d'] <0.6)]
        elif iD == 2:
            df = df[(df['d'] >=0.6) & (df['d'] <=1)]
        aw = .8/len(all_data)
        plt.figure(1)
        ax = plt.axes([.07 + iR*1.1*aw, .14 + iD*.3,  aw, .2])
        ax2 = ax.twinx()
        for iStat in range(len(diffDetails['labels'])):
            alpha = 1.
            if highlight:
                if diffDetails['labels'][iStat] in highlight:
                    alpha = 1.
                else:
                    alpha = 0.2
            coef = df[diffDetails['labels'][iStat]].values
            x = df['scale_uv_dist'].values
            #smooth data
            coef_smooth, x_smooth = smoothen(coef, x)
            if iR < 5:
                all_diff_profiles[depth_labels[iD]][diffDetails['labels'][iStat]].append(coef_smooth)

            ax.plot(x_smooth, coef_smooth, color=diffDetails['colors'][iStat], alpha=alpha)
            ax.set_ylim([-4, ymax])
            ax.set_xticks([0, 1, 2, 3])
            if iD == 2:
                ax.set_title(label, fontsize=8, color=lcolor) 
            ax.plot([2, 2], [0, ymax], '--', color='gray', alpha=0.15)
            fix_axes(ax, lcolor=lcolor, fcolor=fcolor)
            if iR > 0:
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
    
            else:
                plt.ylabel(r'$\beta$, %s' %depth_labels[iD], fontsize = 8, color=lcolor)
                # if iD == 2:
                    # ax.set_yticks([0, 2, 4, 6, 8, 10, 12])
                    # ax.set_yticklabels(['0', '2', '4', '6', '8', '10', '12'], fontsize=8, color=lcolor)
                # else:
                    # ax.set_yticks([0, 2, 4, 6])
                    # ax.set_yticklabels(['0', '2', '4', '6'], fontsize=8, color=lcolor)
            ax.set_xticklabels(['0', '', '2', ''], fontsize=8, color=lcolor)
            if iR == 0:
                if iD == 0:
                    ax.set_xlabel("$\sigma$ from ROI center", fontsize = 8, color=lcolor)
            else:
                ax.set_xticklabels([])
                
#%% Average Surface Difference Profiles

fig = plt.figure(figsize=(4, 6))
fig.set_size_inches((4, 6))
fig.patch.set_facecolor(fcolor)

fig.clf()
fsize = 12

for iD, depth_label in enumerate(all_diff_profiles.keys()):
    print(depth_label)
    p = fig.add_axes([.15, .1 + iD*.3, .7, .25])
    fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
    for iStat, label in enumerate(diffDetails['labels']):
        nProfiles = len(all_diff_profiles[depth_label][label])
        stat_avg = np.mean(np.asarray(all_diff_profiles[depth_label][label]), axis=0)
        stat_std = np.std(np.asarray(all_diff_profiles[depth_label][label]), axis=0)

        p.plot([2, 2], [0, 6], '--', color=(.3, .3, .3))

        p.plot(x_smooth, stat_avg, color=diffDetails['colors'][iStat])
        p.fill_between(x_smooth,
                        stat_avg - stat_std/np.sqrt(nProfiles),
                        stat_avg + stat_std/np.sqrt(nProfiles),
                        linewidth=0.,
                        alpha=0.4,
                        color=diffDetails['colors'][iStat])
        if iD == 1:
            p.text(3.5, 5.5-iStat*.7, diffDetails['labels'][iStat],
                color=diffDetails['colors'][iStat],
                fontsize=fsize - 1)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fsize, color=lcolor)
        else:
            p.set_xticklabels([])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        p.set_ylim([-1, 2])
        p.set_ylabel('BOLD % change', fontsize=fsize, color=lcolor)