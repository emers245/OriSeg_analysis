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
from oriseg_funcs_v2 import *

plt.close('all')
    
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code'
#datasets = glob.glob(os.path.join(mainDir, 'roi_data', 'pnr???_??_??_?????.csv'))
#or exclude pnr102 for bad ellipse fit, pnr161 for missing superfical data, and rh of pn510 for no stria
# datasets = ['/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V1_lh_rad10.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V1_rh_rad10.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_lh_rad10.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_rh_rad10.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V1_lh_rad10.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_lh_rad10.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_rh_rad10.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr756_V1_lh_rad10.csv']
datasets = ['/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V1_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr256_V1_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr328_V1_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V1_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr510_V1_rh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_lh_rad10.csv',
   '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr739_V1_rh_rad10.csv',
  '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr756_V1_lh_rad10.csv',
  '/Users/joe/Documents/Olman_Lab/OriSeg/code/roi_data/pnr756_V1_rh_rad10.csv']
# datasets = ['/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr256_lh.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr256_rh.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr328_lh.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr328_rh.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr510_lh.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr510_rh.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr739_lh.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr739_rh.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr756_lh.csv',
#   '/Users/joe/Documents/Olman_Lab/OriSeg/code/Analysis_102022/roi_data/pnr756_rh.csv']
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
fig = plt.figure(num=1)
fig.clf()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = roi[roi['scale_xy_dist'] > 0]
    dataDict = makeProfile1D(roi['z'].values,
                             10, 
                             roi['t1'].values,
                             0, #min depth val
                             20, #max depth val
                             False) #do not use laynii depths
    
    plt.subplot(int(np.ceil(len(all_data.keys())/2.)), 2, 1 + iR)
    plt.plot(dataDict['profile']['depth'],
             dataDict['profile']['avg'][0])
    plt.title('%s (%d vox)' %(label, len(roi)), fontsize=8)
    
#%% Check on xy mapping
from matplotlib.patches import Ellipse

frad = plt.figure()
floc = plt.figure()
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
    ellipse = Ellipse(com,
                      width=2*2*np.sqrt(a),
                      height=2*2*np.sqrt(b),
                      angle=180*theta/np.pi,
                      zorder=100, alpha=1., edgecolor='r', facecolor='None')
    # show localizer data
    minx = np.min(df['x'].values)
    miny = np.min(df['y'].values)
    ax = frad.add_subplot(np.ceil(len(datasets)/2),2,iR+1)
    
    # Plot the radius determined by the normalized uv coordinates (this should be in SD of a 2D Gaussian fitted to the loc data)
    cmap = plt.cm.get_cmap('viridis')
    pcm = ax.scatter(df['x'],df['y'],c=df['scale_xy_dist'],s=10,cmap=cmap)
    plt.colorbar(pcm,ax=ax)
    ax.add_patch(ellipse)
    ax.patch.set_facecolor('r')
    ax.set_title(label+" radius: SD<2 Nvox = %d" %(np.sum(df['scale_xy_dist']<2)),fontsize=6)
    ax.axis('off')
    
    # Plot the ctr-sur betas
    plt.figure(floc)
    cmap_rev = cmap.reversed()
    plt.subplot(np.ceil(len(datasets)/2),2,iR+1)
    plt.scatter(df['x'],df['y'],c=df['ctr-sur'],s=10,cmap=cmap_rev)
    plt.colorbar()
    plt.title(label+" localizer",fontsize=6)

#%% Histograms of p-values
floc = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
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
    
    roi = df[df['scale_xy_dist'] < roiRad]
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
roiRad = 4
fdhist = plt.figure()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = df[df['scale_xy_dist'] < roiRad]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval':'loc p-val'})
    plt.subplot(2,np.ceil(len(datasets)/2),iR+1)
    plt.hist(roi['z'].values,bins=10)
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
    mnv = all_data[key]['task stdev_xerrts'].values**2 #the variance from the xerrts are already normalized by the mean
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
    plt.scatter(all_data[key]['x'],all_data[key]['y'],s = 1,c = (all_data[key]['task stdev_xerrts'].values),cmap='Reds')
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
    plt.title(key+" Var/Mean th=%.1f" %(mnv_thresh),fontsize=5)
    plt.xticks(ticks = [], labels = [])
    plt.yticks(ticks = [], labels = [])
    plt.clim([0,100])
    
    #plot histograms
    plt.figure("hist")
    plt.subplot(4,len(all_data.keys()),(k_i+1))
    plt.hist(mnv,bins=np.linspace(-100,100,200))
    plt.title(key+" Var/Mean",fontsize=6)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    
    plt.subplot(4,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.hist(avg_betas,bins=np.linspace(-10,10,200))
    plt.title(key+" Mean $\\beta$",fontsize=6)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    
    plt.subplot(4,len(all_data.keys()),(k_i+1)+2*len(all_data.keys()))
    plt.hist(all_data[key]['task stdev_xerrts'].values,bins=np.linspace(-10,10,200))
    plt.title(key+" SD/Mean",fontsize=6)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    
    plt.subplot(4,len(all_data.keys()),(k_i+1)+3*len(all_data.keys()))
    plt.hist(mnv[mnv_mask],bins=np.linspace(-100,100,200))
    plt.title(key+" Var/Mean th=%.1f" %(mnv_thresh),fontsize=5)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    
    #check depth histograms
    nD = np.max(all_data[key]['z'])+1
    frac_included = np.zeros((int(nD),))
    for di in range(int(nD)):
        dmasked = all_data[key]['z'][mnv_mask]
        frac_included[di] = np.sum(dmasked == di)/np.sum(all_data[key]['z'] == di)
    plt.figure("depth")
    plt.subplot(2,len(all_data.keys()),(k_i+1))
    nD = np.max(all_data[key]['z'])+1
    plt.hist(all_data[key]['z'],bins=np.arange(0,nD))
    plt.title(key+" Depth Hist",fontsize=5)
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    plt.hist(all_data[key]['z'][mnv_mask],bins=np.arange(0,nD),alpha=0.7)
    plt.legend(["unmasked","masked"],fontsize=4)
    
    plt.subplot(2,len(all_data.keys()),(k_i+1)+len(all_data.keys()))
    plt.bar(np.arange(0,nD),frac_included,width=1,color='tomato')
    plt.xticks(ticks = None, labels = None, fontsize=4)
    plt.yticks(ticks = None, labels = None, fontsize=4)
    plt.xlabel("Depth (WM --> Pia)",fontsize=5)
    plt.ylim([0,1])
    plt.ylabel("Frac Included",fontsize=5)
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(np.abs(mnv) < mnv_thresh),np.size(mnv),key))
    
    mask_dict[key] = mnv_mask #add mask to dictionary
    all_data[key]['mnv'] = mnv #add mnv to all_data
    
fhist.tight_layout(pad=0.5)
fmap.tight_layout(pad=0.5)
fdepth.tight_layout(pad=0.5)

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
        plt.title(key_name+" MNV "+depth_label,fontsize=5)
        plt.xticks(ticks = [], labels = [])
        plt.yticks(ticks = [], labels = [])
        
        #plot hist
        plt.subplot(Ngroups*4,NROIs,(k_i+1)+(2*NROIs)+(4*d_i*NROIs))
        plt.hist(mnv[level_mask],bins=np.linspace(0,100,100))
        plt.xticks(ticks = None, labels = None, fontsize=4)
        plt.yticks(ticks = None, labels = None, fontsize=4)
        plt.xlabel("Var/Mean",fontsize=5)
    
# Plot blurred data
# plt.figure()
# smooth_factor = 0.1
# resamp_factor = 100
# mnv_lim = [-100,100] #max limit for mnv for plotting purposes
# pthresh = 1
# for iR, label in enumerate(all_data.keys()):
#     df = all_data[label]
    
#     # Calculate mean beta
#     tmp = df['iso0'].values
#     betas = np.zeros([len(tmp),len(conditions)])
#     for c_i,con in enumerate(conditions):
#         tmp = df[con].values
#         betas[:,c_i] = tmp
#     avg_betas = np.mean(betas,1)
    
#     # Plot mean normalize variance
#     minx = np.min(df['x'].values)
#     maxx = np.max(df['x'].values)
#     miny = np.min(df['y'].values)
#     maxy = np.max(df['y'].values)
#     x_interp = np.linspace(minx,maxx,resamp_factor)
#     y_interp = np.linspace(miny,maxy,resamp_factor)
#     # beta_interp = sciInterp.interp2d(df['x'].values,df['y'].values,df[cond].values,kind='linear')
#     # beta_resamp = beta_interp(x_interp,y_interp)
#     # beta_resamp[beta_resamp > cmax] = cmax #truncate
#     # beta_resamp[beta_resamp < -cmax] = -cmax
    
#     # 2D blurring
#     mnv = (df['task stdev_xerrts'].values**2)/avg_betas #mean normalized variance
#     mnv = mnv[df['ctr-sur pvals'] < pthresh]
#     mnv_x = df['x'].values[df['ctr-sur pvals'] < pthresh]
#     mnv_y = df['y'].values[df['ctr-sur pvals'] < pthresh]
#     mnv_resamp = smoothen2D(mnv,np.array([mnv_x,mnv_y]),np.array([x_interp,y_interp]),smooth_kernel2D,smooth_factor)
#     mnv_resamp[mnv_resamp > mnv_lim[1]] = mnv_lim[1] #truncate at limit for plotting purposes
#     mnv_resamp[mnv_resamp < mnv_lim[0]] = mnv_lim[0]

#     # plot
#     plt.subplot(1,len(all_data),iR+1)
#     plt.imshow(mnv_resamp,cmap='Reds')
#     plt.colorbar()
#     plt.title(label+': Mean Normalized Variance',fontsize=4)
    
#%% Compute Depth Profiles
    
statDetails = {'labels': ['sur', 'iso0', 'iso90', 'orth', 'ctr', 'sur.1', 'ctr-sur'],
                'colors': [[.7, .7, .7], 'red', 'darkviolet', 'orange', 'yellow', 'purple', 'black']}
diffDetails = {'dSI': ['orth','iso0'], 'FGM': ['iso90','iso0'], 'OTSS':['orth','iso90']}
diffColors = ['tab:cyan','tab:magenta','tab:green']
profile_method = 'bin' # bin or smooth
nDepths = 10
roiRad = 2
#pick out ROIs where we're sure of localization
roi_dict = {}
for key in all_data.keys():
    df = all_data[key]
    roi_dict[key] = df['scale_xy_dist'] < roiRad
useSI = False #use suppression index rather than differences (cond1 - cond2 / cond1 + cond2)
depthProfiles = compute_all_depth_profiles(all_data,roi_dict,statDetails,profile_method,nDepths,mask_dict,depthParam='z',radialParam = 'scale_xy_dist',spec_Drange='MinMax')
diffProfiles = compute_diff_profiles(all_data,roi_dict,statDetails,diffDetails,profile_method,nDepths,useSI,mask_dict,depthParam='z',radialParam='scale_xy_dist',spec_Drange='MinMax')

#%% Plot depth profiles for each subject

hprofiles = plt.figure('profiles')
hloc = plt.figure('loc')
hdiff = plt.figure('diff')

taskProfiles = statDetails['labels'][0:4]
locProfiles = statDetails['labels'][4:]
taskColors = statDetails['colors'][0:4]
locColors = statDetails['colors'][4:]
taskDiffs = list(diffDetails.keys())

for iR, roiID in enumerate(all_data.keys()):
    
    plt.figure('profiles')
    plt.subplot(2, int(np.ceil(len(all_data.keys())/2.)), 1 + iR) 
    plt.figure('loc')
    plt.subplot(2, int(np.ceil(len(all_data.keys())/2.)), 1 + iR)
    plt.figure('diff')
    plt.subplot(2, int(np.ceil(len(all_data.keys())/2.)), 1 + iR)
    
    for iStat, stat in enumerate(taskProfiles):
        plt.figure('profiles')
        plt.plot(depthProfiles[stat]['avg'][iR],
                         depthProfiles[stat]['depths'][iR],
                         color=taskColors[iStat])
        if profile_method == 'bin': #only plot error bar if we have error bars (bin method)
            plt.fill_betweenx(depthProfiles[stat]['depths'][iR],
                        depthProfiles[stat]['avg'][iR] - depthProfiles[stat]['stdev'][iR]/np.sqrt(depthProfiles[stat]['N'][iR]),
                        depthProfiles[stat]['avg'][iR] + depthProfiles[stat]['stdev'][iR]/np.sqrt(depthProfiles[stat]['N'][iR]),
                        linewidth=0.,
                        alpha=0.4,
                        color=taskColors[iStat])
            
    for iStat, stat in enumerate(locProfiles):
        plt.figure('loc')
        plt.plot(depthProfiles[stat]['avg'][iR],
                         depthProfiles[stat]['depths'][iR],
                         color=locColors[iStat])
        if profile_method == 'bin': #only plot error bar if we have error bars (bin method)
            plt.fill_betweenx(depthProfiles[stat]['depths'][iR],
                        depthProfiles[stat]['avg'][iR] - depthProfiles[stat]['stdev'][iR]/np.sqrt(depthProfiles[stat]['N'][iR]),
                        depthProfiles[stat]['avg'][iR] + depthProfiles[stat]['stdev'][iR]/np.sqrt(depthProfiles[stat]['N'][iR]),
                        linewidth=0.,
                        alpha=0.4,
                        color=locColors[iStat])
        
    for iDiff, diff in enumerate(taskDiffs):
        plt.figure('diff')
        plt.plot(diffProfiles[diff]['avg'][iR],
                         diffProfiles[diff]['depths'][iR],
                         color=diffColors[iDiff])
        if profile_method == 'bin':
            plt.fill_betweenx(diffProfiles[diff]['depths'][iR],
                        diffProfiles[diff]['avg'][iR] - diffProfiles[diff]['stdev'][iR]/np.sqrt(diffProfiles[diff]['N'][iR]),
                        diffProfiles[diff]['avg'][iR] + diffProfiles[diff]['stdev'][iR]/np.sqrt(diffProfiles[diff]['N'][iR]),
                        linewidth=0.,
                        alpha=0.4,
                        color=diffColors[iDiff])
    plt.figure('profiles')
    plt.title('%s (%d vox)' %(roiID, np.sum(depthProfiles[stat]['N'][iR])), fontsize=8)
    plt.figure('loc')
    plt.title('%s (%d vox)' %(roiID, np.sum(depthProfiles[stat]['N'][iR])), fontsize=8)
    plt.figure('diff')
    plt.title('%s (%d vox)' %(roiID, np.sum(diffProfiles[diff]['N'][iR])), fontsize=8)
                
#%% Deconvolution

#reformat data to fit decon_rois specs
keep_rois = np.zeros((NROIs,len(statDetails['labels']),nDepths))
for iR, roiID in enumerate(all_data.keys()):
    for iStat, stat in enumerate(statDetails['labels']):
        keep_rois[iR,iStat,:] = depthProfiles[stat]['avg'][iR]

#define point spread function
p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
Nbins = nDepths #number of bins to use in this analysis

normalize_psf = False #True if you want to normalize the psf by the deepest layer  

decon_rois = depth_deconv(keep_rois,p2t_model,Nbins_model,Nbins,normalize_psf)

#now put back in dictionary
for iStat, stat in enumerate(statDetails['labels']):
    depthProfiles[stat]['avg_decon'] = np.squeeze(np.array(decon_rois)[:,iStat,:])

#%% now make some average plots

prop_err = False # do error propagation?
use_decon = True
useSI = False

taskStats = statDetails['labels'][0:4]
taskColors = statDetails['colors'][0:4]
taskDiffs = list(diffDetails.keys())[0:3]
locStats = statDetails['labels'][4:]
locColors = statDetails['colors'][4:]
locDiffs = []

[avgTaskProfiles, avgTaskDiffs] = compute_avg_depth_profile(depthProfiles,statDetails,diffDetails,taskStats,taskDiffs,use_decon,prop_err,useSI)  
[avgLocProfiles, avgLocDiffs] = compute_avg_depth_profile(depthProfiles,statDetails,diffDetails,locStats,locDiffs,use_decon,prop_err,useSI)

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
plot_avg_diff_profile(p2,avgTaskDiffs,taskDiffs,diffColors,ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI)

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
xlim = [-0.3,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['dSI'],['tab:cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh)

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
plot_avg_depth_profile(p1,avgTaskProfiles,['iso0','iso90'],['red','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

#FGM
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.3,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['FGM'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh)

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
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.3,1.5]
p2 = fig.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
plot_avg_diff_profile(p2,avgTaskDiffs,['OTSS'],['tab:green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh)
    
#%% Stats: Comparisons across depths

dSI = np.array(dSI)
deep = np.mean([dSI[0,:],dSI[1,:],dSI[2,:]],axis=0)
middle = np.mean([dSI[3,:],dSI[4,:],dSI[5,:]],axis=0)
superficial = np.mean([dSI[6,:],dSI[7,:],dSI[8,:]],axis=0)
#[Fanova,panova] = stats.f_oneway(dSI[0,:],dSI[1,:],dSI[2,:],dSI[3,:],dSI[4,:],dSI[5,:],dSI[6,:],dSI[7,:],dSI[8,:],dSI[9,:])
[Fanova,panova] = stats.f_oneway(deep,middle,superficial)
#p_ttest = np.zeros([len(all_data.keys()),len(all_data.keys())])
p_ttest = np.zeros([3,3])
#for iR in range(len(all_data.keys())):
    #for jR in range(len(all_data.keys())):
dms = np.array([deep,middle,superficial])
for iR in range(3):
    for jR in range(3):
        p_ttest[iR,jR] = stats.ttest_ind(dms[iR,:],dms[jR,:])[1]
        
plt.imshow(p_ttest < 0.05)

#%% Estimating FF, FB, and HC contributions at each depth for each condition

# do this separately for each subject

# A = design matrix (nStim x nFactors), 
# x = concatenated responses (nFactors x nLayers), 
# y = data (nStim x nLayers)
#
# A * x = y
#  so
# x = A.I * x


ff = {'sur': 0, 'iso0': 1, 'iso90': 1, 'orth': 1}
hc = {'sur': 0, 'iso0': 1, 'iso90': 0, 'orth': 0}
fb = {'sur': 1, 'iso0': 0, 'iso90': 1, 'orth': 1}
nMod = 3
nStim = 4

# Same design for everyone
A = np.matrix(np.zeros((4, 3)))
B = np.zeros((4, 3))
for iStat, stat in enumerate(['sur', 'iso0', 'iso90', 'orth']):
    for iM, mod in enumerate([ff, hc, fb]):
        A[iStat, iM] = mod[stat]

# cherry-pick 3 depths to represent deep, middle, and superficial
if len(keep_rois[0][0]) == 10:
    depths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] #[1, 4, 7]
elif len(keep_rois[0][0]) == 5:
    depths = [0, 2, 4]
nDepths = len(depths)

# a matrix to catch the answers
nSubj = len(keep_rois)
answers = np.zeros((nMod, nDepths, nSubj))

# we'll see if our work is plausible by reconstructing
inputs = np.zeros((nStim, nDepths, nSubj))
reconstructions = np.zeros((nStim, nDepths, nSubj))
input_variance = np.zeros((nSubj,))
residual_variance = np.zeros((nSubj,))

if use_decon: # set in the average profile cell
    data_list = decon_rois.copy()
else:
    data_list = keep_rois.copy()

# cruise through and do the regressions
for iSubj in range(nSubj):
    y = np.matrix(np.zeros((nStim, nDepths)))
    for iStat, stat in enumerate(statDetails['labels']):
        for iD, depth in enumerate(depths):
            y[iStat, iD] = data_list[iSubj][iStat][depth]
    input_variance[iSubj] = np.var(y)
    x = A.I*y
    answers[:, :, iSubj] = x
    # reconstruct profiles from answers
    yest = A*x
    residual_variance[iSubj] = np.var(yest - y)
    reconstructions[:, :, iSubj] = yest
    inputs[:, :, iSubj] = y
    # plt.subplot(nSubj, 2, iSubj*2 + 1)
    # plt.plot(y.transpose())
    # plt.subplot(nSubj, 2, iSubj*2 + 2)
    # plt.plot(yest.transpose())
    
#see what we got for x!
modColors = ['lightgray', 'lightgreen', 'cyan']
# for iSubj in range(nSubj):
#     plt.subplot(2, 7, iSubj+1)
#     plt.imshow(answers[:, :, iSubj].transpose(), vmin=-3, vmax=3, cmap='plasma')
#     plt.title('%2.2f / %2.2f' %(input_variance[iSubj], residual_variance[iSubj]), fontsize=8)
#     plt.axis('off')

fig = plt.figure(figsize=(6, 4))
fig.set_size_inches((6, 4))
fig.patch.set_facecolor(fcolor)

ax = plt.subplot(1, 2, 1)
ax.imshow(A.T, cmap='gray', vmin=-.5, vmax=2)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['ff', 'hc', 'fb'], color=lcolor)
ax.set_xticks(range(4))
ax.set_xticklabels(statDetails['labels'], color=lcolor)
ax.set_title('avg %d%% var explained' %(int(100*(1-np.mean(np.array(residual_variance)/np.array(input_variance))))),
             color=lcolor)

ax = plt.axes([.7, .2, .25, .7])
ax.plot([0, 0], [0, 1], '--', color='gray', linewidth=.5)
for iM, mod in enumerate(['ff', 'hc', 'fb']):
    ax.plot(np.mean(answers[iM, :, :], axis=1).transpose(),
             np.linspace(0, 1, nDepths),
             color=modColors[iM])
    ax.text(2.2, .5-.1*iM, mod, color=modColors[iM])
ax.set_xlim([-.5, 2.5])
ax.set_xticks(np.arange(0, 2.5, 1.))
ax.set_xticklabels(np.arange(0, 2.5, 1.), color=lcolor)
ax.set_ylabel('Relative depth', color=lcolor)
ax.set_xlabel('% change', color=lcolor)
fix_axes(ax, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    title = 'model on deconvolved data'
else:
    title = 'model on orig data'
ax.set_title('Estimated weights', color=lcolor)

fig.savefig('model_fits.png', facecolor=fig.get_facecolor(), edgecolor='none')
#%% An iteration to see what kind of A for middle layers does best prediction
plt.figure()
ff = {'sur': 0, 'iso0': 1, 'iso90': 1, 'orth': 1}
hc = {'sur': 0, 'iso0': 1, 'iso90': 0, 'orth': 0}
fb = {'sur': 1, 'iso0': 0, 'iso90': 1, 'orth': 1}
nMod = 3
nStim = 4
if use_decon: # set in the avg profile cell
    data_list = decon_rois.copy()
else:
    data_list = keep_rois.copy()

# Same design for everyone
Abase = np.zeros((4, 3))
for iStat, stat in enumerate(statDetails['labels']):
    for iM, mod in enumerate([ff, hc, fb]):
        Abase[iStat, iM] = mod[stat]
residual_variance = np.zeros((16, nSubj))
iA = -1
for iSur in range(2):
    for iIso in range(2):
        for iIso90 in range(2):
            for iOrth in range(2):
                A = Abase.copy()
                A[:, 1] = [iSur, iIso, iIso90, iOrth]
                A = np.matrix(A)
                iA += 1
                # cruise through and do the regressions
                answers = np.zeros((nMod, nDepths, nSubj))
                for iSubj in range(nSubj):
                    y = np.matrix(np.zeros((nStim, nDepths)))
                    for iStat, stat in enumerate(statDetails['labels']):
                        for iD, depth in enumerate(depths):
                            y[iStat, iD] = data_list[iSubj][iStat][depth]
                    if iA == 0:
                        input_variance[iSubj] = np.var(y)
                    x = A.I*y
                    answers[:, :, iSubj] = x
                    # reconstruct profiles from answers
                    yest = A*x
                    residual_variance[iA, iSubj] = np.var(yest - y)
                ax = plt.subplot(4, 8, 2*iA + 1)
                ax.imshow(A)
                ax.set_title('%.2f' %np.mean(residual_variance[iA, :]))
                if iA == 0:
                    ax.set_yticks(range(4))
                    ax.set_yticklabels(hc.keys())
                ax = plt.subplot(4, 8, 2*iA + 2)
                for iM, mod in enumerate(['ff', 'hc', 'fb']):
                    ax.plot(np.mean(answers[iM, :, :], axis=1).transpose(),
                             np.linspace(0, 1, nDepths),
                             color=modColors[iM])
                ax.set_xlim([-3, 3])
    

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
    # show localizer data
    minx = np.min(df['x'].values)
    miny = np.min(df['y'].values)
    aw = .8/len(all_data)
    ax = plt.axes([.07 + iR*1.1*aw, .5,  aw, .26])
    for iV, beta in enumerate(df['ctr-sur']):
        if df['ctr-sur F'][iV] > 6.66:
            weight = np.min((np.max((beta, -cmax)), cmax))/cmax
            if weight > 0:
                color = middle + weight*(yellow - middle)
            else:
                color = middle - weight*(purple - middle)
            ax.plot(df['x'][iV], df['y'][iV], '.', color=color)
    ax.set_xlim([minx, minx+18])
    ax.set_ylim([miny, miny+18])
    ax.add_patch(ellipse)
    ax.patch.set_facecolor(fcolor)
    plt.axis('off')
    ax.set_title(label, fontsize=fontsize, color=lcolor)
    
    # showdistance metric
    ax = plt.axes([.07 + iR*1.1*aw, .1,  aw, .26])
    for iV, beta in enumerate(df['scale_xy_dist']):
        if df['scale_xy_dist'][iV] < 2:
            weight = np.min((np.max((beta, -2)), 2))/2
            color = red + weight*(white-red)
            ax.plot(df['x'][iV], df['y'][iV], '.', color=color)
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
fig.savefig('patches.png', facecolor=fig.get_facecolor(), edgecolor='none')

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
    minx = np.min(df['x'].values)
    miny = np.min(df['y'].values)
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
            ax.plot(df['x'][iV], df['y'][iV], '.', color=color)
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
    minx = np.min(df['x'].values)
    maxx = np.max(df['x'].values)
    miny = np.min(df['y'].values)
    maxy = np.max(df['y'].values)
    x_interp = np.linspace(minx,maxx,resamp_factor)
    y_interp = np.linspace(miny,maxy,resamp_factor)
    # beta_interp = sciInterp.interp2d(df['x'].values,df['y'].values,df[cond].values,kind='linear')
    # beta_resamp = beta_interp(x_interp,y_interp)
    # beta_resamp[beta_resamp > cmax] = cmax #truncate
    # beta_resamp[beta_resamp < -cmax] = -cmax
    
    # 2D blurring
    betas = df[cond].values[df['task p-val'] < pthresh]
    betas_x = df['x'].values[df['task p-val'] < pthresh]
    betas_y = df['y'].values[df['task p-val'] < pthresh]
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
depthBoundaries = np.array([[0,7],[7,14],[14,21]])
all_profiles = {depth_label: {label: [] for label in statDetails['labels']} for depth_label in depth_labels}

for iR, label in enumerate(all_data.keys()):
    for iD in range(3):
        df = all_data[label][mask_dict[label]]
        df = df[(df['z'] >=depthBoundaries[iD,0]) & (df['z'] <depthBoundaries[iD,1])]
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
            x = df['scale_xy_dist'].values
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
                
fig1.savefig('surface_profiles.png', facecolor=fig.get_facecolor(), edgecolor='none')

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
depthBoundaries = np.array([[0,7],[7,14],[14,21]])
all_profiles = {depth_label: {label: [] for label in locStatDetails['labels']} for depth_label in depth_labels}

for iR, label in enumerate(all_data.keys()):
    for iD in range(3):
        df = all_data[label][mask_dict[label]]
        df = df[(df['z'] >=depthBoundaries[iD,0]) & (df['z'] <depthBoundaries[iD,1])]
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

fig.savefig('avg_surf_profiles.png', facecolor=fig.get_facecolor(), edgecolor='none')

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
            df = df[(df['z'] >=1) & (df['z'] <3)]
        elif iD == 1:
            df = df[(df['z'] >=4) & (df['z'] <6)]
        elif iD == 2:
            df = df[(df['z'] >=6) & (df['z'] <=9)]
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
            x = df['scale_xy_dist'].values
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
        
#%% Centroid plots
# Let's take a look at raw voxel betas across depth by condition

roiRad = 2
nDepths = 20
maxDepth = 20 #leave off pial surface
Nsubj = len(all_data.keys())
for iStat in range(len(statDetails['labels'])):
    plt.figure(statDetails['labels'][iStat])
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = roi_dict[label] & mask_dict[label]
        roi = df[roi_idx]
        roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
        roi = roi[roi['scale_xy_dist'] >= 0] #don't allow negative radii    
        dataDict = makeProfile1D(roi['z'].values,
                                     nDepths,
                                     roi[statDetails['labels']].values)
        normDepths = (roi['z']-1)/maxDepth
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
diffDetails['statIDs'] = {'odss': [np.where(np.array(statDetails['labels']) == 'orth')[0][0],np.where(np.array(statDetails['labels']) == 'iso90')[0][0]],
                          'fgm': [np.where(np.array(statDetails['labels']) == 'iso90')[0][0],np.where(np.array(statDetails['labels']) == 'iso0')[0][0]],
                          'dsi': [np.where(np.array(statDetails['labels']) == 'orth')[0][0],np.where(np.array(statDetails['labels']) == 'iso0')[0][0]]}
                        #get stat IDs needed tocompute each difference profile
for iDiff in range(len(diffDetails['labels'])):
    plt.figure(diffDetails['labels'][iDiff])
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        roi_idx = roi_dict[label] & mask_dict[label]
        roi = df[roi_idx]
        roi = df[df['scale_xy_dist'] < roiRad] # only very center, to be sure!
        roi = roi[roi['scale_xy_dist'] >= 0] #don't allow negative radii    
        dataDict = makeProfile1D(roi['z'].values,
                                     nDepths,
                                     roi[statDetails['labels']].values)
        normDepths = (roi['z']-1)/maxDepth
        avg_normDepths = (dataDict['profile']['depth'] - np.nanmin(dataDict['profile']['depth']))/(np.nanmax(dataDict['profile']['depth']) - np.nanmin(dataDict['profile']['depth']))
        
        #compute differences
        statIDs = diffDetails['statIDs'][diffDetails['labels'][iDiff]]
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

#%% Compare depth profiles at different radii

statDetails = {'labels': ['sur', 'iso0', 'iso90', 'orth'],
                'colors': [[.7, .7, .7], 'red', 'darkviolet', 'orange']}
profile_method = 'bin' # bin or smooth
nDepths = 10
#pick out ROIs where we're sure of localization
roi_dict = {'center': {}, 'border': {}, 'surround': {}}
centerRad = 1.5 #center of ROI
borderRad = [1.5,2.5] #range for border
surRad = 2.5 #outside of this range will be considered the surround
for key in all_data.keys():
    df = all_data[key]
    roi_dict['center'][key] = df['scale_xy_dist'] < centerRad
    roi_dict['border'][key] = (df['scale_xy_dist'] > borderRad[0]) & (df['scale_xy_dist'] < borderRad[1])
    roi_dict['surround'][key] = df['scale_xy_dist'] > surRad
keep_depths = {'center': [], 'border': [], 'surround': []}
keep_rois = {'center': [], 'border': [], 'surround': []}
keep_std = {'center': [], 'border': [], 'surround': []}
fgm = {'center': [], 'border': [], 'surround': []}
odss = {'center': [], 'border': [], 'surround': []}
dSI = {'center': [], 'border': [], 'surround': []}
[hprofiles_ctr, hdiff_ctr, keep_depths['center'], keep_rois['center'], keep_std['center'], fgm['center'], odss['center'], dSI['center']] = plot_depth_profiles(all_data, roi_dict['center'], statDetails, profile_method, nDepths, ['profiles_ctr','diff_ctr'],useSI,mask_dict)
[hprofiles_bor, hdiff_bor, keep_depths['border'], keep_rois['border'], keep_std['border'], fgm['border'], odss['border'], dSI['border']] = plot_depth_profiles(all_data, roi_dict['border'], statDetails, profile_method, nDepths, ['profiles_bor','diff_bor'],useSI,mask_dict)
[hprofiles_sur, hdiff_sur, keep_depths['surround'], keep_rois['surround'], keep_std['surround'], fgm['surround'], odss['surround'], dSI['surround']] = plot_depth_profiles(all_data, roi_dict['surround'], statDetails, profile_method, nDepths, ['profiles_sur','diff_sur'],useSI,mask_dict)

#%% Deconvolve depth

decon_roi = {'center': {}, 'border': {}, 'surround': {}}
p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
Nbins = 10 #number of bins to use in this analysis
normalize_psf = False #True if you want to normalize the psf by the deepest layer  
for key in decon_roi.keys():
    decon_roi[key] = depth_deconv(keep_rois[key],p2t_model,Nbins_model,Nbins,normalize_psf)
    
#%% Plot average profiles

use_decon = True
prop_err = True
fig_ctr = plot_avg_depth_profile(decon_roi['center'],keep_depths['center'],keep_std['center'],use_decon,prop_err)
fig_bor = plot_avg_depth_profile(decon_roi['border'],keep_depths['border'],keep_std['border'],use_decon,prop_err)
fig_sur = plot_avg_depth_profile(decon_roi['surround'],keep_depths['surround'],keep_std['surround'],use_decon,prop_err)

if use_decon:
    fig_ctr.savefig('task_profiles_decon_ctr.png', facecolor=fig.get_facecolor(), edgecolor='none')
    fig_bor.savefig('task_profiles_decon_border.png', facecolor=fig.get_facecolor(), edgecolor='none')
    fig_sur.savefig('task_profiles_decon_surround.png', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    fig_ctr.savefig('task_profiles_ctr.png', facecolor=fig.get_facecolor(), edgecolor='none')
    fig_bor.savefig('task_profiles_border.png', facecolor=fig.get_facecolor(), edgecolor='none')
    fig_sur.savefig('task_profiles_sur.png', facecolor=fig.get_facecolor(), edgecolor='none')