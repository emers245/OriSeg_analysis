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
savefigs =  False #True #if true save all figures
figDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code/figs/'
fig_format = 'svg'
statCorrType = 'fdr_bh'

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset

mainDir = '.'
datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual', 'pnr???_??_???_??.csv'))
#datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_filled', 'pnr???_??_???_??.csv'))
#or exclude
exclude_initial = ['pnr143_V1_tgt_rh','pnr143_V1_tgt_lh','pnr161_V1_tgt_lh','pnr161_V1_tgt_rh','pnr352_V1_tgt_lh','pnr352_V1_tgt_rh','pnr579_V1_tgt_lh','pnr579_V1_tgt_rh','pnr668_V1_tgt_rh', 'pnr668_V1_tgt_lh']
#exclude_initial = ['pnr352_V1_tgt_lh_rad10']
for e_i, excl in enumerate(exclude_initial):
    print('Removing roi_data_manualSeg/target_roi_manual',excl+'.csv')
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
roiRad = 1.
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
nDepths = 5 #3
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
    
#%% Make Rings Graphic
from matplotlib.patches import Ellipse

centerRad = 1.0 #1.7 #center of ROI
borderRad = [1.0,3.0] #[1.7,2.3] #range for border
surRad = [3.5] #[3, 3.5] #outside of this range will be considered the surround
ring_rads = {'ctr':centerRad,'border':borderRad,'sur':surRad}
rings = ['ctr','border','sur']

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
        pcm = ax.scatter(df['x'],df['y'],c=df['ctr-sur'],s=10,cmap=cmap,vmin=-1,vmax=1)
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
roiRad = 1
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
    dmap = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0,100], fsize, fname = 'dmap', pad=0.0)
        
    #plot thresholded map
    dmap_thresh = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, NROIs, [0, 100], fsize, fname='dmap_thresh', mask=mnv_mask, pad=0.0)
        
    # Plot voxel loss at each depth after masking
    fdepth_hist = plot_depth_voxel_loss(z, mnv_mask, nDepths, NROIs, key, k_i, fsize)
    
    #report number of voxels after threshold
    print("%d/%d Voxels Survive for %s" %(np.sum(mnv_mask),np.size(mnv),key))
    
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
pthresh_fullmodel = 0.05
if use_fullmodel_mask:
    for k_i, key in enumerate(all_data.keys()):
        df = all_data[key]
        pvals = df['task p-val']
        pval_mask = pvals < pthresh_fullmodel
        print("%d/%d voxels survive full model p-val mask" %(np.sum(pval_mask),np.size(pval_mask)))
        mask_dict[key] = mask_dict[key] & pval_mask   
        all_data[key]['sig'] = pval_mask

#%% Compare depth profiles at different radii
useSI=False

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
#pick out ROIs where we're sure of localization
masks = {'ctr':{},'border':{},'sur':{}}
for key in all_data.keys():
    df = all_data[key]
    ctr_mask = df['scale_xy_dist'] < centerRad
    masks['ctr'][key] = ctr_mask*df['sig']*df['no_vein']
    bor_mask = ((df['scale_xy_dist'] >= borderRad[0]) & (df['scale_xy_dist'] <= borderRad[1]))
    masks['border'][key] = bor_mask*df['sig']*df['no_vein']
    if len(surRad) == 1:
        sur_mask = df['scale_xy_dist'] > surRad[0]
    elif len(surRad) > 1:
        sur_mask = ((df['scale_xy_dist'] >= surRad[0]) & (df['scale_xy_dist'] <= surRad[1]))
    masks['sur'][key] = sur_mask*df['sig']*df['no_vein']
    
depthProfiles = {}
diffProfiles = {}
for r_i, ring in enumerate(rings):
    depthProfiles[ring] = compute_all_depth_profiles(all_data,statDetails,profile_method,nDepths,masks[ring],depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')
    diffProfiles[ring] = compute_diff_profiles(all_data,statDetails,diffDetails['statIDs'],profile_method,nDepths,useSI,masks[ring],depthParam='d',radialParam='scale_xy_dist',spec_Drange='MinMax')

#%% Depth Histograms

for ring in depthProfiles.keys():
    cond = list(depthProfiles[ring].keys())[0]
    fdhist = plt.figure(figsize=(15,5))
    for iR, label in enumerate(all_data.keys()):
        plt.subplot(2,int(np.ceil(len(datasets)/2)),iR+1)
        plt.bar(np.linspace(0,1,nDepths),height=depthProfiles[ring][cond]['N'][iR],width=(1+1/nDepths)/nDepths)
        plt.title(label,fontsize=10)
        plt.xlabel("Normalize Depth WM -> GM")
        plt.ylabel("Num. Voxels")
        plt.legend(['N='+str(np.sum(depthProfiles[ring][cond]['N'][iR])),], fontsize = 6)
    plt.suptitle(ring)
    fdhist.tight_layout(pad=0.0)

#%% Centroid plots
# Let's take a look at raw voxel betas across depth by condition

roiRad = 2
Nsubj = len(all_data.keys())

for r_i, ring in enumerate(rings):
    
    #plot centroids for each condition and ROI
    plot_centroids(all_data, masks[ring], statDetails, roiRad, nDepths)
            
    #calculate difference profiles
    plot_centroids_diff(all_data, masks[ring], statDetails, diffDetails, roiRad, nDepths)

#%% Deconvolution

#reformat data to fit decon_rois specs/
keep_rois = {}
keep_diffs = {}
for r_i,ring in enumerate(rings):
    
    keep_rois[ring] = np.zeros((NROIs,len(statDetails['labels']),nDepths))
    for iR, roiID in enumerate(all_data.keys()):
        for iStat, stat in enumerate(statDetails['labels']):
            keep_rois[ring][iR,iStat,:] = depthProfiles[ring][stat]['avg'][iR]
            
    keep_diffs[ring] = np.zeros((NROIs,len(diffDetails['statIDs'].keys()),nDepths))
    for iR, roiID in enumerate(all_data.keys()):
        for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
            keep_diffs[ring][iR,iDiff,:] = diffProfiles[ring][diff]['avg'][iR]
    
    #define point spread function
    p2t_model = 6.2 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
    Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
    Nbins = nDepths #number of bins to use in this analysis
    
    normalize_psf = False #True if you want to normalize the psf by the deepest layer  
    
    decon_rois = depth_deconv(keep_rois[ring],p2t_model,Nbins_model,Nbins,normalize_psf)
    decon_diffs = depth_deconv(keep_diffs[ring],p2t_model,Nbins_model,Nbins,normalize_psf)
    
    #now put back in dictionary
    for iStat, stat in enumerate(statDetails['labels']):
        depthProfiles[ring][stat]['avg_decon'] = np.squeeze(np.array(decon_rois)[:,iStat,:])
        
    for iDiff, diff in enumerate(diffDetails['statIDs'].keys()):
        diffProfiles[ring][diff]['avg_decon'] = np.squeeze(np.array(decon_diffs)[:,iDiff,:])
    
#%% Plot average profiles

use_decon = True
prop_err = False

taskStats = statDetails['labels'][0:4]
taskColors = statDetails['colors'][0:4]
taskDiffs = list(diffDetails['statIDs'].keys())[0:4]

# Plot ctr average profiles
[ctrTaskProfiles, ctrTaskDiffs] = compute_avg_depth_profile(depthProfiles['ctr'],statDetails,diffDetails['statIDs'],taskStats,taskDiffs,use_decon,prop_err,useSI)  

locStats = statDetails['labels'][4:]
locColors = statDetails['colors'][4:]
locDiffs = list(diffDetails['statIDs'].keys())[4:]

# average loc profiles
[ctrLocProfiles, ctrLocDiffs] = compute_avg_depth_profile(depthProfiles['ctr'],statDetails,diffDetails['statIDs'],locStats,locDiffs,use_decon,prop_err,useSI)  
[borLocProfiles, borLocDiffs] = compute_avg_depth_profile(depthProfiles['border'],statDetails,diffDetails['statIDs'],locStats,locDiffs,use_decon,prop_err,useSI)  
[surLocProfiles, surLocDiffs] = compute_avg_depth_profile(depthProfiles['sur'],statDetails,diffDetails['statIDs'],locStats,locDiffs,use_decon,prop_err,useSI)  

fig_ctr = plt.figure(figsize=(6, 4))
fig_ctr.set_size_inches((6,4))
fig_ctr.patch.set_facecolor(fcolor)
    
fig_ctr.clf()
fsize = 14
    
p1 = fig_ctr.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,ctrTaskProfiles,taskStats,taskColors,ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_ctr.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.3,1.5]
plot_avg_diff_profile(p2,ctrTaskDiffs,taskDiffs,diffDetails['colors'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI)

if savefigs:
    if use_decon:
        fig_ctr.savefig(os.path.join(figDir,'task_profiles_decon_center.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_ctr.savefig(os.path.join(figDir,'task_profiles_ctr.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# Plot border average profiles
[borTaskProfiles, borTaskDiffs] = compute_avg_depth_profile(depthProfiles['border'],statDetails,diffDetails['statIDs'],taskStats,taskDiffs,use_decon,prop_err,useSI)  

fig_bor = plt.figure(figsize=(6, 4))
fig_bor.set_size_inches((6,4))
fig_bor.patch.set_facecolor(fcolor)
    
fig_bor.clf()
fsize = 14
    
p1 = fig_bor.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,borTaskProfiles,taskStats,taskColors,ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_bor.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.3,1.5]
plot_avg_diff_profile(p2,borTaskDiffs,taskDiffs,diffDetails['colors'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI)

if savefigs:
    if use_decon:
        fig_bor.savefig(os.path.join(figDir,'task_profiles_decon_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_bor.savefig(os.path.join(figDir,'task_profiles_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
# Plot sur average profiles
[surTaskProfiles, surTaskDiffs] = compute_avg_depth_profile(depthProfiles['sur'],statDetails,diffDetails['statIDs'],taskStats,taskDiffs,use_decon,prop_err,useSI)  

fig_sur = plt.figure(figsize=(6, 4))
fig_sur.set_size_inches((6,4))
fig_sur.patch.set_facecolor(fcolor)
    
fig_sur.clf()
fsize = 14
    
p1 = fig_sur.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,surTaskProfiles,taskStats,taskColors,ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_sur.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.3,1.5]
plot_avg_diff_profile(p2,surTaskDiffs,taskDiffs,diffDetails['colors'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI)

if savefigs:
    if use_decon:
        fig_sur.savefig(os.path.join(figDir,'task_profiles_decon_surround.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_sur.savefig(os.path.join(figDir,'task_profiles_sur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
#%% Run Stats
# In this section I'll correct the p-values for all of the depth bins in each
# context modulation contion in each ROI (ring). Plus, I will test all of the bins against all of 
# the other bins. I'll need to do an ANOVA for this part. In the end, I will 
# need to correct for multiple comparisons taking into account all the 
# comparisons made. This will be 
#     N_depths x N_conditions x N_rings 
#        + (N_conditions choose 2) x N_rings x N_depths 
#        + (N_rings choose 2) x N_conditions x N_depths comparisons

# I have already collected the p-values when running compute_depth_profiles.
# This tested each depth against a null hypothesis that the distribution has a
# mean of zero.

all_pvals = np.array([]) #initialize a 1D array that will contain all pvals
pvals_lookup = {'ctr':{},'border':{},'sur':{}} #initialize a dictionary that will keep track of which p-vals correspond to which comparison

# 1-sample t-tests
cond_iter = 0
for c_i, c in enumerate(ctrTaskDiffs.keys()):
    all_pvals = np.append(all_pvals,ctrTaskDiffs[c]['p-vals'].pvalue)
    pvals_lookup['ctr'][c] = [cond_iter*nDepths,cond_iter*nDepths+nDepths]
    cond_iter += 1
for c_i, c in enumerate(borTaskDiffs.keys()):
    all_pvals = np.append(all_pvals,borTaskDiffs[c]['p-vals'].pvalue)
    pvals_lookup['border'][c] = [cond_iter*nDepths,cond_iter*nDepths+nDepths]
    cond_iter += 1
for c_i, c in enumerate(surTaskDiffs.keys()):
    all_pvals = np.append(all_pvals,surTaskDiffs[c]['p-vals'].pvalue)
    pvals_lookup['sur'][c] = [cond_iter*nDepths,cond_iter*nDepths+nDepths]
    cond_iter += 1
for c_i, c in enumerate(ctrLocDiffs.keys()):
    all_pvals = np.append(all_pvals,ctrLocDiffs[c]['p-vals'].pvalue)
    pvals_lookup['ctr'][c] = [cond_iter*nDepths,cond_iter*nDepths+nDepths]
    cond_iter += 1
for c_i, c in enumerate(borLocDiffs.keys()):
    all_pvals = np.append(all_pvals,borLocDiffs[c]['p-vals'].pvalue)
    pvals_lookup['border'][c] = [cond_iter*nDepths,cond_iter*nDepths+nDepths]
    cond_iter += 1
for c_i, c in enumerate(surLocDiffs.keys()):
    all_pvals = np.append(all_pvals,surLocDiffs[c]['p-vals'].pvalue)
    pvals_lookup['sur'][c] = [cond_iter*nDepths,cond_iter*nDepths+nDepths]
    cond_iter += 1
    
# 3-way ANOVA
diffProfiles_list = {}
for ring in diffProfiles.keys(): 
    diffProfiles_list[ring] = {}
    for cond in diffProfiles[ring].keys():
        diffProfiles_list[ring][cond] = diffProfiles[ring][cond]['avg_decon']
    
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def run_three_way_anova(data):
    """
    This function runs a 3-way ANOVA on a dataset structured as nested dictionaries.
    
    Parameters:
    - data: A dictionary with the structure described by the user.
    
    Returns:
    - A pandas DataFrame containing the ANOVA table with the main effects and interactions.
    """
    # Transform the nested dictionary into a flat DataFrame
    condition_list = []
    depth_bin_list = []
    ring_list = []
    value_list = []

    # Loop over each ring, condition, and its samples
    for ring, conditions in data.items():
        for condition, samples in conditions.items():
            for sample in samples:
                for depth_bin, value in enumerate(sample):
                    # Append data to lists
                    condition_list.append(condition)
                    ring_list.append(ring)
                    depth_bin_list.append(depth_bin)
                    value_list.append(value)

    # Create a dataframe with the lists
    df = pd.DataFrame({
        'Condition': condition_list,
        'Ring': ring_list,
        'Depth_Bin': depth_bin_list,
        'Value': value_list
    })
    
    # Define the model formula for 3-way ANOVA
    model_formula = 'Value ~ C(Condition) + C(Ring) + Depth_Bin + \
                     C(Condition):C(Ring) + C(Condition):Depth_Bin + \
                     C(Ring):Depth_Bin + C(Condition):C(Ring):Depth_Bin'

    # Fit the model
    model = ols(model_formula, data=df).fit()

    # Perform ANOVA and get the table
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    return anova_table

anova_results = run_three_way_anova(diffProfiles_list)
print("ANOVA Results: ")
print(anova_results)

# To perform a two-sample t-test between each of the depth bins for each condition, we will:
# 1. Loop through each condition.
# 2. For each condition, perform a t-test between each unique pair of depth bins.
# 3. Collect and return the p-values for each test.

# Dictionary to hold the p-values
p_values = {}
                
from itertools import combinations

# Check for significance across conditions within rings
pval_i = len(all_pvals)
for ring in diffProfiles_list.keys():
    p_values[ring] = {}
    pvals_lookup[ring+'_2samp'] = {}
    conditions = diffProfiles_list[ring].keys()
    combinations_list = list(combinations(conditions, 2))
    for (condition1, condition2) in combinations_list:
        p_values[ring][(condition1, condition2)] = {}
        # Get the samples for the condition
        sample1 = diffProfiles_list[ring][condition1]
        sample2 = diffProfiles_list[ring][condition2]
        # Perform a t-test between each unique pair of depth bins
        pvals_lookup[ring+'_2samp'][(condition1,condition2)] = {}
        # Perform the t-test
        t_stat, p_val = stats.ttest_ind(sample1, sample2, axis=0, equal_var=False)
        # Store the p-value
        p_values[ring][(condition1, condition2)] = p_val
        # Add to big list
        all_pvals = np.append(all_pvals,p_val)
        pvals_lookup[ring+'_2samp'][(condition1,condition2)] = [pval_i,pval_i+nDepths]
        pval_i += nDepths
            
# Check for significance across rings within conditions
pval_i = len(all_pvals)
for condition in diffProfiles_list[ring].keys():
    p_values[condition] = {}
    pvals_lookup[condition+'_2samp'] = {}
    rings = diffProfiles_list.keys()
    combinations_list = list(combinations(rings, 2))
    for (ring1, ring2) in combinations_list:
        p_values[condition][(ring1,ring2)] = {}
        # Get the samples for the condition
        sample1 = diffProfiles_list[ring1][condition]
        sample2 = diffProfiles_list[ring2][condition]
        # Perform a t-test between each unique pair of depth bins
        pvals_lookup[condition+'_2samp'][(ring1,ring2)] = {}
        # Perform the t-test
        t_stat, p_val = stats.ttest_ind(sample1, sample2, axis=0, equal_var=False)
        # Store the p-value
        p_values[condition][(ring1, ring2)] = p_val
        # Add to big list
        all_pvals = np.append(all_pvals,p_val)
        pvals_lookup[condition+'_2samp'][(ring1,ring2)] = [pval_i,pval_i+nDepths]
        pval_i += nDepths
        
# Check for differences within rings within conditions across depths
pval_i = len(all_pvals)
for ring in diffProfiles_list.keys():
    pvals_lookup[ring+'_2sampDepth'] = {}
    for condition in diffProfiles_list[ring].keys():
        p_values[condition] = {}
        # Get the samples for the condition
        samples = diffProfiles_list[ring][condition]
        # Perform a t-test between each unique pair of depth bins
        pvals_lookup[ring+'_2sampDepth'][condition+'_2sampDepth'] = {}
        for i in range(len(samples[0])):
            for j in range(i+1, len(samples[0])):
                depth_bin_i_samples = [sample[i] for sample in samples]
                depth_bin_j_samples = [sample[j] for sample in samples]
                # Perform the t-test
                t_stat, p_val = stats.ttest_ind(depth_bin_i_samples, depth_bin_j_samples, equal_var=False)
                # Store the p-value
                p_values[condition][(i, j)] = p_val
                # Add to big list
                all_pvals = np.append(all_pvals,p_val)
                pvals_lookup[ring+'_2sampDepth'][condition+'_2sampDepth'][(i,j)] = pval_i
                pval_i += 1
            
# Now do a big multiple-comparisons correction
all_pvals_corrected = multipletests(all_pvals,method=statCorrType)[1]
all_pvals_dict = {}

# Repackage all of the corrected p-values into a new dictionary
for key in pvals_lookup:
    if isinstance(pvals_lookup[key],dict):
        all_pvals_dict[key] = {}
        for combo in pvals_lookup[key]:
            if isinstance(pvals_lookup[key][combo],dict):
                all_pvals_dict[key][combo] = {}
                for combo2 in pvals_lookup[key][combo]:
                    all_pvals_dict[key][combo][combo2] = all_pvals_corrected[pvals_lookup[key][combo][combo2]]
            else:
                all_pvals_dict[key][combo] = all_pvals_corrected[pvals_lookup[key][combo][0]:pvals_lookup[key][combo][1]]
    else:
        all_pvals_dict[key] = all_pvals_corrected[pvals_lookup[key][0]:pvals_lookup[key][1]]

#%% iso - sur

pthresh = 0.05 #pval thresh
pvals_task_ctr = {cond:all_pvals_dict['ctr'][cond] for cond in ['odss','fgm','dsi','iso-sur','ctr-sur']}
pvals_task_border = {cond:all_pvals_dict['border'][cond] for cond in ['odss','fgm','dsi','iso-sur','ctr-sur']}
pvals_task_sur = {cond:all_pvals_dict['sur'][cond] for cond in ['odss','fgm','dsi','iso-sur','ctr-sur']}

# Plot ctr average profiles
fig_ctr = plt.figure(figsize=(6, 4))
fig_ctr.set_size_inches((6,4))
fig_ctr.patch.set_facecolor(fcolor)
    
fig_ctr.clf()
fsize = 14
    
p1 = fig_ctr.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,ctrTaskProfiles,['iso0','sur'],['red',[0.7,0.7,0.7]],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_ctr.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,ctrTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_ctr)
plt.title('ctr')

if savefigs:
    if use_decon:
        fig_ctr.savefig(os.path.join(figDir,'iso_sur_profiles_decon_center.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_ctr.savefig(os.path.join(figDir,'iso_sur_profiles_ctr.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# Plot border average profiles
fig_bor = plt.figure(figsize=(6, 4))
fig_bor.set_size_inches((6,4))
fig_bor.patch.set_facecolor(fcolor)
    
fig_bor.clf()
fsize = 14
    
p1 = fig_bor.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,borTaskProfiles,['iso0','sur'],['red',[0.7,0.7,0.7]],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_bor.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,borTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_border)
plt.title('border')

if savefigs:
    if use_decon:
        fig_bor.savefig(os.path.join(figDir,'iso_sur_profiles_decon_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_bor.savefig(os.path.join(figDir,'iso_sur_profiles_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
# Plot sur average profiles
fig_sur = plt.figure(figsize=(6, 4))
fig_sur.set_size_inches((6,4))
fig_sur.patch.set_facecolor(fcolor)
    
fig_sur.clf()
fsize = 14
    
p1 = fig_sur.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,surTaskProfiles,['iso0','sur'],['red',[0.7,0.7,0.7]],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_sur.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,surTaskDiffs,['iso-sur'],['black'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_sur)
plt.title('sur')

if savefigs:
    if use_decon:
        fig_sur.savefig(os.path.join(figDir,'iso_sur_profiles_decon_surround.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_sur.savefig(os.path.join(figDir,'iso_sur_profiles_sur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
#%% iso - sur Compare all diff profiles
# ctr vs border
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_bor_isosur = {'ctr_iso': ctrTaskProfiles['iso0'], 'ctr_sur': ctrTaskProfiles['sur'], 'bor_iso': borTaskProfiles['iso0'], 'bor_sur': borTaskProfiles['sur']}
plot_avg_depth_profile(p1,ctr_bor_isosur,['ctr_iso', 'ctr_sur','bor_iso', 'bor_sur'],['red',[0.7,0.7,0.7],'red',[0.7,0.7,0.7]],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,2]

ctr_border_avg = {'ctr': ctrTaskDiffs['iso-sur'], 'border': borTaskDiffs['iso-sur']}
ctr_border_profiles = {'ctr': diffProfiles['ctr']['iso-sur'], 'border': diffProfiles['border']['iso-sur']}
plot_profile_comparisons(p2,ctr_border_avg,ctr_border_profiles,['ctr','border'],['black','black'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['iso-sur_2samp'][('ctr', 'border')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'iso_sur_CtrVBor_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'iso_sur_CtrVBor.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# ctr vs sur
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_sur_isosur = {'ctr_iso': ctrTaskProfiles['iso0'], 'ctr_sur': ctrTaskProfiles['sur'], 'sur_iso': surTaskProfiles['iso0'], 'sur_sur': surTaskProfiles['sur']}
plot_avg_depth_profile(p1,ctr_sur_isosur,['ctr_iso', 'ctr_sur','sur_iso', 'sur_sur'],['red',[0.7,0.7,0.7],'red',[0.7,0.7,0.7]],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,2]

ctr_sur_avg = {'ctr': ctrTaskDiffs['iso-sur'], 'sur': surTaskDiffs['iso-sur']}
ctr_sur_profiles = {'ctr': diffProfiles['ctr']['iso-sur'], 'sur': diffProfiles['sur']['iso-sur']}
plot_profile_comparisons(p2,ctr_sur_avg,ctr_sur_profiles,['ctr','sur'],['black','black'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['iso-sur_2samp'][('ctr', 'sur')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'iso_sur_CtrVSur_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'iso_sur_CtrVSur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    
        
#%% dsi

pthresh = 0.05 #pval thresh

# Plot ctr average profiles
fig_ctr = plt.figure(figsize=(6, 4))
fig_ctr.set_size_inches((6,4))
fig_ctr.patch.set_facecolor(fcolor)
    
fig_ctr.clf()
fsize = 14
    
p1 = fig_ctr.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,ctrTaskProfiles,['orth','iso0'],['orange','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_ctr.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,ctrTaskDiffs,['dsi'],['cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_ctr)

if savefigs:
    if use_decon:
        fig_ctr.savefig(os.path.join(figDir,'dsi_profiles_decon_center.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_ctr.savefig(os.path.join(figDir,'dsi_profiles_ctr.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# Plot border average profiles
fig_bor = plt.figure(figsize=(6, 4))
fig_bor.set_size_inches((6,4))
fig_bor.patch.set_facecolor(fcolor)
    
fig_bor.clf()
fsize = 14
    
p1 = fig_bor.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,borTaskProfiles,['orth','iso0'],['orange','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_bor.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,borTaskDiffs,['dsi'],['cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_border)

if savefigs:
    if use_decon:
        fig_bor.savefig(os.path.join(figDir,'dsi_profiles_decon_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_bor.savefig(os.path.join(figDir,'dsi_profiles_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
# Plot sur average profiles
fig_sur = plt.figure(figsize=(6, 4))
fig_sur.set_size_inches((6,4))
fig_sur.patch.set_facecolor(fcolor)
    
fig_sur.clf()
fsize = 14
    
p1 = fig_sur.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,surTaskProfiles,['orth','iso0'],['orange','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_sur.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,surTaskDiffs,['dsi'],['cyan'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_sur)

if savefigs:
    if use_decon:
        fig_sur.savefig(os.path.join(figDir,'dsi_profiles_decon_surround.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_sur.savefig(os.path.join(figDir,'dsi_profiles_sur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

#%% dsi Compare all diff profiles
# ctr vs border
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_bor_dsi = {'ctr_iso': ctrTaskProfiles['iso0'], 'ctr_orth': ctrTaskProfiles['orth'], 'bor_iso': borTaskProfiles['iso0'], 'bor_orth': borTaskProfiles['orth']}
plot_avg_depth_profile(p1,ctr_bor_dsi,['ctr_iso', 'ctr_orth','bor_iso', 'bor_orth'],['red','orange','red','orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,2]

ctr_border_avg = {'ctr': ctrTaskDiffs['dsi'], 'border': borTaskDiffs['dsi']}
ctr_border_profiles = {'ctr': diffProfiles['ctr']['dsi'], 'border': diffProfiles['border']['dsi']}
plot_profile_comparisons(p2,ctr_border_avg,ctr_border_profiles,['ctr','border'],['cyan','cyan'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['dsi_2samp'][('ctr', 'border')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'dsi_CtrVBor_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'dsi_CtrVBor.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# ctr vs sur
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_sur_dsi = {'ctr_iso': ctrTaskProfiles['iso0'], 'ctr_orth': ctrTaskProfiles['orth'], 'sur_iso': surTaskProfiles['iso0'], 'sur_orth': surTaskProfiles['orth']}
plot_avg_depth_profile(p1,ctr_sur_dsi,['ctr_iso', 'ctr_orth','sur_iso', 'sur_orth'],['red','orange','red','orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,2]

ctr_sur_avg = {'ctr': ctrTaskDiffs['dsi'], 'sur': surTaskDiffs['dsi']}
ctr_sur_profiles = {'ctr': diffProfiles['ctr']['dsi'], 'sur': diffProfiles['sur']['dsi']}
plot_profile_comparisons(p2,ctr_sur_avg,ctr_sur_profiles,['ctr','sur'],['cyan','cyan'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['dsi_2samp'][('ctr', 'sur')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'dsi_CtrVSur_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'dsi_CtrVSur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
         
#%% odss

pthresh = 0.05 #pval thresh

# Plot ctr average profiles
fig_ctr = plt.figure(figsize=(6, 4))
fig_ctr.set_size_inches((6,4))
fig_ctr.patch.set_facecolor(fcolor)
    
fig_ctr.clf()
fsize = 14
    
p1 = fig_ctr.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,ctrTaskProfiles,['orth','iso90'],['orange','darkviolet'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_ctr.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,ctrTaskDiffs,['odss'],['green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_ctr)

if savefigs:
    if use_decon:
        fig_ctr.savefig(os.path.join(figDir,'odss_profiles_decon_center.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_ctr.savefig(os.path.join(figDir,'odss_profiles_ctr.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# Plot border average profiles
fig_bor = plt.figure(figsize=(6, 4))
fig_bor.set_size_inches((6,4))
fig_bor.patch.set_facecolor(fcolor)
    
fig_bor.clf()
fsize = 14
    
p1 = fig_bor.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,borTaskProfiles,['orth','iso90'],['orange','darkviolet'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_bor.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,borTaskDiffs,['odss'],['green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_border)

if savefigs:
    if use_decon:
        fig_bor.savefig(os.path.join(figDir,'odss_profiles_decon_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_bor.savefig(os.path.join(figDir,'odss_profiles_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
# Plot sur average profiles
fig_sur = plt.figure(figsize=(6, 4))
fig_sur.set_size_inches((6,4))
fig_sur.patch.set_facecolor(fcolor)
    
fig_sur.clf()
fsize = 14
    
p1 = fig_sur.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,surTaskProfiles,['orth','iso90'],['orange','darkviolet'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_sur.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,surTaskDiffs,['odss'],['green'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_sur)

if savefigs:
    if use_decon:
        fig_sur.savefig(os.path.join(figDir,'odss_profiles_decon_surround.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_sur.savefig(os.path.join(figDir,'odss_profiles_sur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

#%% odss Compare all diff profiles
# ctr vs border
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_bor_odss = {'ctr_iso90': ctrTaskProfiles['iso90'], 'ctr_orth': ctrTaskProfiles['orth'], 'bor_iso90': borTaskProfiles['iso90'], 'bor_orth': borTaskProfiles['orth']}
plot_avg_depth_profile(p1,ctr_bor_odss,['ctr_iso90', 'ctr_orth','bor_iso90', 'bor_orth'],['darkviolet','orange','darkviolet','orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,2]

ctr_border_avg = {'ctr': ctrTaskDiffs['odss'], 'border': borTaskDiffs['odss']}
ctr_border_profiles = {'ctr': diffProfiles['ctr']['odss'], 'border': diffProfiles['border']['odss']}
plot_profile_comparisons(p2,ctr_border_avg,ctr_border_profiles,['ctr','border'],['green','green'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['odss_2samp'][('ctr', 'border')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'odss_CtrVBor_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'odss_CtrVBor.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# ctr vs sur
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_sur_odss = {'ctr_iso90': ctrTaskProfiles['iso90'], 'ctr_orth': ctrTaskProfiles['orth'], 'sur_iso90': surTaskProfiles['iso90'], 'sur_orth': surTaskProfiles['orth']}
plot_avg_depth_profile(p1,ctr_sur_odss,['ctr_iso90', 'ctr_orth','sur_iso90', 'sur_orth'],['darkviolet','orange','darkviolet','orange'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,2]

ctr_sur_avg = {'ctr': ctrTaskDiffs['odss'], 'sur': surTaskDiffs['odss']}
ctr_sur_profiles = {'ctr': diffProfiles['ctr']['odss'], 'sur': diffProfiles['sur']['odss']}
plot_profile_comparisons(p2,ctr_sur_avg,ctr_sur_profiles,['ctr','sur'],['green','green'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['odss_2samp'][('ctr', 'sur')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'odss_CtrVSur_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'odss_CtrVSur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

        
#%% fgm

pthresh = 0.05 #pval thresh

# Plot ctr average profiles
fig_ctr = plt.figure(figsize=(6, 4))
fig_ctr.set_size_inches((6,4))
fig_ctr.patch.set_facecolor(fcolor)
    
fig_ctr.clf()
fsize = 14
    
p1 = fig_ctr.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,ctrTaskProfiles,['iso90','iso0'],['darkviolet','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_ctr.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,ctrTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_ctr)

if savefigs:
    if use_decon:
        fig_ctr.savefig(os.path.join(figDir,'fgm_profiles_decon_center.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_ctr.savefig(os.path.join(figDir,'fgm_profiles_ctr.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# Plot border average profiles
fig_bor = plt.figure(figsize=(6, 4))
fig_bor.set_size_inches((6,4))
fig_bor.patch.set_facecolor(fcolor)
    
fig_bor.clf()
fsize = 14
    
p1 = fig_bor.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,borTaskProfiles,['iso90','iso0'],['darkviolet','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_bor.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,borTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_border)

if savefigs:
    if use_decon:
        fig_bor.savefig(os.path.join(figDir,'fgm_profiles_decon_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_bor.savefig(os.path.join(figDir,'fgm_profiles_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
# Plot sur average profiles
fig_sur = plt.figure(figsize=(6, 4))
fig_sur.set_size_inches((6,4))
fig_sur.patch.set_facecolor(fcolor)
    
fig_sur.clf()
fsize = 14
    
p1 = fig_sur.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,surTaskProfiles,['iso90','iso0'],['darkviolet','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_sur.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,surTaskDiffs,['fgm'],['magenta'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_sur)

if savefigs:
    if use_decon:
        fig_sur.savefig(os.path.join(figDir,'fgm_profiles_decon_surround.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_sur.savefig(os.path.join(figDir,'fgm_profiles_sur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
#%% fgm Compare all diff profiles
# ctr vs border
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_bor_fgm = {'ctr_iso90': ctrTaskProfiles['iso90'], 'ctr_iso': ctrTaskProfiles['iso0'], 'bor_iso90': borTaskProfiles['iso90'], 'bor_iso': borTaskProfiles['iso0']}
plot_avg_depth_profile(p1,ctr_bor_fgm,['ctr_iso90', 'ctr_iso','bor_iso90', 'bor_iso'],['darkviolet','red','darkviolet','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,2]

ctr_border_avg = {'ctr': ctrTaskDiffs['fgm'], 'border': borTaskDiffs['fgm']}
ctr_border_profiles = {'ctr': diffProfiles['ctr']['fgm'], 'border': diffProfiles['border']['fgm']}
plot_profile_comparisons(p2,ctr_border_avg,ctr_border_profiles,['ctr','border'],['magenta','magenta'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['fgm_2samp'][('ctr', 'border')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'fgm_CtrVBor_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'fgm_CtrVBor.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# ctr vs sur
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_sur_fgm = {'ctr_iso90': ctrTaskProfiles['iso90'], 'ctr_iso': ctrTaskProfiles['iso0'], 'sur_iso90': surTaskProfiles['iso90'], 'sur_iso': surTaskProfiles['iso0']}
plot_avg_depth_profile(p1,ctr_sur_fgm,['ctr_iso90', 'ctr_iso','sur_iso90', 'sur_iso'],['darkviolet','red','darkviolet','red'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,2]

ctr_sur_avg = {'ctr': ctrTaskDiffs['fgm'], 'sur': surTaskDiffs['fgm']}
ctr_sur_profiles = {'ctr': diffProfiles['ctr']['fgm'], 'sur': diffProfiles['sur']['fgm']}
plot_profile_comparisons(p2,ctr_sur_avg,ctr_sur_profiles,['ctr','sur'],['magenta','magenta'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['fgm_2samp'][('ctr', 'sur')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'fgm_CtrVSur_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'fgm_CtrVSur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

#%% ctr-sur

pthresh = 0.05 #pval thresh

# Plot ctr average profiles
fig_ctr = plt.figure(figsize=(6, 4))
fig_ctr.set_size_inches((6,4))
fig_ctr.patch.set_facecolor(fcolor)
    
fig_ctr.clf()
fsize = 14
    
p1 = fig_ctr.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,ctrLocProfiles,['ctr','sur.1'],['gold','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_ctr.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,ctrLocDiffs,['ctr-sur'],['salmon'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_ctr)

if savefigs:
    if use_decon:
        fig_ctr.savefig(os.path.join(figDir,'ctr-sur_profiles_decon_center.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_ctr.savefig(os.path.join(figDir,'ctr-sur_profiles_ctr.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# Plot border average profiles
fig_bor = plt.figure(figsize=(6, 4))
fig_bor.set_size_inches((6,4))
fig_bor.patch.set_facecolor(fcolor)
    
fig_bor.clf()
fsize = 14
    
p1 = fig_bor.add_axes([.15, .2, .3, .7])
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
plot_avg_depth_profile(p1,borLocProfiles,['ctr','sur.1'],['gold','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_bor.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,1.5]
plot_avg_diff_profile(p2,borLocDiffs,['ctr-sur'],['salmon'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_border)

if savefigs:
    if use_decon:
        fig_bor.savefig(os.path.join(figDir,'ctr-sur_profiles_decon_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_bor.savefig(os.path.join(figDir,'ctr-sur_profiles_border.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
# Plot sur average profiles
fig_sur = plt.figure(figsize=(6, 4))
fig_sur.set_size_inches((6,4))
fig_sur.patch.set_facecolor(fcolor)
    
fig_sur.clf()
fsize = 14
    
p1 = fig_sur.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7

ylim = [-0.02,1.02]
xlim = [-2,6]
Ntext = [4,0.05]
plot_avg_depth_profile(p1,surLocProfiles,['ctr','sur.1'],['gold','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_sur.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-2,1.5]
plot_avg_diff_profile(p2,surLocDiffs,['ctr-sur'],['salmon'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=pvals_task_sur)

if savefigs:
    if use_decon:
        fig_sur.savefig(os.path.join(figDir,'ctr-sur_profiles_decon_surround.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_sur.savefig(os.path.join(figDir,'ctr-sur_profiles_sur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
        
#%% ctr-sur Compare all diff profiles

# ctr vs border
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_bor_fgm = {'ctr_ctr': ctrLocProfiles['ctr'], 'ctr_sur': ctrLocProfiles['sur.1'], 'bor_ctr': borLocProfiles['ctr'], 'bor_sur': borLocProfiles['sur.1']}
plot_avg_depth_profile(p1,ctr_bor_fgm,['ctr_ctr', 'ctr_sur','bor_ctr', 'bor_sur'],['gold','purple','gold','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-0.5,2]

ctr_border_avg = {'ctr': ctrLocDiffs['ctr-sur'], 'border': borLocDiffs['ctr-sur']}
ctr_border_profiles = {'ctr': diffProfiles['ctr']['ctr-sur'], 'border': diffProfiles['border']['ctr-sur']}
plot_profile_comparisons(p2,ctr_border_avg,ctr_border_profiles,['ctr','border'],['salmon','salmon'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['fgm_2samp'][('ctr', 'border')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'ctr-sur_CtrVBor_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'ctr-sur_CtrVBor.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')

# ctr vs sur
fig_comp = plt.figure(figsize=(6, 4))
fig_comp.set_size_inches((6,4))
fig_comp.patch.set_facecolor(fcolor)

fig_comp.clf()
fsize = 14
    
p1 = fig_comp.add_axes([.15, .2, .3, .7])
fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

if use_decon:
    dx = 4.
    dy = .7
else:
    dx = 1.
    dy = .7
    
ylim = [-0.02,1.02]
xlim = [-1,6]
Ntext = [4,0.05]

ctr_sur_fgm = {'ctr_ctr': ctrLocProfiles['ctr'], 'ctr_sur': ctrLocProfiles['sur.1'], 'sur_ctr': surLocProfiles['ctr'], 'sur_sur': surLocProfiles['sur.1']}
plot_avg_depth_profile(p1,ctr_sur_fgm,['ctr_ctr', 'ctr_sur','sur_ctr', 'sur_sur'],['gold','purple','gold','purple'],ylim,xlim,dx,dy,Ntext,lcolor,fsize)

p2 = fig_comp.add_axes([.7, .2, .25, .7])
fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
xlim = [-4,2]

ctr_sur_avg = {'ctr': ctrLocDiffs['ctr-sur'], 'sur': surLocDiffs['ctr-sur']}
ctr_sur_profiles = {'ctr': diffProfiles['ctr']['ctr-sur'], 'sur': diffProfiles['sur']['ctr-sur']}
plot_profile_comparisons(p2,ctr_sur_avg,ctr_sur_profiles,['ctr','sur'],['salmon','salmon'],['solid','dashed'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=all_pvals_dict['fgm_2samp'][('ctr', 'sur')])

if savefigs:
    if use_decon:
        fig_comp.savefig(os.path.join(figDir,'ctr-sur_CtrVSur_decon.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
    else:
        fig_comp.savefig(os.path.join(figDir,'ctr-sur_CtrVSur.%s' %(fig_format)), facecolor=fig.get_facecolor(), edgecolor='none')
