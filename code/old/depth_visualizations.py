#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:16:35 2023

@author: joe

Depth Visualization
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
from oriseg_funcs_v3 import *

plt.close('all')
    
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]
savefigs = False #True #if true save all figures
figDir = '/Users/joe/Documents/Olman_Lab/OriSeg/code/figs/'
fig_format = 'svg'
statCorrType = 'fdr_bh' #'bonferroni'

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
        
masks = {roi:all_data[roi]['sig']*all_data[roi]['no_vein'] for roi in all_data.keys()}

#%% Add in Differernce Conditions

diffDetails = {'odss': ['orth','iso90'],
                          'fgm': ['iso90','iso0'],
                          'dsi': ['orth','iso0'],
                          'iso-sur': ['iso0','sur'],
                          'iso90-sur': ['iso90','sur'],
                          'orth-sur': ['orth','sur']}

for roi in all_data.keys():
    for diff in diffDetails.keys():
        cond1 = diffDetails[diff][0]
        cond2 = diffDetails[diff][1]
        D = all_data[roi][cond1] - all_data[roi][cond2]
        all_data[roi][diff] = D
        
#%% Plot 3D Scatter plot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_3d_scatter(x, y, z, color_values, colormap='viridis', color_range=None, title=''):
    """
    Plots a 3D scatter plot with the given x, y, z values and corresponding color values mapped to a colormap.

    Parameters:
    x (list or array): x-coordinates of the points
    y (list or array): y-coordinates of the points
    z (list or array): z-coordinates of the points
    color_values (list or array): 1D array of floats to map to colors
    colormap (str): Name of the matplotlib colormap to use
    color_range (tuple, optional): Tuple of (min, max) values for color mapping
    """
    # Create a figure and add a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the color range if specified
    vmin, vmax = color_range if color_range else (min(color_values), max(color_values))

    # Scatter plot
    scatter = ax.scatter(x, y, z, c=color_values, cmap=colormap, vmin=vmin, vmax=vmax, alpha=0.8)

    # Adding a color bar
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

    # Adding labels
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.set_zlabel('Depth')
    ax.set_title(title)
    
    ax.set_box_aspect((np.ptp(x), np.ptp(y), 40*np.ptp(z)))

    # Show the plot
    plt.show()
    
plot_condition = 'ctr-sur_unwarp'
for roi in all_data.keys():
    
    df = all_data[roi]

    np.random.seed(0)  # For reproducible results
    x = df['x'].values
    x_masked = x[masks[roi]]
    y = df['y'].values
    y_masked = y[masks[roi]]
    z = df['d'].values
    z_masked = z[masks[roi]]
    z_norm = (z-np.min(z))/(np.max(z)-np.min(z))
    z_norm_masked = z[masks[roi]]
    color_values = df[plot_condition].values  # Array of float values
    color_values_masked = color_values[masks[roi]]
    
    plot_3d_scatter(x_masked, y_masked, z_norm_masked, color_values_masked, colormap='plasma', color_range=(-2, 2),title=roi)
    
#%% Now discretize the depth bins and plot

def discretize_z_values(z, nD=3):
    """
    Discretizes the z values into 3 bins that divide the range into three equal parts.

    Parameters:
    z (list or array): z-coordinates of the points

    Returns:
    np.ndarray: Array of binned z values
    """
    # Find the range of z values
    z_min = np.min(z)
    z_max = np.max(z)

    # Calculate the bin edges
    bins = np.linspace(z_min-1e-10, z_max+1e-10, nD+1)

    # Digitize the z values into bins
    z_binned = np.digitize(z, bins, right=True) - 1

    return z_binned

for roi in all_data.keys():
    
    df = all_data[roi]
    
    np.random.seed(0)  # For reproducible results
    x = df['x'].values
    x_masked = x[masks[roi]]
    y = df['y'].values
    y_masked = y[masks[roi]]
    z = df['d'].values
    z_masked = z[masks[roi]]
    z_norm = (z-np.min(z))/(np.max(z)-np.min(z))
    z_norm_masked = z[masks[roi]]
    color_values = df[plot_condition].values  # Array of float values
    color_values_masked = color_values[masks[roi]]
    
    z_binned = discretize_z_values(z)
    z_discrete = z_binned/np.max(z_binned)
    z_discrete_masked = z_discrete[masks[roi]]

    plot_3d_scatter(x_masked, y_masked, z_discrete_masked, color_values_masked, colormap='plasma', color_range=(-2, 2),title=roi)
    