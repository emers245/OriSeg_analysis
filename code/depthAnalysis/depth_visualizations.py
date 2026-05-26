#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""
Created on Mon Nov 20 16:16:35 2023

@author: joe

Depth Visualization
"""
#import nibabel
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

# Add funcs path
funcs_dir = '..'
if funcs_dir not in sys.path:
    sys.path.append(funcs_dir)

#Import custom functions
from oriseg_funcs import *

plt.close('all')
    
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]
savefigs = True #True #if true save all figures
mainDir = '../..'
figDir = os.path.join(mainDir, 'figs', 'depth_visualization')
fig_format = 'svg'
statCorrType = 'fdr_bh' #'bonferroni'

#%%###########################################################################
#############################################################################
########### Notice that each hemisphere is treated as a dataset

datasets = glob.glob(os.path.join(mainDir, 'data', 'roi_data/target_roi_manual', 'pnr???_??_???_??.csv'))
#datasets = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_filled', 'pnr???_??_???_??_rad10.csv'))
#or exclude
exclude_initial = ['pnr143_V1_tgt_rh',
                   'pnr161_V1_tgt_lh','pnr161_V1_tgt_rh',
                   'pnr352_V1_tgt_lh','pnr352_V1_tgt_rh',
                   'pnr579_V1_tgt_lh',
                   'pnr668_V1_tgt_rh']
#exclude_initial = ['pnr352_V1_tgt_lh_rad10']
for e_i, excl in enumerate(exclude_initial):
    datasets.remove(os.path.join(mainDir, 'data', 'roi_data/target_roi_manual',excl+'.csv'))
    #datasets.remove(os.path.join(mainDir,'roi_data_manualSeg/target_filled',excl+'.csv'))        
datasets.sort()

roiRad = 1. #2
import pandas as pd
all_data = {}
for dataset in datasets:
    p, f = os.path.split(dataset)
    f, ex = os.path.splitext(f)
    all_data[f] = pd.read_csv(dataset, sep=',', index_col=False)
    
## THIS IS A HACKY FIX TO GET RID OF DEPTH = 0 VOXELS;
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

if os.path.exists(figDir) == False:
    os.makedirs(figDir)

if savefigs:
    fig.savefig(os.path.join(figDir,'t1w_profiles.%s' %(fig_format)))
    
#%% Check on xy to uv mapping in 2D space first

ellipse_params = {}
show_rad = False
frad = plt.figure(figsize=(8,8))
floc = plt.figure(figsize=(8,8))
fsize = plt.figure(figsize=(8,8))
a_vec = np.zeros((len(all_data.keys()),))
b_vec = np.zeros((len(all_data.keys()),))
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    subjID = label[0:6]
    hemi = label[14:16]

    # redo the ellipse fitting, which is a bit of overkill, but it gets us 
    # an accurate ellipse
    sig = df['loc p-val'] < 0.01
    tgt_df = df[(df['ctr-sur_unwarp'] > 0) & sig] #df[df['ctr-sur'] > 0]
    #tgt_df = df[df['scale_xy_dist'] <= 2]
    cov = np.cov(tgt_df['x'][df['scale_xy_dist'] < 2.2],
                 tgt_df['y'][df['scale_xy_dist'] < 2.2])
    com = (np.mean(tgt_df['x'][df['scale_xy_dist'] < 2.2]),
           np.mean(tgt_df['y'][df['scale_xy_dist'] < 2.2]))
    a = (cov[0,0] + cov[1,1])/2 + np.sqrt(((cov[0,0] - cov[1,1])/2)**2 + cov[0,1]**2)
    b = (cov[0,0] + cov[1,1])/2 - np.sqrt(((cov[0,0] - cov[1,1])/2)**2 + cov[0,1]**2)
    a_vec[iR] = a
    b_vec[iR] = b
    print('avg radius: %2.2f' %(2*(np.sqrt(a) + np.sqrt(b))/2))
    theta = np.arctan2(a - cov[0,0], cov[1,0])
    ellipse = Ellipse(com,
                      width=2*2*np.sqrt(a),
                      height=2*2*np.sqrt(b),
                      angle=180*theta/np.pi,
                      zorder=100, alpha=1., edgecolor='r', facecolor='None')
    
    #Save ellipse params
    ellipse_params[label] = {'a': a, 'b': b, 'theta': theta, 'com': com}
    
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
    all_data[key]['mnv'] = mnv
    
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

def plot_3d_scatter(x, y, z, color_values, marker_size=20, colormap='viridis', color_range=None, title='', labels = ['U','V','Depth'], box_aspect = [1,1,40], cbar_label = None):
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
    scatter = ax.scatter(x, y, z, c=color_values, s=marker_size, cmap=colormap, vmin=vmin, vmax=vmax, alpha=0.8)

    # Adding a color bar
    cb = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    cb.set_label(cbar_label)

    # Adding labels
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    
    ax.set_box_aspect((box_aspect[0]*np.ptp(x), box_aspect[1]*np.ptp(y), box_aspect[2]*np.ptp(z)))

    # Show the plot
    plt.show()
    
plot_condition = 'ctr-sur_unwarp'
color_range = (-1,1)
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
    
    plot_3d_scatter(x_masked, y_masked, z_norm_masked, color_values_masked, colormap='plasma', color_range=color_range,title=roi,cbar_label=plot_condition)

    # Save figure    if savefigs:
    plt.savefig(os.path.join(figDir,'3d_scatter_%s.%s' % (roi, fig_format)))

#%% Plot in Original XYZ space

plot_condition = 'ctr-sur_unwarp'#'no_vein'#'d'#'ctr-sur_unwarp'
color_range = (-1,1)#(None,None)#(0,1)#(-2,2)
for roi in all_data.keys():
    
    df = all_data[roi]

    np.random.seed(0)  # For reproducible results
    x = df['xv'].values
    x_masked = x[masks[roi]]
    y = df['yv'].values
    y_masked = y[masks[roi]]
    z = df['zv'].values
    z_masked = z[masks[roi]]
    color_values = df[plot_condition].values  # Array of float values
    color_values_masked = color_values[masks[roi]]
    
    plot_3d_scatter(x, y, z, color_values, colormap='plasma', color_range=color_range, title=roi, labels = ['X', 'Y', 'Z'], box_aspect = [1,1,1])
    #plot_3d_scatter(x_masked, y_masked, z_masked, color_values_masked, colormap='plasma', color_range=color_range, title=roi, labels = ['X', 'Y', 'Z'], box_aspect = [1,1,1])

    # Save figure
    if savefigs:
        plt.savefig(os.path.join(figDir,'3d_scatter_xyz_%s.%s' % (roi, fig_format)))
    
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

plot_condition = 'ctr-sur_unwarp'
color_range = (-1,1)
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

    plot_3d_scatter(x_masked, y_masked, z_discrete_masked, color_values_masked, colormap='plasma', color_range=color_range,title=roi)
    
    # Save figure
    if savefigs:
        plt.savefig(os.path.join(figDir,'3d_scatter_discrete_depth_%s.%s' % (roi, fig_format)))
    
#%% Visualize impact of deveining
# Calculate log MNV
for roi in all_data.keys():
    all_data[roi]['log(mnv)'] = np.log(all_data[roi]['mnv'])

# Before Deveining
plot_condition = 'log(mnv)'
color_range = (2,5)
for roi in all_data.keys():
    
    df = all_data[roi]
    
    np.random.seed(0)  # For reproducible results
    x = df['x'].values
    x_masked = x
    y = df['y'].values
    y_masked = y
    z = df['d'].values
    z_masked = z
    z_norm = (z-np.min(z))/(np.max(z)-np.min(z))
    z_norm_masked = z
    color_values = df[plot_condition].values  # Array of float values
    color_values_masked = color_values
    
    z_binned = discretize_z_values(z)
    z_discrete = z_binned/np.max(z_binned)
    z_discrete_masked = z_discrete

    plot_3d_scatter(x_masked, y_masked, z_discrete_masked, color_values_masked, marker_size=2, colormap='Reds', color_range=color_range,title=roi, cbar_label = "log(MNV)")

    # Save figure
    if savefigs:
        plt.savefig(os.path.join(figDir,'3d_scatter_discrete_depth_before_deveining_%s.%s' % (roi, fig_format)))

#%% After Deveining
plot_condition = 'log(mnv)'
color_range = (2,5)
for roi in all_data.keys():
    
    df = all_data[roi]
    mask = df['no_vein']
    
    np.random.seed(0)  # For reproducible results
    x = df['x'].values
    x_masked = x[mask]
    y = df['y'].values
    y_masked = y[mask]
    z = df['d'].values
    z_masked = z[mask]
    z_norm = (z-np.min(z))/(np.max(z)-np.min(z))
    z_norm_masked = z[mask]
    color_values = df[plot_condition].values  # Array of float values
    color_values_masked = color_values[mask]
    
    z_binned = discretize_z_values(z)
    z_discrete = z_binned/np.max(z_binned)
    z_discrete_masked = z_discrete[mask]

    plot_3d_scatter(x_masked, y_masked, z_discrete_masked, color_values_masked, marker_size=2, colormap='Reds', color_range=color_range,title=roi, cbar_label = "log(MNV)")

    # Save figure
    if savefigs:
        plt.savefig(os.path.join(figDir,'3d_scatter_discrete_depth_after_deveining_%s.%s' % (roi, fig_format)))
    
#%% Plot Histograms

plot_condition = 'log(mnv)'
x_range = (2,5)
xlabel = "log(MNV)"
ylabel = "Voxel Count"
Nbins = 50
depth_bins = {"superficial": [2/3,1], "middle": [1/3,2/3], "deep": [0,1/3]}
for roi in all_data.keys():
    
    df = all_data[roi]
    bins = np.linspace(x_range[0],x_range[1],Nbins)
    thresh = lmnv_dict[roi]['thresh']
    counts, bins = np.histogram(df[plot_condition],bins=bins)
    z = df['d'].values
    z_norm = (z-np.min(z))/(np.max(z)-np.min(z))
    
    fig, ax = plt.subplots(3,1)
    fig.suptitle(roi)
    for d_i, d_label in enumerate(depth_bins.keys()):
        depth_mask = (z_norm >= depth_bins[d_label][0]) & (z_norm < depth_bins[d_label][1])
        hist_values = df[plot_condition].values
        hist_values_masked = hist_values[depth_mask]
        
        ax[d_i].hist(hist_values_masked,bins=bins,color='r')
        ax[d_i].vlines(thresh,np.min(counts),np.max(counts),'k', linestyle = '--',label="Threshold")
        ax[d_i].set_xlabel(xlabel)
        ax[d_i].set_xlim(x_range)
        ax[d_i].set_ylabel(ylabel)
        ax[d_i].set_ylim([0,np.max(counts)/2+10])
        ax[d_i].set_title(d_label)
        ax[d_i].legend()
    
    plt.tight_layout()
    
    # Save figure
    if savefigs:
        plt.savefig(os.path.join(figDir,'histogram_%s.%s' % (roi, fig_format)))
# %%
