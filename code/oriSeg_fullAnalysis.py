#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:27:28 2024

@author: Joe

OriSeg Full Analysis: Joe was finding the previous code base to become very 
unweildy, so this code combines all previous code into a single script. The
figures saved here are primarily figures that appear in the manuscript.

"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import csv

#Import custom functions
from oriseg_funcs import *

plt.close('all')
    
fcolor = 'white'#[.125, .125, .125]
lcolor = 'black'##[1., 1., 1.]
savefigs = True #if true save all figures
#mainDir = '/home/scat-raid3/data/oriSeg'
mainDir = '.'
figDir = mainDir+'/figs/subjAvg/'
fig_format = 'svg'

#Set random seed
np.random.seed(68752)

#%% ~ Set Parameters ~

# Set Radial Distance Parameters
roiRad = 1. #radius for target ROI
centerRad = 1. #radius for center of ROI
borderRad = [1.,3.] #radius range for border
surRad = [3.] #[3, 3.5] #outside of this radius will be considered the surround
nRad = 5 #how many radius bins
maxRad = 5 #maximum radius
rad_depth_labels = ['all depths'] #['deep', 'middle', 'superficial']
rad_depthBoundaries = np.array([[0,1]]) #np.array([[0,1/3],[1/3,2/3],[2/3,1]])
radBin_comparisons = [[0,4]] #indices of the radial bins to compare

# Set Depth Parameters
nDepths = 7
nDepths_rings = 3
nDepths_gPPI = 3

# Set Mask Parameters
use_fullmodel_mask = True #if true mask out voxels with non-significant iso+iso90+orth+sur vs blank contrast
use_loc_mask = False #True #if true, mask out voxels with non-significant tgt-sur contrast
useSI = False #use suppression index rather than differences (cond1 - cond2 / cond1 + cond2)
pthresh_fullmodel = 0.01 #p-value to use as threshold for individual voxels based on task GLM
pthresh_loc = 0.01 #p-value to use as threshold for individual voxels based on localizer GLM

# Set Statistics parameters
pthresh = 0.05 #p-value threshold for statistical analyses at the subject level
prop_err = False #propogate individual subject error into across subject error?
showSig = True #show significance?
compareRadtoNull = False #compare radial profiles to random permutation over radius
statCorrType = 'fdr_bh' #'bonferroni' #type of correction for multiple comparisons
statTestType = 'permutation' #which kind of statistical test to use ('t-test' for parameteric statistics or 'permutation' for permutation test)
Npermutations = 10000

# Depth Deconvolution
use_decon = {'task': True,
             'loc': True,
             'gPPI': False} #if true, use deconvolved profiles
p2t_model = 6.3 #peak to tail ratio from Markuerkiaga et al. (2021) estimated for TE = 33.3 ms    
Nbins_model = 10 #number of bins used in the model from Markuerkiaga et al. (2021)
normalize_psf = False #True if you want to normalize the psf by the deepest layer  


# Which subjects to use for which analyses
subj_analyses = {'task': 'all', #task condition analysis
                 'loc': 'all', #localizer condition analysis
                 'gPPI':        #gPPI analysis
                     ['pnr256','pnr328','pnr495','pnr510','pnr579','pnr668','pnr685','pnr713','pnr739','pnr756','pnr822']
                 }

def return_included_subj(subj_analyses, analysis_names):
    # A function that returns the subjects to be included in a given analysis
    
    def check_subj_labels(labels1, labels2):
        if labels1 == labels2:
            return labels1
        elif len(labels1) == 0:
            if len(labels2) > 0:
                return labels2
            else:
                print("Empty labels.")
                return []
        elif len(labels2) == 0:
            if len(labels1) > 0:
                return labels1
            else:
                print("Empty labels.")
                return []
        else:
            print("Inconsistent subject list. Returning the intersection of the analyses.")
            intersection = list(set(labels1) & set(labels2))
            return intersection
    
    if type(analysis_names) == str:
        analysis_names = [analysis_names]
    
    included_IDs = []
    for analysis_name in analysis_names:
        if subj_analyses[analysis_name] == 'all':
            included_data = all_data
        elif type(subj_analyses[analysis_name]) == list:
            included_data = {label: all_data[label] for label in subj_analyses[analysis_name]}
        else:
            print("Invalid subj_analyses dictionary")
            return None
        included_IDs = check_subj_labels(list(included_data.keys()),included_IDs)
    included_IDs.sort()
    included_data = {label: all_data[label] for label in included_IDs}
        
    return included_data

#%% ~ Import Data ~

mainDir = '.'
datasets_V1 = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual', 'pnr???_??_???_??.csv'))
datasets_V23 = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/V23_roi_manual', 'pnr???_???_???_??.csv'))

# Exclude Datasets
# For summary of inclusion criteria, see https://docs.google.com/spreadsheets/d/1Y1U2Cm6C-CaQuBdKQHADqLZ4rSNpV4rPMdNPii-WPtg/edit?usp=sharing
exclude_V1 = ['pnr143_V1_tgt_rh',
                   'pnr161_V1_tgt_lh','pnr161_V1_tgt_rh',
                   'pnr352_V1_tgt_lh','pnr352_V1_tgt_rh',
                   'pnr579_V1_tgt_lh',
                   'pnr668_V1_tgt_rh']
exclude_V23 = []
for e_i, excl in enumerate(exclude_V1):
    datasets_V1.remove(os.path.join(mainDir,'roi_data_manualSeg/target_roi_manual',excl+'.csv'))     
datasets_V1.sort()
Ndsets_V1 = len(datasets_V1)
print(f"{Ndsets_V1} V1 ROIs")
for e_i, excl in enumerate(exclude_V23):
    datasets_V23.remove(os.path.join(mainDir,'roi_data_manualSeg/V23_roi_manual',excl+'.csv'))     
datasets_V23.sort()
Ndsets_V23 = len(datasets_V23)
print(f"{Ndsets_V23} V2/3 ROIs")

# Combine datasets
datasets = datasets_V1 + datasets_V23

# Save data to pandas DataFrames
#   all_data is a dictionary containing a pandas dataframe for each subject
all_data = {}

# Loop through each dataset and process the files
for file_path in datasets:
    # Extract metadata from the file name
    file_name = os.path.basename(file_path)
    subjID, visArea, subReg, hemi = file_name.replace('.csv', '').split('_')

    # Load the CSV file into a DataFrame, ensuring the first column is not used as the index
    df = pd.read_csv(file_path, index_col=False)

    # Add metadata columns to the DataFrame
    df['Subject ID'] = subjID
    df['Visual Region'] = visArea
    df['Subregion'] = subReg
    df['hemi'] = hemi

    # If the subject is already in the dictionary, concatenate the new data
    if subjID in all_data:
        all_data[subjID] = pd.concat([all_data[subjID], df], ignore_index=True)
    else:
        all_data[subjID] = df
    
## Occasionally, voxels will not have depth assignments. Remove voxels with depth = 0.
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    
    df = df.drop(df[df['d'] == 0].index)
    
    all_data[label] = df

#%% ~ Visualize Data ~
# The next few code blocks do some initial visualizations so that we know what
# we are looking at.

#%% Check T1w profiles

# Check to see what the Stria profile looks like in each ROI
fig = plt.figure(num=1)
fig.clf()
for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    df = df[df['Visual Region'] == 'V1']
    df = df[df['Subregion'] == 'tgt']
    
    roi = df[df['scale_xy_dist'] < roiRad]
    roi = roi[roi['scale_xy_dist'] > 0]
    dataDict = makeProfile1D(roi['d'].values,
                             7, #number of depths
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
    
#%% Visualize Elliptical Fit on Each V1 ROI
from matplotlib.patches import Ellipse

frad = plt.figure(figsize=(6.5,8))
floc = plt.figure(figsize=(6.5,8))
iR = 0

# Save ellipse parameters in separate dataframe for storage elsewhere
ellipse_df = pd.DataFrame({'subjID': [], 'hemi': [], 'ROI major axis (mm)': [], 'ROI minor axis (mm)': [], 'theta': [], 'comX (mm)': [], 'comY (mm)': [], 'area (mm^2)': [], 'semimajor axis (mm)': [], 'semiminor axis (mm)': []})

for iS, label in enumerate(all_data.keys()):
    df_all = all_data[label]
    
    # We're going to be creating some new columns to hold values for ellipse
    for col in ['xy_dist','ellipse_a', 'ellipse_b', 'ellipse_theta', 'ellipse_comX', 'ellipse_comY']:
        df_all[col] = np.nan  # Create the columns in df_all with NaN values by default
    
    for iH, hemi in enumerate(df_all['hemi'].unique()):
        
        if np.sum((df_all['Visual Region'] == 'V1') & (df_all['Subregion'] == 'tgt') & (df_all['hemi']==hemi)) != 0:
        
            df = df_all[(df_all['Visual Region'] == 'V1') & (df_all['Subregion'] == 'tgt') & (df_all['hemi']==hemi)]
    
            # redo the fitting, which is a bit of overkill, but it gets us 
            # an accurate ellipse
            tgt_df = df[df['ctr-sur'] > 0]
            #tgt_df = df[df['scale_xy_dist'] <= 2]
            cov = np.cov(tgt_df['x'][df['scale_xy_dist'] < 2.2],
                         tgt_df['y'][df['scale_xy_dist'] < 2.2])
            com = (np.mean(tgt_df['x'][df['scale_xy_dist'] < 2.2]),
                   np.mean(tgt_df['y'][df['scale_xy_dist'] < 2.2]))
            
            # Get an xy-distance measure in mm in addition to the normalized version
            df['xy_dist'] = np.sqrt((df['x'].values-com[0])**2 + (df['y'].values-com[1])**2)
            # Update the 'xy_dist' in the original dataframe 'df_all'
            df_all.loc[(df_all['hemi'] == hemi) & (df_all['Visual Region'] == 'V1') & (df_all['Subregion'] == 'tgt'), 'xy_dist'] = df['xy_dist']
            
            # Recompute the major and minor axes of the ellipse
            a = (cov[0,0] + cov[1,1])/2 + np.sqrt(((cov[0,0] - cov[1,1])/2)**2 + cov[0,1]**2)
            b = (cov[0,0] + cov[1,1])/2 - np.sqrt(((cov[0,0] - cov[1,1])/2)**2 + cov[0,1]**2)
            print(label)
            print('\t %s major axis (a): %2.2f' %(hemi,a))
            print('\t %s minor axis (b): %2.2f' %(hemi,b))
            theta = np.arctan2(a - cov[0,0], cov[1,0])
            df['ellipse_a'] = a
            df['ellipse_b'] = b
            df['ellipse_theta'] = theta
            df['ellipse_comX'] = com[0]
            df['ellipse_comY'] = com[1]
            ellipse_df.loc[len(ellipse_df)] = [label, hemi, roiRad*np.sqrt(a), roiRad*np.sqrt(b), theta, com[0], com[1], np.pi*(roiRad*np.sqrt(a))*(roiRad*np.sqrt(b)), (roiRad*np.sqrt(a) + roiRad*np.sqrt(b))/2, np.sqrt((roiRad*np.sqrt(a))*(roiRad*np.sqrt(b)))]
            ellipse = Ellipse(com,
                              width=2*roiRad*np.sqrt(a),
                              height=2*roiRad*np.sqrt(b),
                              angle=180*theta/np.pi,
                              zorder=100, alpha=1., edgecolor='r', facecolor='None')
            # show localizer data
            minx = np.min(df['x'].values)
            miny = np.min(df['y'].values)
            ax1 = frad.add_subplot(int(np.ceil(Ndsets_V1/2)),2,iR+1)
            
            # Plot the radius determined by the normalized uv coordinates (this should be in SD of a 2D Gaussian fitted to the loc data)
            cmap = plt.cm.get_cmap('viridis')
            cmap_rev = cmap.reversed()
            pcm = ax1.scatter(df['x'],df['y'],c=df['scale_xy_dist'],s=0.05,cmap=cmap_rev)
            cbar = plt.colorbar(pcm,ax=ax1)
            ax1.add_patch(ellipse)
            ax1.patch.set_facecolor('r')
            ax1.set_title(label+" radius = %.1f$\sigma$: SD<%.1f Nvox = %d" %(roiRad,roiRad,np.sum(df['scale_xy_dist']<roiRad)),fontsize=6)
            ax1.axis('off')
            ax1.set_aspect('equal')
            
            # Plot the ctr-sur betas
            ellipse2 = Ellipse(com,
                              width=2*roiRad*np.sqrt(a),
                              height=2*roiRad*np.sqrt(b),
                              angle=180*theta/np.pi,
                              zorder=100, alpha=1., edgecolor='r', facecolor='None')
            ax2 = floc.add_subplot(int(np.ceil(Ndsets_V1/2)),2,iR+1)
            pcm = ax2.scatter(df['x'],df['y'],c=df['ctr-sur_unwarp'],s=0.05,cmap=cmap,vmin=-2,vmax=2)
            cbar = plt.colorbar(pcm,ax=ax2)
            ax2.add_patch(ellipse2)
            ax2.patch.set_facecolor('r')
            ax2.set_title(label+" localizer: SD<%.1f Nvox = %d" %(roiRad,np.sum(df['scale_xy_dist']<roiRad)),fontsize=6)
            ax2.axis('off')
            ax2.set_aspect('equal')
            
            iR += 1
            
            #Save ellipse parameters back to original dataframe
            for col in ['xy_dist','ellipse_a', 'ellipse_b', 'ellipse_theta', 'ellipse_comX', 'ellipse_comY']:
                df_all.loc[df.index, col] = df[col]  # Update values for rows that are in df
            
        else:
            print(f"No V1_tgt_{hemi} for {label}")
    
    #save back to main dataframe
    all_data[label] = df_all
    
# Compute statistics across all ROIs
avgs = ellipse_df.mean(axis=0, numeric_only = True)
stds = ellipse_df.std(axis=0, numeric_only = True)
mins = ellipse_df.min(axis=0, numeric_only=True)
maxs = ellipse_df.max(axis=0, numeric_only=True)
Nrois = len(ellipse_df)

ellipse_stats = pd.DataFrame({'avg': avgs, 'st. dev': stds, 'min': mins, 'max': maxs, 'NROIs': Nrois*np.ones(len(avgs))})
    
if savefigs:
    frad.savefig(os.path.join(figDir,'xy_map_rad.%s' %(fig_format)))
    floc.savefig(os.path.join(figDir,'xy_map_loc.%s' %(fig_format)))
    ellipse_df.to_csv(os.path.join(figDir,'stats','ROIstats.csv'))
    ellipse_stats.to_csv(os.path.join(figDir,'stats','ROIsummary.csv'))
    
#%% Histograms of p-values

# Get all unique combinations of subject and visual region
subject_visArea_combinations = [(subjID, vis_region) for subjID, df in all_data.items() for vis_region in df['Visual Region'].unique()]

# Plot loc p-val histograms for each subject and visual region
fig_loc, axes_loc = plt.subplots(2, int(np.ceil(len(subject_visArea_combinations)/2)), figsize=(5 * len(subject_visArea_combinations), 12))
if len(subject_visArea_combinations) == 1:
    axes_loc = [axes_loc]
for i, (subjID, vis_region) in enumerate(subject_visArea_combinations):
    r, c = np.unravel_index(i,(2, int(np.ceil(len(subject_visArea_combinations)/2))))
    ax = axes_loc[r,c]
    df = all_data[subjID]
    roi = df[(df['Visual Region'] == vis_region) & ((df['scale_xy_dist'] < roiRad) | (df['scale_xy_dist'].isna()))]
    if 'loc pval' in roi.keys():
        roi = roi.rename(columns={'loc pval': 'loc p-val'})
    if vis_region == 'V1':
        color = 'b'
    elif vis_region == 'V23':
        color = 'r'
    else:
        color = 'gray'
    minp = 10**-3
    maxp = 1
    logbins = np.logspace(np.log(minp),np.log(maxp),20)
    ax.hist(roi['loc p-val'].values, bins=logbins, density=True, alpha=0.5, color=color)
    ax.set_xscale('log')
    p_less_05 = 100 * (roi['loc p-val'] <= 0.05).sum() / len(roi['loc p-val']) if len(roi['loc p-val']) > 0 else 0
    ax.set_title(f"{subjID} - {vis_region} loc p-val \n (p < 0.05: {p_less_05:.2f}%)", fontsize=6)
    ax.set_xlabel("pval")
    ax.set_ylim([0, 100])
    ax.set_xlim([minp, maxp])
    ax.tick_params(axis='x', labelsize=6)
fig_loc.tight_layout(pad=0.5)

# Plot task p-val histograms for each subject and visual region
fig_task, axes_task = plt.subplots(2, int(np.ceil(len(subject_visArea_combinations)/2)), figsize=(5 * len(subject_visArea_combinations), 12))
if len(subject_visArea_combinations) == 1:
    axes_task = [axes_task]
for i, (subjID, vis_region) in enumerate(subject_visArea_combinations):
    r, c = np.unravel_index(i,(2, int(np.ceil(len(subject_visArea_combinations)/2))))
    ax = axes_task[r,c]
    df = all_data[subjID]
    roi = df[(df['Visual Region'] == vis_region) & ((df['scale_xy_dist'] < roiRad) | (df['scale_xy_dist'].isna()))]
    if 'task pval' in roi.keys():
        roi = roi.rename(columns={'task pval': 'task p-val'})
    if vis_region == 'V1':
        color = 'b'
    elif vis_region == 'V23':
        color = 'r'
    else:
        color = 'gray'
    minp = 10**-3
    maxp = 1
    logbins = np.logspace(np.log(minp),np.log(maxp),20)
    ax.hist(roi['task p-val'].values, bins=logbins, density=True, alpha=0.5, color=color)
    ax.set_xscale('log')
    p_less_05 = 100 * (roi['task p-val'] <= 0.05).sum() / len(roi['task p-val']) if len(roi['task p-val']) > 0 else 0
    ax.set_title(f"{subjID} - {vis_region} task p-val \n (p < 0.05: {p_less_05:.2f}%)", fontsize=6)
    ax.set_xlabel("pval")
    ax.set_ylim([0, 100])
    ax.set_xlim([minp, maxp])
    ax.tick_params(axis='x', labelsize=6)
fig_task.tight_layout(pad=0.5)

# Save figures if needed
if savefigs:
    fig_loc.savefig(os.path.join(figDir, f'pvals_loc.{fig_format}'))
    fig_task.savefig(os.path.join(figDir, f'pvals_task.{fig_format}'))
    
#%% Depth Histograms

# I want to see how much coverage we are getting through depth.
fig_dhist, axes_dhist = plt.subplots(2, int(np.ceil(len(subject_visArea_combinations)/2)), figsize=(5 * len(subject_visArea_combinations), 12))
if len(subject_visArea_combinations) == 1:
    axes_loc = [axes_loc]

for i, (subjID, vis_region) in enumerate(subject_visArea_combinations):
    r, c = np.unravel_index(i,(2, int(np.ceil(len(subject_visArea_combinations)/2))))
    ax = axes_dhist[r,c]
    df = all_data[subjID]
    roi = df[(df['Visual Region'] == vis_region) & ((df['scale_xy_dist'] < roiRad) | (df['scale_xy_dist'].isna()))]
    if vis_region == 'V1':
        color = 'b'
    elif vis_region == 'V23':
        color = 'r'
    else:
        color = 'gray'
    ax.hist(roi['d'].values, bins=nDepths, density=False, alpha=0.5, color=color)
    ax.set_title(f"{subjID} - {vis_region} depth", fontsize=6)
    ax.set_xlabel("Normalize Depth WM -> GM")
    ax.set_ylabel("Num. Voxels")
    #ax.set_ylim([0, 10])
    ax.set_xlim([0, 1])
    plt.legend(['N='+str(len(roi)),], fontsize = 6)
fig_dhist.tight_layout(pad=0.5)

if savefigs:
    fig_dhist.savefig(os.path.join(figDir, f'pvals_loc.{fig_format}'))

#%% ~ Analysis Preprocessing ~
# Before diving into the analysis we have to clean up the data by removing
# venous voxels, removing visually unresponsive voxels, and deconvolving the 
# data.
    
#%% Use the deepest layer as a proxy for non-vein contaminated voxels
# Then define the threshold based on this distribution

# Function to save dropout statistics to CSV
def save_dropout_statistics_to_csv(vArea, subjIDs, hemi, dropout, voxels_before, voxels_after, out_dir='output'):
    # Create the output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Create a DataFrame to store dropout statistics
    data = {
        'Subject ID': subjIDs,
        'Hemi': hemi,
        'Voxels Before Deveining': voxels_before,
        'Voxels After Deveining': voxels_after,
        'Total Dropout Rate': dropout['total'],
        'Superficial Dropout Rate': dropout['superficial'],
        'Middle Dropout Rate': dropout['middle'],
        'Deep Dropout Rate': dropout['deep'],
    }
    df = pd.DataFrame(data)

    # Calculate mean and std and add them as new rows
    mean_data = df.mean(numeric_only=True)
    std_data = df.std(numeric_only=True)

    # Convert mean and std Series to DataFrames with the correct structure
    mean_data['Subject ID'] = 'Mean'
    std_data['Subject ID'] = 'Std Dev'
    mean_data = pd.DataFrame([mean_data])
    std_data = pd.DataFrame([std_data])

    # Concatenate the mean and std rows to the main DataFrame
    df = pd.concat([df, mean_data, std_data], ignore_index=True)

    # Save the DataFrame to a CSV file
    filename = f"{vArea}_dropout_statistics.csv"
    filepath = os.path.join(out_dir, filename)
    df.to_csv(filepath, index=False)

    print(f"Saved dropout statistics to {filepath}")

# Normalize 'd' within each subject, visual region, subregion, and hemisphere
for subjID, df in all_data.items():
    all_data[subjID]['d_norm'] = df.groupby(['Visual Region', 'Subregion', 'hemi'])['d'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

deep_pct = 10 #percentile to call deep layers
conditions = ['iso0','iso90','orth','sur']
depth_groups = {'deep': [0.0,1/3], 'middle': [1/3,2/3], 'superficial': [2/3,1.0]}
depth_labels = ['superficial','middle','deep'] #put them in the right order
depth_var = 'd_norm'
x_var = 'x'
y_var = 'y'
sd_thresh = 2 #how many st. dev. of the deep layer mean to use as the threshold
mask_dict = {} #create a mask dictionary
lmnv_dict = {key:{'mean':0,'std':0,'thresh':0,'deep_mean':0,'deep_std':0} for key in all_data.keys()} #thresh dictionary
fsize=8 #fontsize of title
Ngroups = len(depth_groups.keys())
NROIs = len(datasets)

for iR, vArea in enumerate(['V1','V23']):
    k_i = 0
    subjIDs = [subj for subj in all_data.keys() if all_data[subj]['Visual Region'].str.contains(vArea).any()]
    dsets = []
    for iH, hemi in enumerate(['lh','rh']):
        dsets = dsets + [subj+hemi for subj in all_data.keys() if ((all_data[subj]['Visual Region'].str.contains(vArea)) & (all_data[subj]['hemi'].str.contains(hemi))).any()]
    Ndsets = len(dsets)
    dropout = {'superficial': np.zeros(Ndsets),
               'middle': np.zeros(Ndsets),
               'deep': np.zeros(Ndsets),
               'total': np.zeros(Ndsets)} #dropout rates
    
    voxels_before = []
    voxels_after = []
    all_hemis = []
    all_subj = []

    for iS, label in enumerate(subjIDs):
        df_all = all_data[label]
        
        for iH, hemi in enumerate(df_all['hemi'].unique()):
            
            if np.sum((df_all['Visual Region'] == vArea) & (df_all['Subregion'] == 'tgt') & (df_all['hemi']==hemi)) != 0:
            
                df = df_all[(df_all['Visual Region'] == vArea) & (df_all['Subregion'] == 'tgt') & (df_all['hemi']==hemi)]
    
                # calculate log(MNV)
                lmnv = get_lmnv(df,key='stdev_xerrts') #log of the mean-normalized variance
                mnv = np.exp(lmnv) #get back mean normalized variance
                
                # get deep layer distribution
                z = df[depth_var]
                [deep_mean, deep_std, deep] = get_deep_layer_dist(df,depth_var,deep_pct)
                lmnv_dict[label]['deep_mean'] = deep_mean
                lmnv_dict[label]['deep_std'] = deep_std
                
                # define threshold based on deep layer distribution
                [mnv_mask, lmnv_thresh] = get_mnv_mask(df,depth_var,deep_pct,sd_thresh)
                lmnv_dict[label]['thresh'] = lmnv_thresh
                lmnv_dict[label]['mean'] = np.mean(lmnv)
                lmnv_dict[label]['std'] = np.std(lmnv)
                
                # Plot distributions
                fthresh = plot_mnv_histograms(lmnv, lmnv[deep], mnv_mask, deep_pct, label, k_i, Ndsets, fsize, pad=0.0, figsize=(15,3))
                
                # Plot depth maps
                dmap = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, Ndsets, [2,5], fsize, fname = 'dmap', pad=0.0)
                    
                #plot thresholded map
                dmap_thresh = plot_depth_maps(df, depth_var, depth_groups, depth_labels, x_var, y_var, mnv, k_i, Ndsets, [2,5], fsize, fname='dmap_thresh', mask=mnv_mask, pad=0.0)
                    
                # Plot voxel loss at each depth after masking
                fdepth_hist = plot_depth_voxel_loss(z, mnv_mask, nDepths, Ndsets, label, k_i, fsize)
                
                # report number of voxels after threshold
                voxels_before.append(np.size(mnv))
                voxels_after.append(np.sum(mnv_mask))
                all_hemis.append(hemi)
                all_subj.append(label)
                
                # report number of voxels after threshold
                print("%d/%d Voxels Survive for %s %s %s" %(np.sum(mnv_mask),np.size(mnv),label,vArea,hemi))
                superficial_mask = (z>=depth_groups['superficial'][0])
                middle_mask = (z>=depth_groups['middle'][0])*(z<depth_groups['middle'][1])
                deep_mask = (z<depth_groups['deep'][1])
                print("\t %d/%d Voxels Survive for superficial %s" %(np.sum(mnv_mask*superficial_mask),np.sum(superficial_mask),label))
                print("\t %d/%d Voxels Survive for middle %s" %(np.sum(mnv_mask*middle_mask),np.sum(middle_mask),label))
                print("\t %d/%d Voxels Survive for deep %s" %(np.sum(mnv_mask*deep_mask),np.sum(deep_mask),label))
                dropout['superficial'][k_i] = 1-np.sum(mnv_mask*superficial_mask)/np.sum(superficial_mask)
                dropout['middle'][k_i] = 1-np.sum(mnv_mask*middle_mask)/np.sum(middle_mask)
                dropout['deep'][k_i] = 1-np.sum(mnv_mask*deep_mask)/np.sum(deep_mask)
                dropout['total'][k_i] = 1-np.sum(mnv_mask)/np.size(mnv)
        
                k_i += 1
            
                # Add mnv_mask to the main DataFrame in all_data
                condition = (
                    (all_data[label]['Visual Region'] == vArea) & 
                    (all_data[label]['Subregion'] == 'tgt') & 
                    (all_data[label]['hemi'] == hemi)
                )
                all_data[label].loc[condition, 'no_vein'] = mnv_mask
    
    #Report dropout statistics
    print("Average total dropout rate: %s +/- %s" %(np.mean(dropout['total']),np.std(dropout['total'])))
    print("\t Superficial: %s +/- %s" %(np.mean(dropout['superficial']),np.std(dropout['superficial'])))
    print("\t Middle: %s +/- %s" %(np.mean(dropout['middle']),np.std(dropout['middle'])))
    print("\t Deep: %s +/- %s" %(np.mean(dropout['deep']),np.std(dropout['deep'])))
    
    # Save dropout statistics to CSV
    save_dropout_statistics_to_csv(vArea, all_subj, all_hemis, dropout, voxels_before, voxels_after, out_dir=os.path.join(figDir,'stats'))
        
    if savefigs:
        fthresh.savefig(os.path.join(figDir,'mnv_hist_%s.%s' %(vArea,fig_format)))
        dmap.savefig(os.path.join(figDir,'mnv_depth_map_%s.%s' %(vArea,fig_format)))
        dmap_thresh.savefig(os.path.join(figDir,'mnv_depth_map_thresh_%s.%s' %(vArea,fig_format)))
        fdepth_hist.savefig(os.path.join(figDir,'mnv_depth_hist_%s.%s' %(vArea,fig_format)))
        
    plt.close('all')
    
#%% Compare thresholds between subjects

#try violin plots
spreadF = 2
for iR, vArea in enumerate(['V1','V23']):
    k_i = 0
    subjIDs = [subj for subj in all_data.keys() if all_data[subj]['Visual Region'].str.contains(vArea).any()]
    Nsubj = len(subjIDs)
    f = plt.figure()
    dsets = [subj + h for subj in subjIDs for h in ['lh','rh']]
    
    for iS, label in enumerate(subjIDs):
        df_all = all_data[label]
        
        for iH, hemi in enumerate(df_all['hemi'].unique()):
            
            if np.sum((df_all['Visual Region'] == vArea) & (df_all['Subregion'] == 'tgt') & (df_all['hemi']==hemi)) != 0:
            
                # calculate log(MNV)
                df = df_all[(df_all['Visual Region'] == vArea) & (df_all['Subregion'] == 'tgt') & (df_all['hemi']==hemi)]
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
                plt.hlines(lmnv_dict[label]['thresh'],spreadF*k_i-0.5,spreadF*(k_i+0.2)+0.5,color='r')
                plt.title(vArea)
                
            k_i += 1
    plt.xticks(np.arange(0,2*spreadF*Nsubj,spreadF),dsets,rotation=15,fontsize=6)
    plt.ylabel("log(MNV)")
    
    if savefigs:
        f.savefig(os.path.join(figDir,'mnv_summary_violin_%s.%s' %(vArea,fig_format)))
        
#%% Apply full model p-val mask if desired

#initialize significance mask
for k_i, key in enumerate(all_data.keys()):
    all_data[key]['sig'] = (np.ones(len(all_data[key])) == 1)
    
#full model mask
fullmodel_thresh_df = pd.DataFrame({'subjID': [],' Nvox significant': [], 'Nvox total': [], 'Nvox_sig / Nvox_total': [], 'p_thresh': []})
if use_fullmodel_mask:
    for k_i, key in enumerate(all_data.keys()):
        df = all_data[key]
        pvals = df['task p-val']
        pval_mask = pvals < pthresh_fullmodel
        Nsig_pval = np.sum(pval_mask)
        print("%d/%d voxels survive full model p-val mask" %(Nsig_pval,np.size(pval_mask)))  
        all_data[key]['sig'] = (all_data[key]['sig'] & pval_mask)
        fullmodel_thresh_df.loc[len(fullmodel_thresh_df)] = [key, Nsig_pval, np.size(pval_mask), Nsig_pval/np.size(pval_mask), pthresh_fullmodel]
        
    #Compute summary statistics
    avgs = fullmodel_thresh_df.mean(axis=0,numeric_only=True)
    stds = fullmodel_thresh_df.std(axis=0,numeric_only=True)
    mins = fullmodel_thresh_df.min(axis=0,numeric_only=True)
    maxs = fullmodel_thresh_df.max(axis=0,numeric_only=True)
    fullmodel_thresh_stats = pd.DataFrame({'avg': avgs, 'std': stds, 'Nsubj': Nsubj*np.ones(len(avgs)), 'min': mins, 'max': maxs})
    
    #save to csv
    fullmodel_thresh_df.to_csv(os.path.join(figDir,'stats','task_pval_mask.csv'))
    fullmodel_thresh_stats.to_csv(os.path.join(figDir,'stats','task_pval_mask_summary.csv'))

#loc mask
loc_thresh_df = pd.DataFrame({'subjID': [],' Nvox significant': [], 'Nvox total': [], 'Nvox_sig / Nvox_total': [], 'p_thresh': []})
if use_loc_mask:
    for k_i, key in enumerate(all_data.keys()):
        df = all_data[key]
        pvals = df['loc p-val']
        pval_mask = pvals < pthresh_loc
        Nsig_pval = np.sum(pval_mask)
        print("%d/%d voxels survive loc p-val mask" %(Nsig_pval,np.size(pval_mask)))
        all_data[key]['sig'] = (all_data[key]['sig'] & pval_mask)
        loc_thresh_df.loc[len(loc_thresh_df)] = [key, Nsig_pval, np.size(pval_mask), Nsig_pval/np.size(pval_mask), pthresh_loc]
        
    #Compute summary statistics
    avgs = loc_thresh_df.mean(axis=0,numeric_only=True)
    stds = loc_thresh_df.std(axis=0,numeric_only=True)
    mins = loc_thresh_df.min(axis=0,numeric_only=True)
    maxs = loc_thresh_df.max(axis=0,numeric_only=True)
    loc_thresh_stats = pd.DataFrame({'avg': avgs, 'std': stds, 'Nsubj': Nsubj*np.ones(len(avgs)), 'min': mins, 'max': maxs})
    
    #save to csv
    loc_thresh_df.to_csv(os.path.join(figDir,'stats','task_pval_mask.csv'))
    loc_thresh_stats.to_csv(os.path.join(figDir,'stats','task_pval_mask_summary.csv'))
        
#%% Adding in a quick visualization to see how these thresholds affect our voxel
# count across depth and radial distance

# First let's check voxel count across depth bins for radial analysis
dbins = np.linspace(0, 1, nDepths_rings + 1)
for subj in all_data.keys():
    fig, ax = plt.subplots(nDepths_rings, 3, figsize=(10, 12))
    # Ensure ax has a shape of (nDepths_rings, 3) even if nDepths_rings = 1
    if nDepths_rings == 1:
        ax = np.atleast_2d(ax)
    fig.suptitle(f"{subj}")
    for d_i in range(nDepths_rings):

        # histogram
        ax_i = nDepths_rings - d_i - 1
        test_data = all_data[subj][
            (all_data[subj]['sig'] & all_data[subj]['no_vein'] & 
             (all_data[subj]['d_norm'] >= dbins[d_i]) & 
             (all_data[subj]['d_norm'] < dbins[d_i + 1]))
        ]['scale_xy_dist']
        max_dist = np.max(all_data[subj]['scale_xy_dist'])
        ax[ax_i, 0].hist(test_data, bins=np.arange(0, np.ceil(max_dist), 1))
        if d_i == 0:
            ax[ax_i, 0].set_xlabel("Radial Distance ($\sigma$)")
        ax[ax_i, 0].set_ylabel("Voxel Count")
        ax[ax_i, 0].set_title(f"Depth bin = {d_i}")
        ax[ax_i, 0].set_ylim([0, 500])
        ax[ax_i, 0].set_xlim([0, np.ceil(max_dist)])
        ax[ax_i, 0].plot([0, np.ceil(max_dist)], [10, 10], '--r', label='10 voxels')
        ax[ax_i, 0].legend()

        # surface plots
        for h_i, hemi in enumerate(['lh', 'rh']):
            df_V1 = all_data[subj][all_data[subj]['Visual Region'] == 'V1']
            if hemi in np.unique(df_V1['hemi']):
                # Extract ellipse parameters
                a = df_V1[
                    (df_V1['sig'] & df_V1['no_vein'] & 
                     (df_V1['d_norm'] >= dbins[d_i]) & 
                     (df_V1['d_norm'] < dbins[d_i + 1]) & 
                     (df_V1['hemi'] == hemi))
                ]['ellipse_a'].values[0]
                b = df_V1[
                    (df_V1['sig'] & df_V1['no_vein'] & 
                     (df_V1['d_norm'] >= dbins[d_i]) & 
                     (df_V1['d_norm'] < dbins[d_i + 1]) & 
                     (df_V1['hemi'] == hemi))
                ]['ellipse_b'].values[0]
                theta = df_V1[
                    (df_V1['sig'] & df_V1['no_vein'] & 
                     (df_V1['d_norm'] >= dbins[d_i]) & 
                     (df_V1['d_norm'] < dbins[d_i + 1]) & 
                     (df_V1['hemi'] == hemi))
                ]['ellipse_theta'].values[0]
                comX = df_V1[
                    (df_V1['sig'] & df_V1['no_vein'] & 
                     (df_V1['d_norm'] >= dbins[d_i]) & 
                     (df_V1['d_norm'] < dbins[d_i + 1]) & 
                     (df_V1['hemi'] == hemi))
                ]['ellipse_comX'].values[0]
                comY = df_V1[
                    (df_V1['sig'] & df_V1['no_vein'] & 
                     (df_V1['d_norm'] >= dbins[d_i]) & 
                     (df_V1['d_norm'] < dbins[d_i + 1]) & 
                     (df_V1['hemi'] == hemi))
                ]['ellipse_comY'].values[0]
                com = [comX, comY]

                # Scatter plot
                df = df_V1[
                    (df_V1['sig'] & df_V1['no_vein'] & 
                     (df_V1['d_norm'] >= dbins[d_i]) & 
                     (df_V1['d_norm'] < dbins[d_i + 1]) & 
                     (df_V1['hemi'] == hemi))
                ]
                pcm = ax[ax_i, h_i + 1].scatter(df['x'], df['y'], c=df['ctr-sur_unwarp'], s=4, cmap='plasma', vmin=-5, vmax=5)
                cbar = plt.colorbar(pcm, ax=ax[ax_i, h_i + 1])
                cbar.set_label('Ctr - Sur', fontsize=10)

                # Add multiple ellipses with varying scales
                max_dist = np.max(df['scale_xy_dist']) #reset max dist to be within hemisphere
                for scale in range(1, int(np.ceil(max_dist)) + 1):
                    alpha_value = 1.0 - (scale / np.ceil(max_dist))  # Decrease alpha with increasing scale
                    width = scale * 2 * np.sqrt(a)
                    height = scale * 2 * np.sqrt(b)
                    ellipse = Ellipse(
                        com,
                        width=scale * 2 * np.sqrt(a),
                        height=scale * 2 * np.sqrt(b),
                        angle=180 * theta / np.pi,
                        alpha=alpha_value,
                        edgecolor='r',
                        facecolor='none',
                        label=f'Scale {scale}' if scale == 1 else None  # Label only the first for legend
                    )
                    ax[ax_i, h_i + 1].add_patch(ellipse)

                    # Calculate text position along the minor axis of the ellipse
                    theta_rad = np.deg2rad(180 * theta / np.pi)
                    dx = (height / 2) * np.sin(theta_rad)
                    dy = (height / 2) * np.cos(theta_rad)
                    text_x = com[0] + dx
                    text_y = com[1] - dy

                    # Add text indicating the scale factor
                    ax[ax_i, h_i + 1].text(
                        text_x, text_y, f'{scale}$\sigma$',
                        color='darkred', fontsize=6, ha='center', va='center',
                        rotation=theta
                    )

                # Add labels
                ax[ax_i, h_i + 1].set_title(f"{hemi} Depth bin = {d_i}")
                ax[ax_i, h_i + 1].set_xlim([np.min(df['x'])-1,np.max(df['x'])+1])
                ax[ax_i, h_i + 1].set_ylim([np.min(df['y'])-1,np.max(df['y'])+1])
                ax[ax_i, h_i + 1].axis('off')
                ax[ax_i, h_i + 1].set_aspect('equal')

plt.tight_layout()


if savefigs:
    fig.savefig(os.path.join(figDir,'radial_distance_histograms_%s_V1.%s' %(subj,fig_format)))
        
#%% Get Depth Profiles
# Now we need to organize data into depth bins to compute depth profiles.

def return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis):
    # Given all the stats and diffs, return the relevant ones for a particular analysis
    if len(stat_analyses[analysis]) > 1:
        STATS = {label: statDetails[label][stat_analyses[analysis][0]:stat_analyses[analysis][1]] for label in statDetails.keys()}
        DIFFS = {
            'statIDs': {key: diffDetails['statIDs'][key] for key in diff_analyses[analysis]['list']},
            'colors': diffDetails['colors'][diff_analyses[analysis]['ids'][0]:diff_analyses[analysis]['ids'][1]]
        }
    else:
        STATS = {label: statDetails[label][stat_analyses[analysis][0]:] for label in statDetails.keys()}
        DIFFS = {
            'statIDs': {key: diffDetails['statIDs'][key] for key in diff_analyses[analysis]['list']},
            'colors': diffDetails['colors'][diff_analyses[analysis]['ids'][0]:]
        }
        
    return(STATS, DIFFS)

# Details about which conditions to create profiles for
statDetails = {'labels': ['sur', 'iso0', 'iso90', 'orth', 
                          'ctr_unwarp', 'sur_unwarp', 'ctr-sur_unwarp',
                          'V23_superficial_deveined_orth','V23_middle_deveined_orth', 'V23_deep_deveined_orth',
                          'V23_superficial_deveined_iso90', 'V23_middle_deveined_iso90', 'V23_deep_deveined_iso90',
                          'V23_superficial_deveined_iso0', 'V23_middle_deveined_iso0', 'V23_deep_deveined_iso0',
                          'V23_superficial_deveined_sur', 'V23_middle_deveined_sur', 'V23_deep_deveined_sur',
                          'V1_tgt_superficial_deveined_orth','V1_tgt_middle_deveined_orth', 'V1_tgt_deep_deveined_orth',
                          'V1_tgt_superficial_deveined_iso90', 'V1_tgt_middle_deveined_iso90', 'V1_tgt_deep_deveined_iso90',
                          'V1_tgt_superficial_deveined_iso0', 'V1_tgt_middle_deveined_iso0', 'V1_tgt_deep_deveined_iso0',
                          'V1_tgt_superficial_deveined_sur', 'V1_tgt_middle_deveined_sur', 'V1_tgt_deep_deveined_sur'
                          ],
               'colors': [[.7, .7, .7], 'red', 'darkviolet', 'orange',
                          'gold', 'purple', 'coral',
                          'orange', 'orange', 'orange', 
                          'darkviolet', 'darkviolet', 'darkviolet', 
                          'red', 'red', 'red', 
                          'gray', 'gray', 'gray',
                          'orange', 'orange', 'orange', 
                          'darkviolet', 'darkviolet', 'darkviolet', 
                          'red', 'red', 'red', 
                          'gray', 'gray', 'gray'
                          ]}

# Details about which condition contrasts to create profiles for
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
                          'iso-sur_gPPI_deep_V23':['V23_deep_deveined_iso0','V23_deep_deveined_sur'],
                          'odss_gPPI_superficial_V1':['V1_tgt_superficial_deveined_orth', 'V1_tgt_superficial_deveined_iso90'],
                          'odss_gPPI_middle_V1':['V1_tgt_middle_deveined_orth','V1_tgt_middle_deveined_iso90'],
                          'odss_gPPI_deep_V1':['V1_tgt_deep_deveined_orth','V1_tgt_deep_deveined_iso90'],
                          'fgm_gPPI_superficial_V1':['V1_tgt_superficial_deveined_iso90','V1_tgt_superficial_deveined_iso0'],
                          'fgm_gPPI_middle_V1':['V1_tgt_middle_deveined_iso90','V1_tgt_middle_deveined_iso0'],
                          'fgm_gPPI_deep_V1':['V1_tgt_deep_deveined_iso90','V1_tgt_deep_deveined_iso0'],
                          'iso-sur_gPPI_superficial_V1':['V1_tgt_superficial_deveined_iso0','V1_tgt_superficial_deveined_sur'],
                          'iso-sur_gPPI_middle_V1':['V1_tgt_middle_deveined_iso0','V1_tgt_middle_deveined_sur'],
                          'iso-sur_gPPI_deep_V1':['V1_tgt_deep_deveined_iso0','V1_tgt_deep_deveined_sur']
                          }
diffDetails['colors'] = ['green','magenta','cyan','black','coral','darkmagenta','darkturquoise',[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0],[1,0,1],[0.7,0,0.7],[0.5,0,0.5],[0.7,0.7,0.7],[0.5,0.5,0.5],[0.3,0.3,0.3],[0.1,1,0],[0.1,0.7,0],[0.1,0.5,0],[1,0,1],[0.7,0,0.7],[0.5,0,0.5],[0.7,0.7,0.7],[0.5,0.5,0.5],[0.3,0.3,0.3]]

# Analysis Stat Indices
stat_analyses = {'task': [0,4],
                 'loc': [4,7],
                 'gPPI': [7]}
diff_analyses = {'task': {'list': ['odss','fgm','dsi','iso-sur'], 'ids': [0,4]},
                 'loc': {'list': ['ctr-sur'], 'ids': [4,5]},
                 'gPPI': {'list': ['odss_gPPI_superficial_V23','odss_gPPI_middle_V23','odss_gPPI_deep_V23',
                          'fgm_gPPI_superficial_V23','fgm_gPPI_middle_V23','fgm_gPPI_deep_V23',
                          'iso-sur_gPPI_superficial_V23','iso-sur_gPPI_middle_V23','iso-sur_gPPI_deep_V23',
                          'odss_gPPI_superficial_V1','odss_gPPI_middle_V1','odss_gPPI_deep_V1',
                          'fgm_gPPI_superficial_V1','fgm_gPPI_middle_V1','fgm_gPPI_deep_V1',
                          'iso-sur_gPPI_superficial_V1','iso-sur_gPPI_middle_V1','iso-sur_gPPI_deep_V1'],
                          'ids': [7]}
                 }

# Create region masks within V1
profile_method = 'bin' # bin or smooth
#pick out ROIs where we're sure of localization
for key in all_data.keys():
    df = all_data[key][all_data[key]['Visual Region'] == 'V1']
    all_data[key].loc[all_data[key]['Visual Region'] == 'V1','in_tgt'] = df['scale_xy_dist'] < roiRad
    all_data[key].loc[all_data[key]['Visual Region'] == 'V1','in_ctr'] = df['scale_xy_dist'] < centerRad
    all_data[key].loc[all_data[key]['Visual Region'] == 'V1','in_bor'] = (df['scale_xy_dist'] >= borderRad[0]) & (df['scale_xy_dist'] < borderRad[1])
    if len(surRad) == 1:
        sur_mask = df['scale_xy_dist'] > surRad[0]
    elif len(surRad) > 1:
        sur_mask = ((df['scale_xy_dist'] >= surRad[0]) & (df['scale_xy_dist'] <= surRad[1]))
    all_data[key].loc[all_data[key]['Visual Region'] == 'V1','in_sur'] = sur_mask
    
# Create an extra index for the V23 ROIs, since these have no subregsion
for key in all_data.keys():
    all_data[key]['in_V23'] = all_data[key]['Visual Region'] == 'V23'
    
# Create full masks
masks = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}
for roi in masks.keys():
    masks[roi] = {key:all_data[key][roi]*all_data[key]['sig']*all_data[key]['no_vein'] for key in all_data.keys()}
    
# Now compute depth profiles
depthProfiles = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}
diffProfiles = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}
# For task condition analysis
for roi in depthProfiles.keys():
    if roi == 'in_tgt':
        nD = nDepths
        radialParam = 'scale_xy_dist'
    elif roi == 'in_V23':
        nD = nDepths_gPPI
        radialParam = None
    else:
        nD = nDepths_rings
        radialParam = 'scale_xy_dist'
    
    for analysis in stat_analyses.keys():
        
        # Only include subjects that should be included
        included_data = return_included_subj(subj_analyses, analysis)
        
        # Get the right stats for this analysis
        STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
            
        # Compute depth profiles
        depthProfiles[roi][analysis] = compute_all_depth_profiles(included_data,STATS,profile_method,nD,masks[roi],depthParam='d',radialParam=radialParam,spec_Drange='MinMax',statTestType='t-test')
        diffProfiles[roi][analysis] = compute_diff_profiles(included_data,STATS,DIFFS['statIDs'],profile_method,nD,useSI,masks[roi],depthParam='d',radialParam=radialParam,spec_Drange='MinMax',statTestType='t-test')

#%% Save number of voxels per depth

# Output directory for saving CSV files
# Loop through the data to extract and save the 'N' arrays
for roi, analyses in depthProfiles.items():
    for analysis, conditions in analyses.items():
        # Get the 'N' array from the first condition
        n_list = None
        for condition, stats in conditions.items():
            if 'N' in stats:
                n_list = stats['N']
                break

        # Proceed if 'N' is found
        if n_list is not None:
            # Convert list of arrays to a 2D array
            n_array = np.vstack(n_list)
            
            # Define the CSV file path
            csv_filename = f"{roi}_{analysis}_N.csv"
            csv_filepath = os.path.join(figDir, 'stats', csv_filename)

            # Calculate average and standard deviation across subjects
            avg_n = np.mean(n_array, axis=0)
            std_n = np.std(n_array, axis=0)

            # Save the array to a CSV file
            with open(csv_filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write header (depth labels from 0 to 1)
                depth_numbers = np.linspace(0, 1, n_array.shape[1])
                writer.writerow(["Subject"] + list(depth_numbers))
                # Write data for each subject
                subject_names = list(all_data.keys())  # Replace with your subject names list
                for idx, row in enumerate(n_array):
                    writer.writerow([subject_names[idx]] + list(row))
                # Write average and standard deviation rows
                writer.writerow(["Average"] + list(avg_n))
                writer.writerow(["Std Dev"] + list(std_n))

print(f"CSV files have been saved in '{os.path.join(figDir, 'stats')}' directory.")


#%% Deconvolution

# Only deconvolve for task and loc GLMs
for roi in depthProfiles.keys():
    
    if roi == 'in_tgt':
        nD = nDepths
    elif roi == 'in_V23':
        nD = nDepths_gPPI
    else:
        nD = nDepths_rings
        
    # Get included datasets for task and loc analyses
    for analysis in ['task','loc']:
        if roi != 'in_V23':
            included_data = return_included_subj(subj_analyses, analysis)
        else:
            included_data = return_included_subj(subj_analyses, 'gPPI') #exception for V23, because only subjects with gPPI analysis will have a V23 ROI
    
        # Get stat and diff dictionaries for these analyses
        STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
        
        dP = depthProfiles[roi][analysis]
        diffP = diffProfiles[roi][analysis]
        
        #reformat data to fit decon_rois specs
        keep_rois = np.zeros((len(included_data.keys()),len(STATS['labels']),nD))
        for iR, roiID in enumerate(included_data.keys()):
            for iStat, stat in enumerate(STATS['labels']):
                keep_rois[iR,iStat,:] = dP[stat]['avg'][iR]
                
        keep_diffs = np.zeros((len(included_data.keys()),len(DIFFS['statIDs'].keys()),nD))
        for iR, roiID in enumerate(included_data.keys()):
            for iDiff, diff in enumerate(DIFFS['statIDs'].keys()):
                keep_diffs[iR,iDiff,:] = diffP[diff]['avg'][iR]
        
        decon_rois = depth_deconv(keep_rois,p2t_model,Nbins_model,nD,normalize_psf)
        decon_diffs = depth_deconv(keep_diffs,p2t_model,Nbins_model,nD,normalize_psf)
        
        #now put back in dictionary
        for iStat, stat in enumerate(STATS['labels']):
            depthProfiles[roi][analysis][stat]['avg_decon'] = np.squeeze(np.array(decon_rois)[:,iStat,:])
            
        for iDiff, diff in enumerate(DIFFS['statIDs'].keys()):
            diffProfiles[roi][analysis][diff]['avg_decon'] = np.squeeze(np.array(decon_diffs)[:,iDiff,:])
            
#%% Compute average across subjects

avgDepthProfiles = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}
avgDepthDiffs = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}

for roi in depthProfiles.keys():
    if roi == 'in_tgt':
        nD = nDepths
        radialParam = 'scale_xy_dist'
    elif roi == 'in_V23':
        nD = nDepths
        radialParam = None
    else:
        nD = nDepths_rings
        radialParam = 'scale_xy_dist'
    
    for analysis in stat_analyses.keys():
        
        # Only include subjects that should be included
        included_data = return_included_subj(subj_analyses, analysis)
        
        # Get the right stats for this analysis
        STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
        
        # Note: permutation tests may take longer to compute than t-tests
        [avgDepthProfiles[roi][analysis], avgDepthDiffs[roi][analysis]] = compute_avg_depth_profile(depthProfiles[roi][analysis],STATS,DIFFS['statIDs'],STATS['labels'],list(DIFFS['statIDs'].keys()),use_decon[analysis],prop_err,useSI,statTestType=statTestType,num_permutations=Npermutations)  

#%% ~ Analysis ~
# Now that preprocessing is finished, we are ready to look at some data.

#%% Centroid plots
# Let's take a look at raw voxel betas across depth by condition

for analysis in ['task','loc']:

    # Get included subjects
    included_data = return_included_subj(subj_analyses, ['task','loc'])
    
    # Get stat and diff dictionaries for these analyses
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
        
    Nsubj = len(included_data)
    
    #plot centroids for each condition and ROI
    plot_centroids(included_data, masks['in_tgt'], STATS, roiRad, nDepths=nDepths)
            
    #calculate difference profiles
    plot_centroids_diff(included_data, masks['in_tgt'], STATS, DIFFS, roiRad, nDepths)
              
    if savefigs:
        for l in STATS['labels']:
            plt.figure(l)
            plt.savefig(os.path.join(figDir,'centroids_%s.%s' %(l,fig_format)))
        for l in DIFFS['statIDs'].keys():
            plt.figure(l)
            plt.savefig(os.path.join(figDir,'centroids_%s.%s' %(l,fig_format)))
        
#%% run the radial profiles analysis by binning the data across radial space
binSize = maxRad/nRad
radBins = np.linspace(0,maxRad,nRad+1)
radBinCtrs = radBins[:-1] + binSize/2

# Initialize dictionaries to hold rad profiles
radialProfiles = {analysis_type: {} for analysis_type in subj_analyses.keys()}
radialDiffProfiles = {analysis_type: {} for analysis_type in subj_analyses.keys()}
avgRadialProfiles = {analysis_type: {} for analysis_type in subj_analyses.keys()}
avgRadialDiff = {analysis_type: {} for analysis_type in subj_analyses.keys()}

# Iterate through analyses
for analysis in subj_analyses.keys():
    
    #Get included datasets
    included_data = return_included_subj(subj_analyses, analysis)
    
    # Get stat and diff dictionaries for these analyses
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)

    # layer masks
    layer_mask_dict = {l: {} for l in rad_depth_labels}
    for iR, d_label in enumerate(included_data.keys()):
        for iB, dB in enumerate(rad_depthBoundaries):
            df = all_data[d_label]
            lmask = (df['d'] > dB[0]) & (df['d'] < dB[1])
            roi_mask = (included_data[d_label]['no_vein']) & (included_data[d_label]['sig'])
            df[rad_depth_labels[iB]] = lmask
            layer_mask_dict[rad_depth_labels[iB]][d_label] = lmask & roi_mask
    
    # now run binned analysis for each layer
    for il, l in enumerate(rad_depth_labels):
        radialProfiles[analysis][l] = compute_all_rad_profiles(included_data, 
                                             STATS, 
                                             'bin', 
                                             nRad, 
                                             layer_mask_dict[l],
                                             radParam='scale_xy_dist',
                                             spec_Drange=[0,maxRad],
                                             radMax=maxRad)
        radialDiffProfiles[analysis][l] = compute_rad_diff_profiles(included_data,
                                             STATS,
                                             DIFFS['statIDs'],
                                             'bin',
                                             nRad,
                                             prop_err,
                                             layer_mask_dict[l],
                                             radParam='scale_xy_dist',
                                             spec_Drange=[0,maxRad],
                                             radMax=maxRad)
        # Note: permutation tests can take longer than t-tests
        avgRadialProfiles[analysis][l], avgRadialDiff[analysis][l] = compute_avg_rad_profile(radialProfiles[analysis][l],STATS,DIFFS['statIDs'],STATS['labels'],list(DIFFS['statIDs'].keys()),prop_err,useSI,statTestType=statTestType,npermSamples=Npermutations)
        
#%% Get null radial profiles

# Initialize dictionaries to hold rad profiles
radialProfiles_null = {analysis_type: {} for analysis_type in subj_analyses.keys()}
radialDiffProfiles_null = {analysis_type: {} for analysis_type in subj_analyses.keys()}
avgRadialProfiles_null = {analysis_type: {} for analysis_type in subj_analyses.keys()}
avgRadialDiff_null = {analysis_type: {} for analysis_type in subj_analyses.keys()}

# Iterate through analyses
for analysis in subj_analyses.keys():
    
    #Get included datasets
    included_data = return_included_subj(subj_analyses, analysis)
    
    # Get stat and diff dictionaries for these analyses
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)

    # layer masks
    layer_mask_dict = {l: {} for l in rad_depth_labels}
    included_data[d_label]['scale_xy_dist_scrambled'] = None #initialize a new column to hold scrambled data
    for iR, d_label in enumerate(included_data.keys()):
        for iB, dB in enumerate(rad_depthBoundaries):
            df = all_data[d_label]
            lmask = (df['d'] > dB[0]) & (df['d'] < dB[1])
            roi_mask = (included_data[d_label]['no_vein']) & (included_data[d_label]['sig'])
            df[rad_depth_labels[iB]] = lmask
            layer_mask_dict[rad_depth_labels[iB]][d_label] = lmask & roi_mask
    
            #scramble radius labels
            vArea_mask = included_data[d_label]['Visual Region'] == 'V1'
            included_data[d_label].loc[lmask & roi_mask & vArea_mask, 'scale_xy_dist_scrambled'] = np.random.permutation(included_data[d_label].loc[lmask & roi_mask & vArea_mask, 'scale_xy_dist'])
    
    # now run same binned analysis after scrambling radius labels
    for il, l in enumerate(rad_depth_labels):
        radialProfiles_null[analysis][l] = compute_all_rad_profiles(included_data, 
                                             STATS, 
                                             'bin', 
                                             nRad, 
                                             layer_mask_dict[l],
                                             radParam='scale_xy_dist_scrambled',
                                             spec_Drange=[0,maxRad],
                                             radMax=maxRad)
        radialDiffProfiles_null[analysis][l] = compute_rad_diff_profiles(included_data,
                                             STATS,
                                             DIFFS['statIDs'],
                                             'bin',
                                             nRad,
                                             prop_err,
                                             layer_mask_dict[l],
                                             radParam='scale_xy_dist_scrambled',
                                             spec_Drange=[0,maxRad],
                                             radMax=maxRad)
        # Note: permutation tests can take longer than t-tests
        avgRadialProfiles_null[analysis][l], avgRadialDiff_null[analysis][l] = compute_avg_rad_profile(radialProfiles_null[analysis][l],STATS,DIFFS['statIDs'],STATS['labels'],list(DIFFS['statIDs'].keys()),prop_err,useSI,statTestType=statTestType,npermSamples=Npermutations)
        
#%% Compare radial profiles to null
# Now that we have both the radial profiles and the null radial profiles, I can
# test them against one another.

# Assuming you have already created the avgRadialDiff dictionary
# and imported radialDiffProfiles and radialDiffProfiles_null

if compareRadtoNull:

    # Define the experiment, cortical depths, and condition contrasts of interest
    experiment = 'task'
    cortical_depths = ['deep', 'middle', 'superficial']
    condition_contrasts = ['odss', 'fgm']
    radial_bins_tested = [0,1,2,3,4]
    
    # Loop through each cortical depth and condition contrast
    for depth in cortical_depths:
        for condition in condition_contrasts:
            for b_i in range(nRad):
                if b_i in radial_bins_tested:
                    # Extract the 'avg' values for radialDiffProfiles and radialDiffProfiles_null
                    data_task = radialDiffProfiles[experiment][depth][condition]['avg']
                    null_task = radialDiffProfiles_null[experiment][depth][condition]['avg']
            
                    # Get the distributions for the smallest radius (index 0) for both data and null
                    data_distribution = [subject_data[b_i] for subject_data in data_task]
                    null_distribution = [subject_null[b_i] for subject_null in null_task]
            
                    # Perform the permutation test
                    def statistic(data,null,axis):
                        return np.mean(data, axis=axis) - np.mean(null, axis=axis)
                    
                    perm_result = stats.permutation_test(
                        (data_distribution,null_distribution),
                        statistic,
                        permutation_type='independent',
                        n_resamples=Npermutations,
                        alternative='two-sided',
                    )
            
                    p_value = perm_result.pvalue
            
                    # Save the p-value to avgRadialDiff
                    if 'p-vals vs null' not in avgRadialDiff[experiment][depth][condition]:
                        avgRadialDiff[experiment][depth][condition]['p-vals vs null'] = np.array([])
            
                    avgRadialDiff[experiment][depth][condition]['p-vals vs null'][b_i] = p_value
                else:
                    # Save the p-value to avgRadialDiff
                    if 'p-vals vs null' not in avgRadialDiff[experiment][depth][condition]:
                        avgRadialDiff[experiment][depth][condition]['p-vals vs null'] = np.array([])
            
                    avgRadialDiff[experiment][depth][condition]['p-vals vs null'][b_i] = np.nan
                
    
    print("Permutation tests completed and p-values saved.")


#%% Multisample comparisons
# There are some measurements that we want to compare against others
# statistically.

avgRadialDiff_comparisons = {analysis_type: {} for analysis_type in subj_analyses.keys()}

# Within condition comparisons across bins
contrasts = ['odss','fgm']
experiments = ['task']
for e_i, e in enumerate(experiments):
    avgRadialDiff_comparisons[e]['withinCondition'] = {}
    avgRadialDiff_comparisons[e]['withinCondition']['across_rad'] = {}
    for d_i, d in enumerate(rad_depth_labels):
        avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d] = {}
        for c_i, c in enumerate(contrasts):
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c] = {}
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['p-vals'] = np.empty((nRad,nRad))
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['p-vals'][:] = np.nan
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['t-stat'] = np.empty((nRad,nRad))
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['t-stat'][:] = np.nan
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['df'] = np.empty((nRad,nRad))
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['df'][:] = np.nan
            for rb_i, rb in enumerate(radBin_comparisons):
                if statTestType == 't-test':
                    tstat, pval = stats.ttest_rel(np.array(radialDiffProfiles[e][d][c]['avg'])[:,rb[0]],np.array(radialDiffProfiles[e][d][c]['avg'])[:,rb[1]])
                elif statTestType == 'permutation':
                    diffs = np.array(radialDiffProfiles[e][d][c]['avg'])[:,rb[0]] - np.array(radialDiffProfiles[e][d][c]['avg'])[:,rb[1]]
                    pval = permute_1samp(diffs, np.mean, n_permutations=Npermutations)
                    tstat = np.nan
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['p-vals'][rb[0],rb[1]] = pval
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['p-vals'][rb[1],rb[0]] = pval
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['t-stat'][rb[0],rb[1]] = tstat
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['t-stat'][rb[1],rb[0]] = tstat
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['df'][rb[0],rb[1]] = np.shape(radialDiffProfiles[e][d][c]['avg'])[0] - 1
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['df'][rb[1],rb[0]] = np.shape(radialDiffProfiles[e][d][c]['avg'])[0] - 1
        
#%% Corrections for Multiple Comparisons

# Example exclusion settings
exclude_indiv_ttest = True #if True, only include across-subject statistical tests and not within subject statistical tests
exclude_analysis_types = ['gPPI']
exclude_rois = []
exclude_cond = ['iso0','iso90','orth','sur', #exclude individual condition t-tests, only focus on condition contrasts
                'ctr_unwarp','sur_unwarp', 'ctr-sur_unwarp',
                'V23_superficial_deveined_orth','V23_middle_deveined_orth', 'V23_deep_deveined_orth',
                'V23_superficial_deveined_iso90', 'V23_middle_deveined_iso90', 'V23_deep_deveined_iso90',
                'V23_superficial_deveined_iso0', 'V23_middle_deveined_iso0', 'V23_deep_deveined_iso0',
                'V23_superficial_deveined_sur', 'V23_middle_deveined_sur', 'V23_deep_deveined_sur',
                'V1_tgt_superficial_deveined_orth','V1_tgt_middle_deveined_orth', 'V1_tgt_deep_deveined_orth',
                'V1_tgt_superficial_deveined_iso90', 'V1_tgt_middle_deveined_iso90', 'V1_tgt_deep_deveined_iso90',
                'V1_tgt_superficial_deveined_iso0', 'V1_tgt_middle_deveined_iso0', 'V1_tgt_deep_deveined_iso0',
                'V1_tgt_superficial_deveined_sur', 'V1_tgt_middle_deveined_sur', 'V1_tgt_deep_deveined_sur',
                'iso-sur_gPPI_superficial_V23','iso-sur_gPPI_middle_V23','iso-sur_gPPI_deep_V23',
                'iso-sur_gPPI_superficial_V1','iso-sur_gPPI_middle_V1','iso-sur_gPPI_deep_V1'
                'dsi'
                ]
exclude_combinations = [('in_bor', 'gPPI', 'all'), 
                        ('in_sur', 'gPPI', 'all'),
                        ('in_ctr', 'loc', 'all'),
                        ('in_bor', 'loc', 'all'),
                        ('in_sur', 'loc', 'all')]

def gather_pvals(dictionary, keys_path, all_pvals, path_to_pvals, exclude_analysis_types, exclude_rois, exclude_cond, exclude_combinations):
    for key, value in dictionary.items():
        if key in exclude_analysis_types or key in exclude_rois or key in exclude_cond:
            continue
        
        new_keys_path = keys_path + [key]
        if isinstance(value, dict):
            gather_pvals(value, new_keys_path, all_pvals, path_to_pvals, exclude_analysis_types, exclude_rois, exclude_cond, exclude_combinations)
        elif 'p-vals'  == key:
            # Check if the current combination should be excluded
            exclude = False
            for combination in exclude_combinations:
                match = True
                for i, exclude_key in enumerate(combination):
                    if exclude_key != 'all' and (i >= len(new_keys_path) or new_keys_path[i] != exclude_key):
                        match = False
                        break
                if match:
                    exclude = True
                    break
            
            if exclude:
                continue
            
            p_values = value
            if hasattr(p_values, 'pvalue'):
                p_values = p_values.pvalue
            if isinstance(p_values, np.ndarray):
                all_pvals.append(p_values.flatten())
                path_to_pvals.append((dictionary, key))

# Gather all p-values from all six dictionaries
all_pvals = []
path_to_pvals = []  # Keep track of where p-values are located for updating later
if exclude_indiv_ttest:
    dictionaries = [avgDepthProfiles, avgDepthDiffs, avgRadialProfiles, avgRadialDiff]
else:
    dictionaries = [depthProfiles, diffProfiles, avgDepthProfiles, avgDepthDiffs, avgRadialProfiles, avgRadialDiff]
for dictionary in dictionaries:
    gather_pvals(dictionary, [], all_pvals, path_to_pvals, exclude_analysis_types, exclude_rois, exclude_cond, exclude_combinations)

# Add non-NaN comparisons from avgRadialDiff_comparisons
def gather_avgRadialDiff_comparisons(dictionary, all_p_values, path_to_pvals):
    """
    Gathers p-value matrices (including NaNs) from avgRadialDiff_comparisons.
    
    Parameters:
        dictionary (dict): Dictionary containing the avgRadialDiff_comparisons data
        all_p_values (list): List to store the gathered p-values
        path_to_pvals (list): List to store the path for updating later
    """
    for key, value in dictionary.items():
        if isinstance(value, dict) and 'p-vals' not in value:
            gather_avgRadialDiff_comparisons(value, all_p_values, path_to_pvals)
        elif 'p-vals' in value:
            p_values = value['p-vals']
            if isinstance(p_values, np.ndarray):
                all_p_values.append(p_values.flatten())
                path_to_pvals.append((dictionary[key], 'p-vals'))

# Gather p-values from avgRadialDiff_comparisons
gather_avgRadialDiff_comparisons(avgRadialDiff_comparisons, all_pvals, path_to_pvals)

# Correct for multiple comparisons
flattened_p_values = np.concatenate(all_pvals)

# Identify NaNs and exclude them before correction
nan_mask = np.isnan(flattened_p_values)
non_nan_p_values = flattened_p_values[~nan_mask]

# Perform the multiple comparisons correction only on non-NaN values
_, corrected_non_nan_p_values, _, _ = multipletests(non_nan_p_values, alpha=0.05, method=statCorrType)

# Create an array to store the corrected p-values, with NaNs where they were originally
corrected_p_values = np.full_like(flattened_p_values, np.nan)
corrected_p_values[~nan_mask] = corrected_non_nan_p_values

# Update the dictionaries with corrected p-values
idx = 0
for dictionary, key in path_to_pvals:
    p_values = dictionary['p-vals']
    if hasattr(p_values, 'pvalue'):
        dictionary['corrected p-vals'] = corrected_p_values[idx:idx + len(p_values.pvalue)].reshape(p_values.pvalue.shape)
        idx += len(p_values.pvalue)
    elif isinstance(p_values, np.ndarray):
        corrected_subset = corrected_p_values[idx:idx + p_values.size].reshape(p_values.shape)
        idx += p_values.size

        # Only update non-NaN values
        valid_mask = ~np.isnan(p_values)
        dictionary['corrected p-vals'] = np.where(valid_mask, corrected_subset, np.nan)

        
#%% ~ Plots ~
# Now make plots based on the previous analysis

#%% V1 Depth Profiles

roi_type = 'in_tgt' #the type of ROI to examine

def save_as_df(profiles,roi_type,analysis_type):
    # Save Task Profiles to CSV
    for diff, data in profiles[roi_type][analysis_type].items():
        depth_bins = np.arange(len(data['avg']))  # Create depth bins
        if 'corrected p-vals' in data.keys():
            if statTestType == 't-test':
                df = pd.DataFrame({
                    'depth bin': depth_bins,
                    'avg': data['avg'],
                    'stdev': data['stdev'],
                    'norm_depths': data['norm_depths'],
                    't-statistic': data['p-vals'].statistic,
                    'p-value': data['p-vals'].pvalue,
                    'corrected p-value': data['corrected p-vals'],
                    'df': data['p-vals'].df,
                    'N': [data['Nsamp']] * len(data['avg'])  # Repeat the Nsamp value
                })
                df.to_csv(os.path.join(figDir,f"{diff}_{roi_type}.csv"), index=False)
            elif statTestType == 'permutation':
                df = pd.DataFrame({
                    'depth bin': depth_bins,
                    'avg': data['avg'],
                    'stdev': data['stdev'],
                    'norm_depths': data['norm_depths'],
                    'p-value': data['p-vals'],
                    'corrected p-value': data['corrected p-vals'],
                    'df': data['Nsamp']-1,
                    'N': [data['Nsamp']] * len(data['avg'])  # Repeat the Nsamp value
                })
                df.to_csv(os.path.join(figDir,f"{diff}_{roi_type}.csv"), index=False)
#Save task
save_as_df(avgDepthDiffs,roi_type,'task')
#Save loc
save_as_df(avgDepthDiffs,roi_type,'task')

# Plot average profiles
cm = 1/2.54 # inch/centimeter
for analysis in ['task','loc']:
    fig = plt.figure(figsize=(6*cm, 4*cm))
    fig.set_size_inches((6*cm,4*cm))
    fig.patch.set_facecolor(fcolor)
        
    fig.clf()
    fsize = 7
        
    p1 = fig.add_axes([.22, .27, .3, .7])
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
    # Get the right stats for this analysis
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
    
    xticks=[0,2,4,6] #set xticks
    plot_avg_depth_profile(p1,avgDepthProfiles[roi_type][analysis],STATS['labels'],STATS['colors'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,xticks=xticks)
    
    p2 = fig.add_axes([.7, .27, .25, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    xlim = [-0.3,1.8]
    plot_avg_diff_profile(p2,avgDepthDiffs[roi_type][analysis],DIFFS['statIDs'].keys(),DIFFS['colors'],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI)
    
    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir,'avg_profiles_%s_%s_deconv.%s' %(analysis,roi_type,fig_format)))
        else:
            fig.savefig(os.path.join(figDir,'avg_profiles_%s_%s.%s' %(analysis,roi_type,fig_format)))
            
#%% Plot context modulation effect separately

def save_statistical_results(data_dict, alpha, statTestType, Npermutations = None, output_csv='output.csv', binType = 'norm_depths'):
    
    # Extract inputs from dictionary
    avg = data_dict.get('avg')
    sd = data_dict.get('stdev')
    norm_bins = data_dict.get(binType)
    Nsamp = data_dict.get('Nsamp')
    p_vals = data_dict.get('corrected p-vals')
    
    # Create a dictionary to construct the DataFrame
    data = {
        binType: norm_bins,
        'avg': avg,
        'sd': sd,
        'alpha': [alpha] * len(avg),
    }

    # Handle t-test or permutation test separately
    if statTestType == 't-test' and isinstance(p_vals, ttest_ind.__class__):
        data['df'] = p_vals.df
        data['test statistic'] = p_vals.statistic
        data['p-vals'] = p_vals.pvalue
    elif statTestType == 'permutation' and isinstance(p_vals, np.ndarray):
        data['df'] = [Nsamp] * len(avg)
        data['test statistic'] = avg
        data['p-vals'] = p_vals
        # Cap number of permutations
        Npermutations_array = Npermutations * np.ones(len(avg))
        Npermutations_array[Npermutations_array > 2**Nsamp] = 2**Nsamp
        data['Npermutations'] = Npermutations_array
    else:
        raise ValueError("Invalid input for 'statTestType' or 'corrected p-vals'")
    
    # Convert data to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

fig_size = "large" # "small" or "large"
for roi_type in ['in_tgt','in_V23']:
    for analysis in ['task','loc']:
        # Get the right stats for this analysis
        STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
        for iDiff, diff in enumerate(DIFFS['statIDs'].keys()):
            
            if fig_size == "small":
                fig = plt.figure(figsize=(6*cm, 4*cm))
                fig.set_size_inches((6*cm,4*cm))
            elif fig_size == "large":
                fig = plt.figure(figsize=(6, 4))
                fig.set_size_inches((6, 4))
            else:
                raise ValueError("fig_size must be 'small' or 'large'")
            fig.patch.set_facecolor(fcolor)
            
            fig.clf()
            fsize = 7
            
            p1 = fig.add_axes([.22, .27, .3, .7])
            fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
            
            ylim = [-0.02,1.02]
            xlim = [-1,6]
            Ntext = [4,0.05]
            
            #Get stat labels
            stat_labels = DIFFS['statIDs'][diff]
            stat_labels_i = [STATS['labels'].index(item) for item in stat_labels]
            stat_colors = [STATS['colors'][i] for i in stat_labels_i]
            
            xticks=[0,2,4,6] #set xticks
            plot_avg_depth_profile(p1,avgDepthProfiles[roi_type][analysis],stat_labels,stat_colors,ylim,xlim,dx,dy,Ntext,lcolor,fsize,xticks=xticks)
            
            #diffs
            xlim = [-0.5,1.5]
            p2 = fig.add_axes([.7, .27, .25, .7])
            fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
            if 'corrected p-vals' in avgDepthDiffs[roi_type][analysis][diff].keys():
                plot_avg_diff_profile(p2,avgDepthDiffs[roi_type][analysis],[diff],[DIFFS['colors'][iDiff]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=True,pthresh=pthresh,statCorrType=avgDepthDiffs[roi_type][analysis][diff]['corrected p-vals'])
            else:
                plot_avg_diff_profile(p2,avgDepthDiffs,[diff],[DIFFS['colors'][iDiff]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,showSig=False)
            if savefigs:
                if use_decon:
                    fig.savefig(os.path.join(figDir,'avg_profiles_%s_%s_%s_deconv.%s' %(analysis,roi_type,diff,fig_format)))
                    if 'corrected p-vals' in avgDepthDiffs[roi_type][analysis][diff].keys():
                        save_statistical_results(avgDepthDiffs[roi_type][analysis][diff], pthresh, statTestType, Npermutations=Npermutations, output_csv = os.path.join(figDir,'stats','avg_profiles_%s_%s_%s_%s_deconv.csv' %(analysis,roi_type,diff,statTestType)))
                else:
                    fig.savefig(os.path.join(figDir,'avg_profiles_%s_%s_%s.%s' %(analysis,roi_type,diff,fig_format)))
                    if 'corrected p-vals' in avgDepthDiffs[roi_type][analysis][diff].keys():
                        save_statistical_results(avgDepthDiffs[roi_type][analysis][diff], pthresh, statTestType, Npermutations=Npermutations, output_csv = os.path.join(figDir,'stats','avg_profiles_%s_%s_%s_%s.csv' %(analysis,roi_type,diff,statTestType)))

#%% Make condition depth profiles with individual subject data overlaid

fig_size = "small" # "small" or "large"
for roi_type in ['in_tgt','in_V23']:
    for analysis in ['task','loc']:
        # Get the right stats for this analysis
        STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
        for iStat, stat in enumerate(STATS['labels']):

            if fig_size == "small":
                fig = plt.figure(figsize=(6*cm, 4*cm))
                fig.set_size_inches((6*cm,4*cm))
            elif fig_size == "large":
                fig = plt.figure(figsize=(6, 4))
                fig.set_size_inches((6, 4))
            else:
                raise ValueError("fig_size must be 'small' or 'large'")
            fig.patch.set_facecolor(fcolor)
            
            fig.clf()
            fsize = 7
            
            p1 = fig.add_axes([.2, .25, .3, .7])
            fix_axes(p1, lcolor=lcolor, fcolor=fcolor)
                
            ylim = [-0.02,1.02]
            if analysis == 'task':
                xlim = [0,7]
            elif analysis == 'loc':
                xlim = [-3,7]
            Ntext = [4,0.05]
            
            #Do some housekeeping on the data arrays. If list, convert to numpy array
            if use_decon[analysis]:
                if type(depthProfiles[roi_type][analysis][stat]['avg_decon']) == list:
                    depthProfiles[roi_type][analysis][stat]['avg_decon'] = np.vstack(depthProfiles[roi_type][analysis][stat]['avg_decon'])
            else:
                if type(depthProfiles[roi_type][analysis][stat]['avg']) == list:
                    depthProfiles[roi_type][analysis][stat]['avg'] = np.vstack(depthProfiles[roi_type][analysis][stat]['avg'])
            
            xticks=[0,2,4,6] #set xticks
            plot_avg_depth_profile(p1,avgDepthProfiles[roi_type][analysis],[stat],[STATS['colors'][iStat]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,plot_indiv=depthProfiles[roi_type][analysis],use_decon=use_decon[analysis],xticks=xticks)
            
            if savefigs:
                if use_decon[analysis]:
                    fig.savefig(os.path.join(figDir,'avg_profiles_%s_%s_%s_deconv.%s' %(analysis,roi_type,stat,fig_format)))
                else:
                    fig.savefig(os.path.join(figDir,'avg_profiles_%s_%s_%s.%s' %(analysis,roi_type,stat,fig_format)))
                
#%% Make difference depth profiles with individual subject data overlaid

for roi_type in ['in_tgt','in_V23']:
    for analysis in ['task','loc']:
        # Get the right stats for this analysis
        STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
        for iDiff, diff in enumerate(DIFFS['statIDs'].keys()):

            if fig_size == "small":
                fig = plt.figure(figsize=(6*cm, 4*cm))
                fig.set_size_inches((6*cm,4*cm))
            elif fig_size == "large":
                fig = plt.figure(figsize=(6, 4))
                fig.set_size_inches((6,4))
            fig.patch.set_facecolor(fcolor)
            
            fig.clf()
            fsize = 7
            
            p2 = fig.add_axes([.7, .25, .25, .7])
            fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
                
            ylim = [-0.02,1.02]
            if analysis == 'task':
                xlim = [-0.6,1.8]
            elif analysis == 'loc':
                xlim = [-1.5,2]
            Ntext = [4,0.05]
            plot_avg_diff_profile(p2,avgDepthDiffs[roi_type][analysis],[diff],[DIFFS['colors'][iDiff]],ylim,xlim,dx,dy,Ntext,lcolor,fsize,useSI,plot_indiv=diffProfiles[roi_type][analysis],showSig=True,pthresh=pthresh,statCorrType=avgDepthDiffs[roi_type][analysis][diff]['corrected p-vals'],use_decon=use_decon[analysis])
            
            if savefigs:
                if use_decon[analysis]:
                    fig.savefig(os.path.join(figDir,'avg_diffs_%s_%s_%s_deconv.%s' %(analysis,roi_type,diff,fig_format)))
                else:
                    fig.savefig(os.path.join(figDir,'avg_diffs_%s_%s_%s.%s' %(analysis,roi_type,diff,fig_format)))
                
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

def plot_smoothed_radial_profile(data, analysis_type, condition, kernel, mask=None, smooth_factor=0.3, radMax=4, nRadii=20, ymin=-5, ymax=5, fontsize=8, fcolor='white', lcolor='black', depth_labels = ['deep', 'middle', 'superficial'], depthBoundaries = np.array([[0, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 1]]), plot_indiv=True, statColor='gray',vline=2):
    """
    Plots smoothed radial profiles for each subject and the average across subjects for a given condition or condition contrast.
    
    Parameters:
        data (dict): Dataset containing radial profiles for all subjects
        analysis_type (str): Type of analysis to plot
        condition (str): Condition name
        kernel (func): smoothing kernel to use
        smooth_factor (float): Smoothing factor for the exponential kernel
        radMax (float): Maximum radius for plotting
        nRadii (int): Number of radii points to plot
        ymin (float): Minimum y-axis limit
        ymax (float): Maximum y-axis limit
        fontsize (float): Find size for axes labels
        fcolor (str): Background color for the plot
        lcolor (str): Line color for the plot
        depth_labels (list): Labels for depth bins
        depthBoundaries (list): Boundaries for depth bins
        plot_indiv (bool): If true, overlay individual subject profiles
        statColor (str): Color of plot lines
        vline (float): x-location of a vertical line
    """
    
    all_profiles = {depth_label: {} for depth_label in depth_labels}

    for iR, label in enumerate(data.keys()):
        df = data[label]
        if mask:
            df = df[mask[label]]
        for iD, depth_label in enumerate(depth_labels):
            depth_df = df[(df['d'] >= depthBoundaries[iD, 0]) & (df['d'] < depthBoundaries[iD, 1])]
            coef = depth_df[condition].values
            x = depth_df['scale_xy_dist'].values
            
            coef_smooth, x_smooth = smoothen(coef, x, kernel=kernel, smooth_factor=smooth_factor, radMax=maxRad)
            all_profiles[depth_label][label] = coef_smooth
    
    # Plot average profile across subjects
    fig = plt.figure(figsize=(5, 6))
    fig.patch.set_facecolor(fcolor)
    for iD, depth_label in enumerate(depth_labels):
        p = fig.add_axes([.15, .1 + iD * .3, .7, .2])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        layer_profiles_list = list(all_profiles[depth_label].values())
        stat_avg = np.mean(np.vstack(layer_profiles_list), axis=0)
        stat_std = np.std(np.vstack(layer_profiles_list), axis=0)

        p.plot(x_smooth, stat_avg, color=statColor, label=depth_label)
        p.fill_between(x_smooth,
                       stat_avg - stat_std / np.sqrt(len(all_profiles[depth_label])),
                       stat_avg + stat_std / np.sqrt(len(all_profiles[depth_label])),
                       alpha=0.4,
                       color=statColor)

        p.set_ylim([ymin, ymax])
        p.set_ylabel('BOLD % change', fontsize=fontsize, color=lcolor)
        p.set_title(depth_label)
        if plot_indiv:
            for iR, label in enumerate(data.keys()):
                p.plot(x_smooth.T, np.array(all_profiles[depth_label][label]).T, color=statColor, alpha=0.2)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fontsize, color=lcolor)
        p.plot([vline, vline], [-6, 6], '--', color='black')
        p.plot([0, radMax], [0, 0], '--', color='black')
        p.legend([stat])
            
    return(fig)

# Plots for each condition

#with smoothing
plot_orig_data = False #plot original data?
plot_Fstat = False# True #plot fstat?

kernel = smooth_kernel
#smooth_factor = 0.4/(2*np.sqrt(2*np.log(2))) #This is the st. dev. for a FWHM = 0.4 sigma which is about 0.5 mm for an ROI with a diameter of 5 mm
smooth_factor = 0.3 #This is the smooth factor for the exponential kernel. It roughly corresponds to a 1 mm FWHM on the cortical surface for an ROI with a diameter of 5 mm.
nRadii = 20
ymax = 7
ymin = -7
highlight = False #['ctr-sur'] #['orth', 'iso90']#'orth'
# depth_labels = ['deep', 'middle', 'superficial']
# depthBoundaries = np.array([[0,1/3],[1/3,2/3],[2/3,1]])

# Loop through analyses
for analysis in ['task','loc', 'gPPI']:
    # Get the right stats for this analysis
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
    for s_i, stat in enumerate(STATS['labels']):

        # Only include subjects that should be included
        included_data = return_included_subj(subj_analyses, analysis)
        
        masks = {label: included_data[label]['no_vein'] & included_data[label]['sig'] & (included_data[label]['Visual Region'] == 'V1') for label in included_data.keys()}
        fig = plot_smoothed_radial_profile(included_data,analysis,stat,kernel,mask=masks,vline=roiRad,radMax=maxRad,ymin=ymin,ymax=ymax,statColor=STATS['colors'][s_i],depth_labels = rad_depth_labels, depthBoundaries=rad_depthBoundaries)
        
        if savefigs:
            fig.savefig(os.path.join(figDir, f"radial_profiles_{analysis}_{stat}.{fig_format}"))
            
#%% Check Individual Subject Profiles

def plot_radial_profiles_subplot(data, analysis_type, condition, kernel, mask=None, smooth_factor=0.3, radMax=4, nRadii=20, ymin=-5, ymax=5, fontsize=8, fcolor='white', lcolor='black', depth_labels=['deep', 'middle', 'superficial'], depthBoundaries=np.array([[0, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 1]]), statColor='gray', vline=2):
    """
    Plots radial profiles for each subject in separate subplots organized by depth and subject.

    Parameters:
        data (dict): Dataset containing radial profiles for all subjects
        analysis_type (str): Type of analysis to plot
        condition (list or str): Condition name(s)
        kernel (func): smoothing kernel to use
        smooth_factor (float): Smoothing factor for the exponential kernel
        radMax (float): Maximum radius for plotting
        nRadii (int): Number of radii points to plot
        ymin (float): Minimum y-axis limit
        ymax (float): Maximum y-axis limit
        fontsize (float): Font size for axes labels
        fcolor (str): Background color for the plot
        lcolor (str): Line color for the plot
        depth_labels (list): Labels for depth bins
        depthBoundaries (list): Boundaries for depth bins
        statColor (str): Color of plot lines
        vline (float): x-location of a vertical line
    """

    # Ensure condition is a list
    if isinstance(condition, str):
        condition = [condition]
    if isinstance(statColor, str):
        statColor = [statColor]

    # Initialize all_profiles dictionary for each depth label, subject, and condition
    all_profiles = {depth_label: {cond: {} for cond in condition} for depth_label in depth_labels}

    for iR, label in enumerate(data.keys()):
        df = data[label]
        if mask:
            df = df[mask[label]]
        for iD, depth_label in enumerate(depth_labels):
            depth_df = df[(df['d'] >= depthBoundaries[iD, 0]) & (df['d'] < depthBoundaries[iD, 1])]
            x = depth_df['scale_xy_dist'].values
            for cond in condition:
                coef = depth_df[cond].values
                coef_smooth, x_smooth = smoothen(coef, x, kernel=kernel, smooth_factor=smooth_factor, radMax=maxRad)
                all_profiles[depth_label][cond][label] = coef_smooth

    # Create figure with subplots
    n_rows = len(depth_labels)
    n_cols = len(data.keys())
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    fig.patch.set_facecolor(fcolor)

    # Plot each profile in separate subplots
    for iD, depth_label in enumerate(depth_labels):
        for iR, label in enumerate(data.keys()):
            ax = axs[iD, iR] if n_rows > 1 and n_cols > 1 else axs[max(iD, iR)]
            fix_axes(ax, lcolor=lcolor, fcolor=fcolor)

            for iC, cond in enumerate(condition):
                stat_avg = all_profiles[depth_label][cond][label]
                ax.plot(x_smooth, stat_avg, label=f'{cond}', alpha=0.7, color=statColor[iC])

            ax.set_ylim([ymin, ymax])
            ax.set_title(f'{depth_label} - {label}', fontsize=fontsize, color=lcolor)

            # Add vertical and horizontal reference lines
            ax.plot([vline, vline], [ymin, ymax], '--', color='black')
            ax.plot([0, 4], [0, 0], '--', color='black')

            if iD == n_rows - 1:
                ax.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fontsize, color=lcolor)
            if iR == 0:
                ax.set_ylabel('BOLD % change', fontsize=fontsize, color=lcolor)

            ax.legend(fontsize=fontsize - 2)

    plt.tight_layout()
    return fig

for analysis in ['task','loc','gPPI']:

    # Only include subjects that should be included
    included_data = return_included_subj(subj_analyses, analysis)
    
    # Get the right stats for this analysis
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
    
    stat = STATS['labels']
    
    if isinstance(stat, str):
        stat_id = STATS['labels'].index(stat)
        stat_color = STATS['color'][stat_id]
    else:
        stat_id = []
        stat_color = []
        for s in stat:
            stat_id.append(STATS['labels'].index(s))
            stat_color.append(STATS['colors'][stat_id[-1]])
    
    masks = {label: included_data[label]['no_vein'] & included_data[label]['sig'] & (included_data[label]['Visual Region'] == 'V1') for label in included_data.keys()}
    fig = plot_radial_profiles_subplot(included_data,analysis,stat,kernel,mask=masks,ymin=-2,ymax=10,statColor=stat_color,vline=roiRad,depth_labels=rad_depth_labels,depthBoundaries=rad_depthBoundaries)
    
    if savefigs:
        fig.savefig(os.path.join(figDir,"radial_profiles_%s.%s" %(analysis,fig_format)))

#%% Compute differences and put them back in all_data
for analysis in ['task','loc','gPPI']:
    # Only include subjects that should be included
    included_data = return_included_subj(subj_analyses, analysis)

    # Get the right stats for this analysis
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)

    for diff in DIFFS['statIDs'].keys():
        for iR, label in enumerate(included_data.keys()):
            stat1 = DIFFS['statIDs'][diff][0]
            stat2 = DIFFS['statIDs'][diff][1]
            included_data[label][diff] = included_data[label][stat1] - included_data[label][stat2]

#%% Diff Radial Profiles

for analysis in ['task','loc','gPPI']:
    
    # Only include subjects that should be included
    included_data = return_included_subj(subj_analyses, analysis)
    
    # Get the right stats for this analysis
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
    
    diff = DIFFS['statIDs']
    
    if isinstance(diff, str):
        diff_id = list(DIFFS['statIDs'].keys()).index(diff)
        diff_color = DIFFS['color'][diff_id]
    else:
        diff_id = []
        diff_color = []
        for d in diff:
            diff_id.append(list(DIFFS['statIDs'].keys()).index(d))
            diff_color.append(DIFFS['colors'][diff_id[-1]])
            
    masks = {label: included_data[label]['no_vein'] & included_data[label]['sig'] & (included_data[label]['Visual Region'] == 'V1') for label in included_data.keys()}
    fig = plot_radial_profiles_subplot(included_data,analysis,diff,kernel,mask=masks,ymin=-2,ymax=2,statColor=diff_color,vline=roiRad,depth_labels=rad_depth_labels,depthBoundaries=rad_depthBoundaries)
    
    if savefigs:
        fig.savefig(os.path.join(figDir,"radial_diff_profiles_%s.%s" %(analysis,fig_format)))
        
#%% Add Binned Radial Data

def plot_smoothed_radial_profile_wbins(data, avgRadialProfiles, analysis_type, condition, kernel, mask=None, smooth_factor=0.3, radMax=4, nRadii=20, ymin=-5, ymax=5, ymin_bar=-0.5, ymax_bar=3.0, fontsize=8, fcolor='white', lcolor='black', depth_labels = ['deep', 'middle', 'superficial'], depthBoundaries = np.array([[0, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 1]]), plot_indiv=True, statColor='gray',vline=2, pval_threshold=0.05, nRad=4,comparisons=None,figsize=(8,10),ax_width=0.7,ax_height=0.15,ax_height_bar=0.06,ax_spacing=0.3,ax_left=0.15,ax_bottom=0.15,ax_subspacing=0.05):
    """
    Plots smoothed radial profiles for each subject and the average across subjects for a given condition or condition contrast.
    Adds bar plots for the binned data below each line plot, with significance brackets for specified comparisons.
    
    Parameters:
        data (dict): Dataset containing radial profiles for all subjects
        avgRadialProfiles (dict): Dictionary containing binned data for bar plots
        analysis_type (str): Type of analysis to plot
        condition (str): Condition name
        kernel (func): smoothing kernel to use
        smooth_factor (float): Smoothing factor for the exponential kernel
        radMax (float): Maximum radius for plotting
        nRadii (int): Number of radii points to plot
        ymin (float): Minimum y-axis limit
        ymax (float): Maximum y-axis limit
        fontsize (float): Font size for axes labels
        fcolor (str): Background color for the plot
        lcolor (str): Line color for the plot
        depth_labels (list): Labels for depth bins
        depthBoundaries (list): Boundaries for depth bins
        plot_indiv (bool): If true, overlay individual subject profiles
        statColor (str): Color of plot lines
        vline (float): x-location of a vertical line
        pval_threshold (float): Threshold for significance marking
        comparisons (dict): p-values for multisample comparisons (optional)
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    all_profiles = {depth_label: {} for depth_label in depth_labels}

    for iR, label in enumerate(data.keys()):
        df = data[label]
        if mask:
            df = df[mask[label]]
        for iD, depth_label in enumerate(depth_labels):
            depth_df = df[(df['d'] >= depthBoundaries[iD, 0]) & (df['d'] < depthBoundaries[iD, 1])]
            coef = depth_df[condition].values
            x = depth_df['scale_xy_dist'].values
            
            coef_smooth, x_smooth = smoothen(coef, x, kernel=kernel, smooth_factor=smooth_factor, radMax=radMax)
            all_profiles[depth_label][label] = coef_smooth
    
    # Plot average profile across subjects
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(fcolor)
    for iD, depth_label in enumerate(depth_labels):
        # Line plot
        p = fig.add_axes([ax_left, ax_bottom + ax_height_bar + iD*ax_spacing + ax_subspacing, ax_width, ax_height])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        layer_profiles_list = list(all_profiles[depth_label].values())
        stat_avg = np.mean(np.vstack(layer_profiles_list), axis=0)
        stat_std = np.std(np.vstack(layer_profiles_list), axis=0)

        p.plot(x_smooth, stat_avg, color=statColor, label=depth_label)
        p.fill_between(x_smooth,
                       stat_avg - stat_std / np.sqrt(len(all_profiles[depth_label])),
                       stat_avg + stat_std / np.sqrt(len(all_profiles[depth_label])),
                       alpha=0.4,
                       color=statColor)

        p.set_ylim([ymin, ymax])
        p.set_ylabel('BOLD % change', fontsize=fontsize, color=lcolor)
        p.set_title(depth_label,fontsize=fontsize)
        if plot_indiv:
            for iR, label in enumerate(data.keys()):
                p.plot(x_smooth.T, np.array(all_profiles[depth_label][label]).T, color=statColor, alpha=0.2)
        p.plot([vline, vline], [ymin, ymax], '--', color='black')
        p.plot([0, radMax], [0, 0], '--', color='black')
        p.set_xlim([0,radMax])
        p.set_xticks(np.linspace(0,maxRad,nRad+1),[])
        p.legend([condition])
        
        # Bar plot
        bar_p = fig.add_axes([ax_left, ax_bottom + iD*ax_spacing, ax_width, ax_height_bar])
        fix_axes(bar_p, lcolor=lcolor, fcolor=fcolor)
        bar_data = avgRadialProfiles[depth_label][condition]['avg']
        bar_std = avgRadialProfiles[depth_label][condition]['stdev']/np.sqrt(avgRadialProfiles[depth_label][condition]['Nsamp'])
        bar_pvals = avgRadialProfiles[depth_label][condition].get('p-vals', None)
        corrected_pvals = avgRadialProfiles[depth_label][condition].get('corrected p-vals', None)

        x_ticks = np.linspace(0+(radMax/(2*nRad)),radMax-(radMax/(2*nRad)),nRad)
        bar_p.bar(x_ticks, bar_data, width=0.9, yerr=bar_std, capsize=5, color=statColor, alpha=0.6)
        bar_p.set_ylabel('Binned Avg', fontsize=fontsize, color=lcolor)
        bar_p.set_ylim([ymin_bar,ymax_bar])
        bar_p.set_xlim([0,radMax])
        if iD == 0:
            bar_p.set_xlabel('relative distance from ROI center ($\sigma$)', fontsize=fontsize, color=lcolor)
            bar_p.set_xticks(np.linspace(0,maxRad,nRad+1))
        else:
            bar_p.set_xticks(np.linspace(0,maxRad,nRad+1),[])
        
        
        # Add significance markers for bar comparisons
        if comparisons and depth_label in comparisons:
            comp_matrix = comparisons[depth_label]
            for row in range(comp_matrix.shape[0]):
                for col in range(row + 1, comp_matrix.shape[1]):  # Only upper triangle
                    pval = comp_matrix[row, col]
                    if np.isfinite(pval) and pval < pval_threshold:
                        # Plot a bracket between the compared bars
                        x1, x2 = x_ticks[row], x_ticks[col]
                        y_max = max(bar_data[row] + bar_std[row], bar_data[col] + bar_std[col]) + 0.5
                        bar_p.plot([x1, x1, x2, x2], [y_max, y_max + 0.1, y_max + 0.1, y_max], lw=1.5, color='k')
                        bar_p.text((x1 + x2) / 2, y_max + 0.05, '*', ha='center', color='k', fontsize=fontsize+4)
        
        # Add significance markers
        #else:
        if type(bar_pvals) == np.ndarray:
            bar_pvals_list = list(bar_pvals)  # If we used a permutation, pvals will be returned in a numpy array
        else:
            bar_pvals_list = list(bar_pvals.pvalue)  # If we used a t-test, pvalues will be returned as a stats object
        for i, pval in enumerate(bar_pvals_list):
            if corrected_pvals is not None and corrected_pvals[i] < pval_threshold:
                bar_p.text(x_ticks[i], bar_data[i] + bar_std[i] + 0, '*', ha='center', va='bottom', color='k', fontsize=fontsize+4)
            elif pval < pval_threshold:
                bar_p.text(x_ticks[i], bar_data[i] + bar_std[i] + 0, '*', ha='center', va='bottom', color='k', fontsize=fontsize+4)
        bar_p.plot([0, radMax], [0, 0], '--', color='black')

    return fig

def save_2samp_results(data_dict, bins, alpha, statTestType, Npermutations = None, output_csv='output.csv', binType = 'norm_depths'):
    
    # Extract inputs from dictionary
    p_vals = data_dict.get('corrected p-vals').flatten()
    df = data_dict.get('df').flatten()
    p_vals= data_dict.get('corrected p-vals').flatten()
    
    # Stack bins
    X, Y = np.meshgrid(bins,bins)
    all_pairs = np.vstack([X.ravel(),Y.ravel()])
    
    # Create a dictionary to construct the DataFrame
    data = {
        binType+' 0': all_pairs[0,:],
        binType+' 1': all_pairs[1,:],
        'df': df,
        'alpha': [alpha] * len(df),
        'p-vals': p_vals
    }
    
    if statTestType == 'permutation':
        Npermutations_array = Npermutations*np.ones(len(df))
        Npermutations_array[Npermutations_array < 2**(df+1)] = 2**(df[Npermutations_array < 2**(df+1)]+1)
        data['Npermutations'] = Npermutations_array
    
    # Convert data to DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

figsize = (9*cm,5*cm)
ax_width = 0.7
ax_height = 0.3
ax_height_bar = 0.3
ax_spacing = 0.1
ax_left = 0.2
ax_bottom = 0.25
ax_subspacing = 0.1
for analysis in ['task']:

    # Get the right stats for this analysis
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
    
    for diff in ['iso-sur']: #DIFFS['statIDs'].keys():
    
        # Only include subjects that should be included
        included_data = return_included_subj(subj_analyses, analysis)
        
        # Get the right stats for this analysis
        STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
        
        diff_id = list(DIFFS['statIDs'].keys()).index(diff)
        diff_color = DIFFS['colors'][diff_id]
        
        comparisons = {}
        for d in rad_depth_labels:
            if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                    comparisons[d] = avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff]['corrected p-vals']
        
        masks = {label: included_data[label]['no_vein'] & included_data[label]['sig'] & (included_data[label]['Visual Region'] == 'V1') for label in included_data.keys()}
        fig = plot_smoothed_radial_profile_wbins(included_data,avgRadialDiff[analysis],analysis,diff,kernel,mask=masks,ymin=-2,ymax=2,ymin_bar=-1,ymax_bar=1,vline=roiRad,statColor=diff_color,pval_threshold=pthresh,radMax=maxRad,nRad=nRad,comparisons=comparisons, depth_labels=rad_depth_labels, depthBoundaries=rad_depthBoundaries, figsize=figsize, ax_width=ax_width, ax_height=ax_height, ax_height_bar=ax_height_bar, ax_spacing=ax_spacing, ax_left=ax_left, ax_bottom=ax_bottom, ax_subspacing=ax_subspacing)
        
        if savefigs:
            fig.savefig(os.path.join(figDir, f"radial_profiles_{analysis}_{diff}_wbins.{fig_format}"))
            if 'corrected p-vals' in avgRadialDiff[analysis][d][diff].keys():
                for d in rad_depth_labels:
                    avgRadialDiff[analysis][d][diff]['rad (sigma)'] = np.arange(maxRad/(2*nRad), maxRad, maxRad/nRad)
                    save_statistical_results(avgRadialDiff[analysis][d][diff], pthresh, statTestType, Npermutations=Npermutations, output_csv = os.path.join(figDir,'stats',f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}.csv"), binType='rad (sigma)')
                    if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                        if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                            save_2samp_results(avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff], np.arange(maxRad/(2*nRad), maxRad, maxRad/nRad), pthresh, statTestType, Npermutations=Npermutations, output_csv = os.path.join(figDir,'stats',f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}_multiComp.csv"), binType='rad (sigma)')

figsize = (8,7)
ax_width = 0.7
ax_height = 0.4
ax_height_bar = 0.3
ax_spacing = 0.3
ax_left = 0.15
ax_bottom = 0.12
skip_diffs = {'task': ["iso-sur"], 'loc': [], 'gPPI': []}
for analysis in ['task','loc','gPPI']:

    # Get the right stats for this analysis
    STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
    diff_list = list(DIFFS['statIDs'].keys())
    for diff_key in skip_diffs[analysis]:
        diff_list.remove(diff_key)
    
    
    for diff in diff_list:
    
        # Only include subjects that should be included
        included_data = return_included_subj(subj_analyses, analysis)
        
        # Get the right stats for this analysis
        STATS, DIFFS = return_stats_diffs(statDetails,diffDetails,stat_analyses,diff_analyses,analysis)
        
        diff_id = list(DIFFS['statIDs'].keys()).index(diff)
        diff_color = DIFFS['colors'][diff_id]
        
        comparisons = {}
        for d in rad_depth_labels:
            if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                    comparisons[d] = avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff]['corrected p-vals']
        
        masks = {label: included_data[label]['no_vein'] & included_data[label]['sig'] & (included_data[label]['Visual Region'] == 'V1') for label in included_data.keys()}
        fig = plot_smoothed_radial_profile_wbins(included_data,avgRadialDiff[analysis],analysis,diff,kernel,mask=masks,ymin=-2,ymax=2,ymin_bar=-1,ymax_bar=1,vline=roiRad,statColor=diff_color,pval_threshold=pthresh,radMax=maxRad,nRad=nRad,comparisons=comparisons, depth_labels=rad_depth_labels, depthBoundaries=rad_depthBoundaries, figsize=figsize, ax_width=ax_width, ax_height=ax_height, ax_height_bar=ax_height_bar, ax_spacing=ax_spacing, ax_left=ax_left, ax_bottom=ax_bottom, ax_subspacing=ax_subspacing)
        
        if savefigs:
            fig.savefig(os.path.join(figDir, f"radial_profiles_{analysis}_{diff}_wbins.{fig_format}"))
            if 'corrected p-vals' in avgRadialDiff[analysis][d][diff].keys():
                for d in rad_depth_labels:
                    avgRadialDiff[analysis][d][diff]['rad (sigma)'] = np.arange(maxRad/(2*nRad), maxRad, maxRad/nRad)
                    save_statistical_results(avgRadialDiff[analysis][d][diff], pthresh, statTestType, Npermutations=Npermutations, output_csv = os.path.join(figDir,'stats',f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}.csv"), binType='rad (sigma)')
                    if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                        if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                            save_2samp_results(avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff], np.arange(maxRad/(2*nRad), maxRad, maxRad/nRad), pthresh, statTestType, Npermutations=Npermutations, output_csv = os.path.join(figDir,'stats',f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}_multiComp.csv"), binType='rad (sigma)')

#%% Look at the condition-dependent V1/V23 interactions as grids

def return_gPPI_matrices(data_dict, analysis_type, roi_label, seed, condition, isDiff = False, transpose=False):
    """
    Returns a 3x3 matrix of averages and p-values for gPPI analysis.
    
    Parameters:
        data_dict (dict): Dictionary containing the data (avgDepthProfiles or avgDepthDiffs)
        analysis_type (str): The analysis type (e.g., 'gPPI')
        roi_label (str): The ROI label (e.g., 'in_ctr')
        seed (str): The seed region (e.g., 'V1' or 'V23')
        transpose (bool): By default, the seed layers are given along the columns 
            and the targets along the rows. If transform is True, this is reversed.
    
    Returns:
        avg_matrix (np.ndarray): 3x3 matrix of the averages
        pval_matrix (np.ndarray): 3x3 matrix of the p-values
    """
    depth_bins = ['deep', 'middle', 'superficial']
    avg_matrix = np.zeros((3, 3))
    pval_matrix = np.zeros((3, 3))
    
    for i, seed_depth in enumerate(depth_bins):
        # Construct the key for avgDepthProfiles
        if not isDiff:
            if not condition:
                key = f"{seed}_{seed_depth}_deveined"
            else:
                key = f"{seed}_{seed_depth}_deveined_{condition}"
        else:
            if not condition:
                key = f"gPPI_{seed_depth}_{seed}"
            else:
                key = f"{condition}_gPPI_{seed_depth}_{seed}"
        
        avg = data_dict[roi_label][analysis_type][key]['avg']
        pval = data_dict[roi_label][analysis_type][key].get('corrected p-vals', None)
        if pval is None:
            if statTestType == 't-test':
                pval = data_dict[roi_label][analysis_type][key]['p-vals'].pvalue
            elif statTestType == 'permutation':
                pval = data_dict[roi_label][analysis_type][key]['p-vals']
            else:
                print("Unknown stat test type: using t-test by default.")
                pval = data_dict[roi_label][analysis_type][key]['p-vals'].pvalue
            
        if transpose:
            avg_matrix[i, :] = avg
            pval_matrix[i, :] = pval
        else:
            avg_matrix[:, i] = avg
            pval_matrix[:, i] = pval
            
    return avg_matrix, pval_matrix

def plot_gPPI_mat(data_mat,p_mat,ROIx,ROIy,seed=None,target=None,Nvox=None,cbar_lims = [-2.0,2.0],title_add='',fig=None,ax=None,fontsize=12,invert_yaxis=True,cbar=True,):
    """ Plots the gPPI matrix """
    
    # Set seed
    if not seed:
        seed = ROIx
    if not target:
        target = ROIy
    
    # Creating a figure and axis if one was not passed
    if fig==None or ax==None:
        fig, ax = plt.subplots()
    
    # Set Depth labels
    depth_labels = ['deep', 'middle', 'superficial']

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
        ax.set_xticklabels(depth_labels,fontsize=0.6*fontsize)
        ax.set_yticklabels(depth_labels,fontsize=0.6*fontsize)
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
    
#create individual condition plot
plot_baseline = False #True if you would like to analyze the baseline V1-V2/3 interactions
seedROI = 'V23' #'V1_tgt' #
targetROI = 'V1' #'V23' #
roi_label = 'in_ctr' #'in_V23' #
ROIx = 'V23'
ROIy = 'V1' #'V1_tgt'

if seedROI != ROIx:
    transpose = True
else:
    transpose = False

if plot_baseline:
    fig, ax = plt.subplots(1,5)
else:
    fig, ax = plt.subplots(1,4)
fig.set_figwidth(14)
fig.set_figheight(6)
fontsize=12

if plot_baseline:
    # seed -> target baseline
    data_mat, p_mat = return_gPPI_matrices(avgDepthProfiles, 'gPPI', roi_label, seedROI, None, transpose = transpose)
        
    plot_gPPI_mat(data_mat,p_mat,ROIx,ROIy,seed=seedROI,target=targetROI,fig=fig,ax=ax[0],cbar=False)

# seed -> target condition
for iC, condition in enumerate(['iso0','iso90','orth','sur']):

    data_mat, p_mat = return_gPPI_matrices(avgDepthProfiles, 'gPPI', roi_label, seedROI, condition, transpose = transpose)
        
    plot_gPPI_mat(data_mat,p_mat,ROIx,ROIy,seed=seedROI,target=targetROI,fig=fig,ax=ax[iC],cbar=False,title_add=condition)
    
plt.tight_layout()

if savefigs:
    fig.savefig(os.path.join(figDir,'gPPI_conditions_mat_seed=%s_target=%s.%s' %(seedROI,targetROI,fig_format)),format=fig_format)

#create contrast plot
plot_baseline = False #True if you would like to analyze the baseline V1-V2/3 interactions
seedROI = 'V23' #'V1' #
targetROI = 'V1' #'V23' #
roi_label = 'in_ctr' #'in_V23' #
ROIx = 'V23'
ROIy = 'V1'

if seedROI != ROIx:
    transpose = True
else:
    transpose = False

if plot_baseline:
    fig, ax = plt.subplots(1,3)
else:
    fig, ax = plt.subplots(1,2)
fig.set_figwidth(14)
fig.set_figheight(6)
fontsize=12

if plot_baseline:
    # seed -> target baseline
    data_mat, p_mat = return_gPPI_matrices(avgDepthDiffs, 'gPPI', roi_label, seedROI, None, isDiff = True, transpose = transpose)
        
    plot_gPPI_mat(data_mat,p_mat,ROIx,ROIy,seed=seedROI,target=targetROI,fig=fig,ax=ax[0],cbar=False)

# seed -> target condition
for iC, condition in enumerate(['odss','fgm']):

    data_mat, p_mat = return_gPPI_matrices(avgDepthDiffs, 'gPPI', roi_label, seedROI, condition, isDiff = True, transpose = transpose)
        
    plot_gPPI_mat(data_mat,p_mat,ROIx,ROIy,seed=seedROI,target=targetROI,fig=fig,ax=ax[iC],cbar=False,title_add=condition)
    
plt.tight_layout()

if savefigs:
    fig.savefig(os.path.join(figDir,'gPPI_diffs_mat_seed=%s_target=%s.%s' %(seedROI,targetROI,fig_format)),format=fig_format)
    
#%% Visualize individual subjects radial profiles

def plot_radial_analysis(all_data, column_name, nDepths_rings=3, plot_type="violin"):
    """
    Generates plots for radial analysis across depth bins for each subject in all_data.

    Parameters:
    - all_data (dict): A dictionary containing the data for each subject.
    - column_name (str): The column name to be used for plotting scatter plots and computing histograms.
    - nDepths_rings (int): Number of depth rings/bins to use for analysis.
    - plot_type (str): Type of plot to use for representing data distribution in radial bins ("violin" or "bar").

    Returns:
    - None: Displays the generated plots.
    """
    
    # Define depth bins
    dbins = np.linspace(0, 1, nDepths_rings + 1)
    
    # Loop through each subject in all_data
    for subj in all_data.keys():
        fig, ax = plt.subplots(nDepths_rings, 3, figsize=(10, 12))
        fig.suptitle(f"{subj}")
        
        for d_i in range(nDepths_rings):
            # Determine the appropriate subplot index
            ax_i = nDepths_rings - d_i - 1
            
            # Filter the data for the depth bin
            df_V1 = all_data[subj][all_data[subj]['Visual Region'] == 'V1']
            depth_bin_data = df_V1[
                (df_V1['sig'] & df_V1['no_vein'] & 
                 (df_V1['d_norm'] >= dbins[d_i]) & 
                 (df_V1['d_norm'] < dbins[d_i + 1]))
            ]
            
            # Define radial bins
            max_dist = np.ceil(np.max(depth_bin_data['scale_xy_dist'])) if len(depth_bin_data) > 0 else 1
            radial_bins = np.arange(0, max_dist + 1, 1)
            
            # Prepare data for radial bins
            avg_values = []
            std_values = []
            radial_bin_values = []
            
            for r_i in range(len(radial_bins) - 1):
                radial_bin_data = depth_bin_data[
                    (depth_bin_data['scale_xy_dist'] >= radial_bins[r_i]) & 
                    (depth_bin_data['scale_xy_dist'] < radial_bins[r_i + 1])
                ]
                values = radial_bin_data[column_name].values
                
                # Prepare data for bar plots
                avg_value = values.mean() if len(values) > 0 else 0
                std_value = (values.std()
                             if len(values) > 1 else 0)
                avg_values.append(avg_value)
                std_values.append(std_value)
                
                # Prepare data for violin plots
                if len(values) > 0:
                    radial_bin_values.append(values)
                else:
                    radial_bin_values.append([0])  # Append a dummy value if no data is available
            
            # Plot based on the selected plot type
            if plot_type == "violin":
                ax[ax_i, 0].violinplot(radial_bin_values, positions=np.arange(len(radial_bins) - 1), showmeans=True)
            elif plot_type == "bar":
                ax[ax_i, 0].bar(radial_bins[:-1], avg_values, width=1, color='b', align='edge')
                ax[ax_i, 0].errorbar(radial_bins[:-1] + 0.5, avg_values, yerr=std_values, fmt='o', color='k', capsize=3)
            else:
                raise ValueError("Invalid plot_type. Choose 'violin' or 'bar'.")

            if d_i == 0:
                ax[ax_i, 0].set_xlabel("Radial Distance ($\sigma$)")
            ax[ax_i, 0].set_ylabel("Value")
            ax[ax_i, 0].set_title(f"Depth bin = {d_i}")
            #ax[ax_i, 0].set_ylim([0, 1.2 * max([np.max(v) for v in radial_bin_values])])
            ax[ax_i, 0].set_xlim([-0.5, len(radial_bins) - 1])
            
            # Surface plots
            for h_i, hemi in enumerate(['lh', 'rh']):
                if hemi in np.unique(df_V1['hemi']):
                    # Extract ellipse parameters
                    ellipse_data = depth_bin_data[
                        (depth_bin_data['hemi'] == hemi)
                    ]
                    
                    if len(ellipse_data) > 0:
                        a = ellipse_data['ellipse_a'].values[0]
                        b = ellipse_data['ellipse_b'].values[0]
                        theta = ellipse_data['ellipse_theta'].values[0]
                        comX = ellipse_data['ellipse_comX'].values[0]
                        comY = ellipse_data['ellipse_comY'].values[0]
                        com = [comX, comY]

                        # Scatter plot
                        pcm = ax[ax_i, h_i + 1].scatter(ellipse_data['x'], ellipse_data['y'], 
                                                        c=ellipse_data[column_name], s=4, 
                                                        cmap='plasma', vmin=-5, vmax=5)
                        cbar = plt.colorbar(pcm, ax=ax[ax_i, h_i + 1])
                        cbar.set_label(column_name, fontsize=10)

                        # Add multiple ellipses with varying scales
                        max_dist = np.max(ellipse_data['scale_xy_dist']) if len(ellipse_data) > 0 else 1
                        for scale in range(1, int(np.ceil(max_dist)) + 1):
                            alpha_value = 1.0 - (scale / np.ceil(max_dist))  # Decrease alpha with increasing scale
                            width = scale * 2 * np.sqrt(a)
                            height = scale * 2 * np.sqrt(b)
                            ellipse = Ellipse(
                                com,
                                width=width,
                                height=height,
                                angle=180 * theta / np.pi,
                                alpha=alpha_value,
                                edgecolor='r',
                                facecolor='none',
                                label=f'Scale {scale}' if scale == 1 else None  # Label only the first for legend
                            )
                            ax[ax_i, h_i + 1].add_patch(ellipse)

                            # Calculate text position along the minor axis of the ellipse
                            theta_rad = np.deg2rad(180 * theta / np.pi)
                            dx = (height / 2) * np.sin(theta_rad)
                            dy = (height / 2) * np.cos(theta_rad)
                            text_x = com[0] + dx
                            text_y = com[1] - dy

                            # Add text indicating the scale factor
                            ax[ax_i, h_i + 1].text(
                                text_x, text_y, f'{scale}$\sigma$',
                                color='darkred', fontsize=6, ha='center', va='center',
                                rotation=theta
                            )

                        # Add labels
                        ax[ax_i, h_i + 1].set_title(f"{hemi} Depth bin = {d_i}")
                        ax[ax_i, h_i + 1].set_xlim([np.min(ellipse_data['x'])-1, np.max(ellipse_data['x'])+1])
                        ax[ax_i, h_i + 1].set_ylim([np.min(ellipse_data['y'])-1, np.max(ellipse_data['y'])+1])
                        ax[ax_i, h_i + 1].axis('off')
                        ax[ax_i, h_i + 1].set_aspect('equal')

        plt.tight_layout()



plot_data = 'ctr-sur_unwarp'
plot_radial_analysis(all_data, plot_data, nDepths_rings=nDepths_rings, plot_type='bar')

#%% Save Analyzed Data
for key, df in all_data.items():
    df.to_csv(f'analyzed_data/{key}.csv', index=False)
    
# Save Profiles
# First make dictionaries json serializable
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_list(item) for item in obj)
    else:
        return obj
    
# Save outputs for each visual area
vArea = 'V1'
subjIDs_V1 = [subj for subj in all_data.keys() if all_data[subj]['Visual Region'].str.contains(vArea).any()]
vArea = 'V23'
subjIDs_V23 = [subj for subj in all_data.keys() if all_data[subj]['Visual Region'].str.contains(vArea).any()]

# Add subject IDs to dictionaries
for task_dict in depthProfiles['in_tgt']:
    for cond_dict in depthProfiles['in_tgt'][task_dict]:
        depthProfiles['in_tgt'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in depthProfiles['in_bor']:
    for cond_dict in depthProfiles['in_bor'][task_dict]:
        depthProfiles['in_bor'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in depthProfiles['in_ctr']:
    for cond_dict in depthProfiles['in_ctr'][task_dict]:
        depthProfiles['in_ctr'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in depthProfiles['in_sur']:
    for cond_dict in depthProfiles['in_sur'][task_dict]:
        depthProfiles['in_sur'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in depthProfiles['in_V23']:
    for cond_dict in depthProfiles['in_V23'][task_dict]:
        depthProfiles['in_V23'][task_dict][cond_dict]['subjIDs'] = subjIDs_V23
for task_dict in diffProfiles['in_tgt']:
    for cond_dict in diffProfiles['in_tgt'][task_dict]:
        diffProfiles['in_tgt'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in diffProfiles['in_bor']:
    for cond_dict in diffProfiles['in_bor'][task_dict]:
        diffProfiles['in_bor'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in diffProfiles['in_ctr']:
    for cond_dict in diffProfiles['in_ctr'][task_dict]:
        diffProfiles['in_ctr'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in diffProfiles['in_sur']:
    for cond_dict in diffProfiles['in_sur'][task_dict]:
        diffProfiles['in_sur'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in diffProfiles['in_V23']:
    for cond_dict in diffProfiles['in_V23'][task_dict]:
        diffProfiles['in_V23'][task_dict][cond_dict]['subjIDs'] = subjIDs_V23
for task_dict in radialProfiles:
    for depth_dict in radialProfiles[task_dict]:
        for cond_dict in radialProfiles[task_dict][depth_dict]:
            radialProfiles[task_dict][depth_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in radialDiffProfiles:
    for depth_dict in radialDiffProfiles[task_dict]:
        for cond_dict in radialDiffProfiles[task_dict][depth_dict]:
            radialDiffProfiles[task_dict][depth_dict][cond_dict]['subjIDs'] = subjIDs_V1

# Depth Profiles
with open(f"analyzed_data/DepthProfiles.json",'w') as outfile:
    json.dump(convert_numpy_to_list(depthProfiles),outfile)
    
# Depth Differences
with open(f"analyzed_data/DepthDiffProfiles.json",'w') as outfile:
    json.dump(convert_numpy_to_list(diffProfiles),outfile)

# Avg Depth Profiles
with open(f"analyzed_data/avgDepthProfiles.json",'w') as outfile:
    json.dump(convert_numpy_to_list(avgDepthProfiles),outfile)
    
# Avg Depth Differences
with open(f"analyzed_data/avgDepthDiffProfiles.json",'w') as outfile:
    json.dump(convert_numpy_to_list(avgDepthDiffs),outfile)

# Radial Profiles
with open(f"analyzed_data/RadialProfiles.json",'w') as outfile:
    json.dump(convert_numpy_to_list(radialProfiles),outfile)

# Radial Differences
with open(f"analyzed_data/RadialDiffProfiles.json",'w') as outfile:
    json.dump(convert_numpy_to_list(radialDiffProfiles),outfile)
    
# Avg Radial Profiles
with open(f"analyzed_data/avgRadialProfiles.json",'w') as outfile:
    json.dump(convert_numpy_to_list(avgRadialProfiles),outfile)

# Radial Differences
with open(f"analyzed_data/avgRadialDiffProfiles.json",'w') as outfile:
    json.dump(convert_numpy_to_list(avgRadialDiff),outfile)