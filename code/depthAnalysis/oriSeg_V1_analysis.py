#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Joseph H. Emerson

OriSeg V1 Analysis

This analysis code has been distilled from the original analysis code for
publication of "Orientation-tuned surround suppression displays unique 
laminar profile in human V1".

Runs the complete laminar depth analysis pipeline and produces all data panels
for the manuscript. QC visualizations are delegated to
QC_visualizations.py. Deveining logic lives in deveining.py.
Statistical correction and result-saving utilities live in run_statistics.py.
Condition/contrast configuration is loaded from analysis_config.json. Radial
visualization functions live in radial_visualizations.py. Depth visualizations 
are produced by depth_visualizations.py, which is executed as a subprocess at 
the end of this script.
"""

# %%
import os
import glob
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.stats as stats
import json
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import csv

# Add func dir to path
funcs_dir = '..'
if funcs_dir not in sys.path:
    sys.path.append(funcs_dir)

# Import custom functions
from oriseg_funcs import *
from deveining import devein_voxels
from run_statistics import (apply_fdr_correction, save_statistical_results,
                             save_2samp_results, save_as_df)
from QC_visualizations import (
    plot_ellipse_qc, plot_pval_histograms,
    plot_depth_boxplots, plot_deveining_qc, 
    plot_mnv_threshold_comparison,
    plot_bold_per_subject, 
    plot_bold_summary,
    plot_individual_condition_profiles,
    plot_individual_diff_profiles,
)
from radial_visualizations import (
    gaussian_kernel,
    plot_smoothed_radial_profile,
    plot_smoothed_radial_profile_wbins
)

plt.close('all')

fcolor = 'white'
lcolor = 'black'
savefigs = True
mainDir = '../..'
figDir  = mainDir + '/figs/depthAnalysis/'
fig_format = 'svg'
fig_size   = "small"  # "small" or "large"
statsDir = os.path.join(mainDir, 'code', 'depthAnalysis', 'stats')

if savefigs and not os.path.exists(figDir):
    os.makedirs(figDir)
if not os.path.exists(statsDir):
    os.makedirs(statsDir)

np.random.seed(68752)

# %% ~ Load Config ~

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_config.json')) as f:
    _cfg = json.load(f)
statDetails  = _cfg['statDetails'] #labels and colors for condition-level beta weights
diffDetails  = _cfg['diffDetails'] #labels and colors for condition contrasts
stat_analyses = _cfg['stat_analyses'] #the index ranges corresponding to each analysis type for single conditions
diff_analyses = _cfg['diff_analyses'] #the index ranges corresponding to each analysis type for condition contrasts

# %% ~ Set Parameters ~

# Radial distance parameters
roiRad     = 1. # radius of the target ROI in sigma
centerRad  = 1. # radius of the center patch in sigma
borderRad  = [1., 3.] # inner and outer radii of the border patch in sigma
surRad     = [3.] # inner radius of the surround patch in sigma
nRad       = 5 # number of radial bins
maxRad     = 5 # maximum radial distance in sigma
rad_depth_labels   = ['all depths'] # depth labels for radial analysis, if analyzing at multiple depths
rad_depthBoundaries = np.array([[0, 1]]) # depth boundaries for radial analysis, if analyzing at multiple depths ([lower_bound, upper_bound])
radBin_comparisons  = [[0, 4]] # pairwise comparisons of radial bins ([bin_idx1, bin_idx2])

# Depth parameters
nDepths      = 7 # number of depth bins for depth analysis

# Mask parameters
use_fullmodel_mask  = True # if true, mask voxels that are not singificantly visually responsive (iso+iso90+orth+sur vs blank)
use_loc_mask        = False # if true, mask voxels that are not significantly target-selective (tgt vs sur)
useSI               = False # if true, compute a selectivity index rather than absolute contrast (cond1 - cond2) / (cond1 + cond2)
pthresh_fullmodel   = 0.01 # p-value threshold for full model mask
pthresh_loc         = 0.01 # p-value threshold for localizer mask

# Statistics parameters
pthresh          = 0.05 # p-value threshold for statistical comparisons
prop_err         = False # if True, use error propagation
showSig          = True # if True, display significance on plots
compareRadtoNull = False # if True, compare each radial profile to the null radial profile (radius shuffled)
statCorrType     = 'fdr_bh' # statistical correction for multiple comparisions type
statTestType     = 'permutation' # type of statistical test for pairwise comparisons
Npermutations    = 10000 # max numnber of permutations for permutation tests

# Depth deconvolution
use_decon = {'task': True, 'loc': True} # if true, evaluate deconvolved depth profiles
p2t_model     = 6.3 # peak-to-tail ratio from Markuerkiaga et al. (2021) for TE = 33.3 ms
Nbins_model   = 10 # numnber of depth bins used in Markuerkiaga et al. (2021) PSF model
normalize_psf = False # if true, normalize the PSF by the deepest bin
# NOTE: Setting normalize_psf to True breaks the linearity of the GLM. This
#       means that the order in which deconvolution happens matters for taking
#       convolutions between conditions. That is, the constrast of deconvolved
#       condition A minus deconvolved condition B is different than the 
#       deconvolution of condition A minus condition B.

# Subject inclusion
subj_analyses = {
    'task': 'all',
    'loc':  'all',
}

# Apply inclusion criteria
def return_included_subj(subj_analyses, analysis_names):
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
        included_IDs = check_subj_labels(list(included_data.keys()), included_IDs)
    included_IDs.sort()
    included_data = {label: all_data[label] for label in included_IDs}

    return included_data


def return_stats_diffs(statDetails, diffDetails, stat_analyses, diff_analyses, analysis):
    if len(stat_analyses[analysis]) > 1:
        STATS = {label: statDetails[label][stat_analyses[analysis][0]:stat_analyses[analysis][1]]
                 for label in statDetails.keys()}
        DIFFS = {
            'statIDs': {key: diffDetails['statIDs'][key]
                        for key in diff_analyses[analysis]['list']},
            'colors':  diffDetails['colors'][diff_analyses[analysis]['ids'][0]:
                                             diff_analyses[analysis]['ids'][1]],
        }
    else:
        STATS = {label: statDetails[label][stat_analyses[analysis][0]:]
                 for label in statDetails.keys()}
        DIFFS = {
            'statIDs': {key: diffDetails['statIDs'][key]
                        for key in diff_analyses[analysis]['list']},
            'colors':  diffDetails['colors'][diff_analyses[analysis]['ids'][0]:],
        }

    return (STATS, DIFFS)


# %% ~ Import Data ~

datasets  = glob.glob(os.path.join(mainDir, 'data', 'roi_data/target_roi_manual',
                                       'pnr???_??_???_??.csv'))

exclude_V1 = [
    'pnr143_V1_tgt_rh',
    'pnr161_V1_tgt_lh', 'pnr161_V1_tgt_rh',
    'pnr352_V1_tgt_lh', 'pnr352_V1_tgt_rh',
    'pnr579_V1_tgt_lh',
    'pnr668_V1_tgt_rh',
]
for e_i, excl in enumerate(exclude_V1):
    datasets.remove(
        os.path.join(mainDir, 'data', 'roi_data/target_roi_manual', excl+'.csv'))
datasets.sort()
Ndsets_V1 = len(datasets)
print(f"{Ndsets_V1} V1 ROIs")

all_data  = {}

for file_path in datasets:
    file_name = os.path.basename(file_path)
    subjID, visArea, subReg, hemi = file_name.replace('.csv', '').split('_')
    df = pd.read_csv(file_path, index_col=False)
    df['Subject ID']    = subjID
    df['Visual Region'] = visArea
    df['Subregion']     = subReg
    df['hemi']          = hemi
    if subjID in all_data:
        all_data[subjID] = pd.concat([all_data[subjID], df], ignore_index=True)
    else:
        all_data[subjID] = df

for iR, label in enumerate(all_data.keys()):
    df = all_data[label]
    df = df.drop(df[df['d'] == 0].index)
    all_data[label] = df

# %% ~ Visualize Data (QC) ~

plotDir = os.path.join(figDir, 'QC')
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

# %% Ellipse computation (modifies all_data; needed for downstream analysis)
# Fit an ellipse to the u and v coordinates of the target-selective voxels

ellipse_df = pd.DataFrame({
    'subjID': [], 'hemi': [],
    'ROI major axis (mm)': [], 'ROI minor axis (mm)': [],
    'theta': [], 'comX (mm)': [], 'comY (mm)': [],
    'area (mm^2)': [], 'semimajor axis (mm)': [], 'semiminor axis (mm)': [],
})

for iS, label in enumerate(all_data.keys()):
    df_all = all_data[label]

    for col in ['xy_dist', 'ellipse_a', 'ellipse_b',
                'ellipse_theta', 'ellipse_comX', 'ellipse_comY']:
        df_all[col] = np.nan

    for iH, hemi in enumerate(df_all['hemi'].unique()):

        if np.sum((df_all['Visual Region'] == 'V1') &
                  (df_all['Subregion'] == 'tgt') &
                  (df_all['hemi'] == hemi)) != 0:

            df = df_all[(df_all['Visual Region'] == 'V1') &
                            (df_all['Subregion'] == 'tgt') &
                            (df_all['hemi'] == hemi)]
            tgt_df = df[df['ctr-sur'] > 0]
            cov = np.cov(tgt_df['x'][df['scale_xy_dist'] < 2.2],
                            tgt_df['y'][df['scale_xy_dist'] < 2.2])
            com = (np.mean(tgt_df['x'][df['scale_xy_dist'] < 2.2]),
                      np.mean(tgt_df['y'][df['scale_xy_dist'] < 2.2]))

            df['xy_dist'] = np.sqrt((df['x'].values-com[0])**2 +
                                     (df['y'].values-com[1])**2)
            df_all.loc[(df_all['hemi'] == hemi) &
                       (df_all['Visual Region'] == 'V1') &
                       (df_all['Subregion'] == 'tgt'), 'xy_dist'] = df['xy_dist']

            a = ((cov[0,0] + cov[1,1])/2 +
                 np.sqrt(((cov[0,0] - cov[1,1])/2)**2 + cov[0,1]**2))
            b = ((cov[0,0] + cov[1,1])/2 -
                 np.sqrt(((cov[0,0] - cov[1,1])/2)**2 + cov[0,1]**2))
            print(label)
            print('\t %s major axis (a): %2.2f' % (hemi, a))
            print('\t %s minor axis (b): %2.2f' % (hemi, b))
            theta = np.arctan2(a - cov[0,0], cov[1,0])
            df['ellipse_a'] = a
            df['ellipse_b'] = b
            df['ellipse_theta'] = theta
            df['ellipse_comX'] = com[0]
            df['ellipse_comY'] = com[1]
            ellipse_df.loc[len(ellipse_df)] = [
                label, hemi,
                roiRad*np.sqrt(a), roiRad*np.sqrt(b), theta, com[0], com[1],
                np.pi*(roiRad*np.sqrt(a))*(roiRad*np.sqrt(b)),
                (roiRad*np.sqrt(a) + roiRad*np.sqrt(b))/2,
                np.sqrt((roiRad*np.sqrt(a))*(roiRad*np.sqrt(b))),
            ]

            for col in ['xy_dist', 'ellipse_a', 'ellipse_b',
                        'ellipse_theta', 'ellipse_comX', 'ellipse_comY']:
                df_all.loc[df.index, col] = df[col]

        else:
            print(f"No V1_tgt_{hemi} for {label}")

    all_data[label] = df_all

avgs  = ellipse_df.mean(axis=0, numeric_only=True)
stds  = ellipse_df.std(axis=0,  numeric_only=True)
mins  = ellipse_df.min(axis=0,  numeric_only=True)
maxs  = ellipse_df.max(axis=0,  numeric_only=True)
Nrois = len(ellipse_df)
ellipse_stats = pd.DataFrame({
    'avg': avgs, 'st. dev': stds, 'min': mins, 'max': maxs,
    'NROIs': Nrois * np.ones(len(avgs)),
})

if savefigs:
    if not os.path.exists(os.path.join(statsDir, "ROIs")):
        os.makedirs(os.path.join(statsDir, "ROIs"))
    ellipse_df.to_csv(os.path.join(statsDir, "ROIs", 'ROIstats.csv'))
    ellipse_stats.to_csv(os.path.join(statsDir, "ROIs", 'ROIsummary.csv'))

plot_ellipse_qc(all_data, ellipse_df, 2*roiRad, Ndsets_V1, savefigs, plotDir, fig_format)

# %% p-value and depth statistics (QC)

subject_visArea_combinations = [
    (subjID, vis_region)
    for subjID, df in all_data.items()
    for vis_region in df['Visual Region'].unique()
]

plot_pval_histograms(all_data, 10, subject_visArea_combinations,
                     savefigs, plotDir, fig_format)

# %% ~ Analysis Preprocessing ~

# %% Deveining

lmnv_dict = devein_voxels(
    all_data,
    depth_var='d_norm',
    deep_pct=10,
    sd_thresh=2,
    out_dir=os.path.join(statsDir, 'devein'),
    vAreas=['V1']
)

# Deveining QC plots
plot_deveining_qc(all_data, lmnv_dict, nDepths,
                  savefigs=savefigs, figDir=plotDir, fig_format=fig_format)
plot_mnv_threshold_comparison(all_data, lmnv_dict,
                               savefigs=savefigs, figDir=plotDir, fig_format=fig_format)

# %% Apply full model p-val mask

Nsubj = len([subj for subj in all_data.keys()])

for k_i, key in enumerate(all_data.keys()):
    all_data[key]['sig'] = (np.ones(len(all_data[key])) == 1)

fullmodel_thresh_df = pd.DataFrame({
    'subjID': [], ' Nvox significant': [], 'Nvox total': [],
    'Nvox_sig / Nvox_total': [], 'p_thresh': [],
})
if use_fullmodel_mask:
    for k_i, key in enumerate(all_data.keys()):
        df         = all_data[key]
        pvals      = df['task p-val']
        pval_mask  = pvals < pthresh_fullmodel
        Nsig_pval  = np.sum(pval_mask)
        print("%d/%d voxels survive full model p-val mask" % (Nsig_pval, np.size(pval_mask)))
        all_data[key]['sig'] = (all_data[key]['sig'] & pval_mask)
        fullmodel_thresh_df.loc[len(fullmodel_thresh_df)] = [
            key, Nsig_pval, np.size(pval_mask),
            Nsig_pval/np.size(pval_mask), pthresh_fullmodel,
        ]

    avgs = fullmodel_thresh_df.mean(axis=0, numeric_only=True)
    stds = fullmodel_thresh_df.std(axis=0,  numeric_only=True)
    mins = fullmodel_thresh_df.min(axis=0,  numeric_only=True)
    maxs = fullmodel_thresh_df.max(axis=0,  numeric_only=True)
    fullmodel_thresh_stats = pd.DataFrame({
        'avg': avgs, 'std': stds,
        'Nsubj': Nsubj*np.ones(len(avgs)), 'min': mins, 'max': maxs,
    })
    if not os.path.exists(os.path.join(statsDir, "QC")):
        os.makedirs(os.path.join(statsDir, "QC"))
    fullmodel_thresh_df.to_csv(os.path.join(statsDir, 'QC', 'task_pval_mask.csv'))
    fullmodel_thresh_stats.to_csv(os.path.join(statsDir, 'QC', 'task_pval_mask_summary.csv'))

loc_thresh_df = pd.DataFrame({
    'subjID': [], ' Nvox significant': [], 'Nvox total': [],
    'Nvox_sig / Nvox_total': [], 'p_thresh': [],
})
if use_loc_mask:
    for k_i, key in enumerate(all_data.keys()):
        df        = all_data[key]
        pvals     = df['loc p-val']
        pval_mask = pvals < pthresh_loc
        Nsig_pval = np.sum(pval_mask)
        print("%d/%d voxels survive loc p-val mask" % (Nsig_pval, np.size(pval_mask)))
        all_data[key]['sig'] = (all_data[key]['sig'] & pval_mask)
        loc_thresh_df.loc[len(loc_thresh_df)] = [
            key, Nsig_pval, np.size(pval_mask),
            Nsig_pval/np.size(pval_mask), pthresh_loc,
        ]

    avgs = loc_thresh_df.mean(axis=0, numeric_only=True)
    stds = loc_thresh_df.std(axis=0,  numeric_only=True)
    mins = loc_thresh_df.min(axis=0,  numeric_only=True)
    maxs = loc_thresh_df.max(axis=0,  numeric_only=True)
    loc_thresh_stats = pd.DataFrame({
        'avg': avgs, 'std': stds,
        'Nsubj': Nsubj*np.ones(len(avgs)), 'min': mins, 'max': maxs,
    })
    if not os.path.exists(os.path.join(statsDir, "QC")):
        os.makedirs(os.path.join(statsDir, "QC"))
    loc_thresh_df.to_csv(os.path.join(statsDir, 'QC', 'loc_pval_mask.csv'))
    loc_thresh_stats.to_csv(os.path.join(statsDir, 'QC', 'loc_pval_mask_summary.csv'))

# %% Compute iso for localizer condition

for key in list(all_data.keys()):
    all_data[key]['iso0_loc'] = all_data[key]['iso0'] - all_data[key]['sur']

# %% ~ Get Depth Profiles ~

profile_method = 'bin'
for key in all_data.keys():
    df = all_data[key][all_data[key]['Visual Region'] == 'V1']
    all_data[key].loc[all_data[key]['Visual Region'] == 'V1', 'in_tgt'] = df['scale_xy_dist'] < roiRad

masks = {'in_tgt': {}}
for roi in masks.keys():
    masks[roi] = {key: all_data[key][roi]*all_data[key]['sig']*all_data[key]['no_vein']
                  for key in all_data.keys()}

depthProfiles = {'in_tgt': {}}
diffProfiles  = {'in_tgt': {}}

for roi in depthProfiles.keys():
    if roi == 'in_tgt':
        nD           = nDepths
        radialParam  = 'scale_xy_dist'

    for analysis in stat_analyses.keys():
        included_data = return_included_subj(subj_analyses, analysis)
        STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                            stat_analyses, diff_analyses, analysis)
        depthProfiles[roi][analysis] = compute_all_depth_profiles(
            included_data, STATS, profile_method, nD, masks[roi],
            depthParam='d', radialParam=radialParam,
            spec_Drange='MinMax', statTestType='t-test',
        )
        diffProfiles[roi][analysis] = compute_diff_profiles(
            included_data, STATS, DIFFS['statIDs'], profile_method, nD, useSI,
            masks[roi], depthParam='d', radialParam=radialParam,
            spec_Drange='MinMax', statTestType='t-test',
        )

# %% Save number of voxels per depth

for roi, analyses in depthProfiles.items():
    for analysis, conditions in analyses.items():
        n_list = None
        for condition, stats_d in conditions.items():
            if 'N' in stats_d:
                n_list = stats_d['N']
                break

        if n_list is not None:
            n_array      = np.vstack(n_list)
            csv_filename = f"{roi}_{analysis}_N.csv"
            csv_filepath = os.path.join(statsDir, 'ROIs', csv_filename)
            avg_n        = np.mean(n_array, axis=0)
            std_n        = np.std(n_array,  axis=0)

            with open(csv_filepath, mode='w', newline='') as file:
                writer        = csv.writer(file)
                depth_numbers = np.linspace(0, 1, n_array.shape[1])
                writer.writerow(["Subject"] + list(depth_numbers))
                subject_names = list(all_data.keys())
                for idx, row in enumerate(n_array):
                    writer.writerow([subject_names[idx]] + list(row))
                writer.writerow(["Average"] + list(avg_n))
                writer.writerow(["Std Dev"] + list(std_n))

print(f"CSV files have been saved in '{os.path.join(statsDir, 'ROIs')}' directory.")

# Also write depth bin values back to all_data
for key in all_data.keys():
    df = all_data[key]
    roi = df[masks['in_tgt'][key] == 1]

    # set up the depths
    depthMin = roi['d'].min()
    depthMax = roi['d'].max()
    binSize = (depthMax-depthMin)/nDepths
    print('depth bin size: %.2f' %binSize)
    depthBoundaries = np.linspace(depthMin-binSize/2, depthMax, nDepths+1)
    
    # discretize the depth values into bins
    df['d_bin'] = np.digitize(df['d'], depthBoundaries, right=True) - 1
    all_data[key] = df

# Plot voxels per depth
plot_depth_boxplots(all_data, nDepths, masks=masks['in_tgt'], savefigs=savefigs, figDir=plotDir, fig_format=fig_format, plottype="box")

# Check consistency with depthProfiles
for roi in depthProfiles.keys():
    for analysis in depthProfiles[roi].keys():
        for condition in depthProfiles[roi][analysis].keys():
            N_list = depthProfiles[roi][analysis][condition]['N']
            for iS, label in enumerate(all_data.keys()):
                df = all_data[label]
                df = df[masks[roi][label] == 1]
                n_voxels_per_bin = df['d_bin'].value_counts().sort_index()
                n_voxels_per_bin = n_voxels_per_bin.reindex(range(0, nDepths), fill_value=0)
                assert np.array_equal(n_voxels_per_bin.values, N_list[iS]), f"Inconsistency in voxel counts for {label}, {roi}, {analysis}, {condition}"

# %% Total % BOLD change (QC)

if not os.path.exists(os.path.join(statsDir, "QC")):
    os.makedirs(os.path.join(statsDir, "QC"))

plot_bold_per_subject(all_data, statDetails, savefigs, plotDir, fig_format)
plot_bold_summary(all_data, statDetails, savefigs, figDir, fig_format, statsDir)

# %% Deconvolution

for roi in depthProfiles.keys():

    nD = nDepths

    for analysis in ['task', 'loc']:
        
        included_data = return_included_subj(subj_analyses, analysis)

        STATS, DIFFS = return_stats_diffs(statDetails, diffDetails,
                                           stat_analyses, diff_analyses, analysis)

        dP    = depthProfiles[roi][analysis]
        diffP = diffProfiles[roi][analysis]

        keep_rois  = np.zeros((len(included_data.keys()), len(STATS['labels']), nD))
        for iR, roiID in enumerate(included_data.keys()):
            for iStat, stat in enumerate(STATS['labels']):
                keep_rois[iR, iStat, :] = dP[stat]['avg'][iR]

        keep_diffs = np.zeros((len(included_data.keys()), len(DIFFS['statIDs'].keys()), nD))
        for iR, roiID in enumerate(included_data.keys()):
            for iDiff, diff in enumerate(DIFFS['statIDs'].keys()):
                keep_diffs[iR, iDiff, :] = diffP[diff]['avg'][iR]

        decon_rois  = depth_deconv(keep_rois,  p2t_model, Nbins_model, nD, normalize_psf)
        decon_diffs = depth_deconv(keep_diffs, p2t_model, Nbins_model, nD, normalize_psf)

        for iStat, stat in enumerate(STATS['labels']):
            depthProfiles[roi][analysis][stat]['avg_decon'] = np.squeeze(
                np.array(decon_rois)[:, iStat, :]
            )

        for iDiff, diff in enumerate(DIFFS['statIDs'].keys()):
            diffProfiles[roi][analysis][diff]['avg_decon'] = np.squeeze(
                np.array(decon_diffs)[:, iDiff, :]
            )

# %% Compute average across subjects

avgDepthProfiles = {'in_tgt': {}}
avgDepthDiffs    = {'in_tgt': {}}

for roi in depthProfiles.keys():
    nD = nDepths

    for analysis in stat_analyses.keys():
        included_data = return_included_subj(subj_analyses, analysis)
        STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                            stat_analyses, diff_analyses, analysis)
        [avgDepthProfiles[roi][analysis],
         avgDepthDiffs[roi][analysis]] = compute_avg_depth_profile(
            depthProfiles[roi][analysis], STATS, DIFFS['statIDs'],
            STATS['labels'], list(DIFFS['statIDs'].keys()),
            use_decon[analysis], prop_err, useSI,
            statTestType=statTestType, num_permutations=Npermutations,
        )

# %% ~ Analysis ~

# %% Radial profiles analysis

binSize      = maxRad / nRad
radBins      = np.linspace(0, maxRad, nRad + 1)
radBinCtrs   = radBins[:-1] + binSize/2

radialProfiles     = {analysis_type: {} for analysis_type in subj_analyses.keys()}
radialDiffProfiles = {analysis_type: {} for analysis_type in subj_analyses.keys()}
avgRadialProfiles  = {analysis_type: {} for analysis_type in subj_analyses.keys()}
avgRadialDiff      = {analysis_type: {} for analysis_type in subj_analyses.keys()}

for analysis in subj_analyses.keys():
    included_data = return_included_subj(subj_analyses, analysis)
    STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                        stat_analyses, diff_analyses, analysis)

    layer_mask_dict = {l: {} for l in rad_depth_labels}
    for iR, d_label in enumerate(included_data.keys()):
        for iB, dB in enumerate(rad_depthBoundaries):
            df      = all_data[d_label]
            lmask   = (df['d'] > dB[0]) & (df['d'] < dB[1])
            roi_mask = (included_data[d_label]['no_vein'] & included_data[d_label]['sig'])
            df[rad_depth_labels[iB]] = lmask
            layer_mask_dict[rad_depth_labels[iB]][d_label] = lmask & roi_mask

    for il, l in enumerate(rad_depth_labels):
        radialProfiles[analysis][l] = compute_all_rad_profiles(
            included_data, STATS, 'bin', nRad, layer_mask_dict[l],
            radParam='scale_xy_dist', spec_Drange=[0, maxRad], radMax=maxRad,
        )
        radialDiffProfiles[analysis][l] = compute_rad_diff_profiles(
            included_data, STATS, DIFFS['statIDs'], 'bin', nRad, prop_err,
            layer_mask_dict[l],
            radParam='scale_xy_dist', spec_Drange=[0, maxRad], radMax=maxRad,
        )
        avgRadialProfiles[analysis][l], avgRadialDiff[analysis][l] = compute_avg_rad_profile(
            radialProfiles[analysis][l], STATS, DIFFS['statIDs'],
            STATS['labels'], list(DIFFS['statIDs'].keys()),
            prop_err, useSI, statTestType=statTestType, npermSamples=Npermutations,
        )

# %% Multisample comparisons
# Within condition comparisons across bins

avgRadialDiff_comparisons = {analysis_type: {} for analysis_type in subj_analyses.keys()}

contrasts   = ['odss', 'fgm']
experiments = ['task']
for e_i, e in enumerate(experiments):
    avgRadialDiff_comparisons[e]['withinCondition'] = {}
    avgRadialDiff_comparisons[e]['withinCondition']['across_rad'] = {}
    for d_i, d in enumerate(rad_depth_labels):
        avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d] = {}
        for c_i, c in enumerate(contrasts):
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c] = {}
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['p-vals'] = np.empty((nRad, nRad))
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['p-vals'][:] = np.nan
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['t-stat'] = np.empty((nRad, nRad))
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['t-stat'][:] = np.nan
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['df'] = np.empty((nRad, nRad))
            avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['df'][:] = np.nan
            for rb_i, rb in enumerate(radBin_comparisons):
                if statTestType == 't-test':
                    tstat, pval = stats.ttest_rel(
                        np.array(radialDiffProfiles[e][d][c]['avg'])[:, rb[0]],
                        np.array(radialDiffProfiles[e][d][c]['avg'])[:, rb[1]],
                    )
                elif statTestType == 'permutation':
                    diffs_rb = (np.array(radialDiffProfiles[e][d][c]['avg'])[:, rb[0]] -
                                np.array(radialDiffProfiles[e][d][c]['avg'])[:, rb[1]])
                    pval  = permute_1samp(diffs_rb, np.mean, n_permutations=Npermutations)
                    tstat = np.nan
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['p-vals'][rb[0], rb[1]] = pval
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['p-vals'][rb[1], rb[0]] = pval
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['t-stat'][rb[0], rb[1]] = tstat
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['t-stat'][rb[1], rb[0]] = tstat
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['df'][rb[0], rb[1]] = (
                    np.shape(radialDiffProfiles[e][d][c]['avg'])[0] - 1
                )
                avgRadialDiff_comparisons[e]['withinCondition']['across_rad'][d][c]['df'][rb[1], rb[0]] = (
                    np.shape(radialDiffProfiles[e][d][c]['avg'])[0] - 1
                )

# %% Corrections for Multiple Comparisons

exclude_indiv_ttest     = True #if True, only include across-subject statistical tests and not within subject statistical tests
exclude_analysis_types  = []
exclude_rois            = []
exclude_cond = ['iso0', 'iso90', 'orth', 'sur', #exclude individual condition t-tests, only focus on condition contrasts
                'ctr_unwarp', 'sur_unwarp', 'ctr-sur_unwarp',
                'V23_superficial_deveined_orth', 'V23_middle_deveined_orth', 'V23_deep_deveined_orth',
                'V23_superficial_deveined_iso90', 'V23_middle_deveined_iso90', 'V23_deep_deveined_iso90',
                'V23_superficial_deveined_iso0', 'V23_middle_deveined_iso0', 'V23_deep_deveined_iso0',
                'V23_superficial_deveined_sur', 'V23_middle_deveined_sur', 'V23_deep_deveined_sur',
                'V1_tgt_superficial_deveined_orth', 'V1_tgt_middle_deveined_orth', 'V1_tgt_deep_deveined_orth',
                'V1_tgt_superficial_deveined_iso90', 'V1_tgt_middle_deveined_iso90', 'V1_tgt_deep_deveined_iso90',
                'V1_tgt_superficial_deveined_iso0', 'V1_tgt_middle_deveined_iso0', 'V1_tgt_deep_deveined_iso0',
                'V1_tgt_superficial_deveined_sur', 'V1_tgt_middle_deveined_sur', 'V1_tgt_deep_deveined_sur',
                'iso-sur_gPPI_superficial_V23', 'iso-sur_gPPI_middle_V23', 'iso-sur_gPPI_deep_V23',
                'iso-sur_gPPI_superficial_V1', 'iso-sur_gPPI_middle_V1', 'iso-sur_gPPI_deep_V1'
                'dsi'
                ]
exclude_combinations = [
    ('in_ctr', 'loc', 'all'),
    ('in_bor', 'loc', 'all'),
    ('in_sur', 'loc', 'all'),
]

if exclude_indiv_ttest:
    search_dicts = [avgDepthProfiles, avgDepthDiffs, avgRadialProfiles, avgRadialDiff]
else:
    search_dicts = [depthProfiles, diffProfiles,
                    avgDepthProfiles, avgDepthDiffs, avgRadialProfiles, avgRadialDiff]

apply_fdr_correction(
    search_dicts, avgRadialDiff_comparisons,
    exclude_analysis_types=exclude_analysis_types,
    exclude_rois=exclude_rois,
    exclude_cond=exclude_cond,
    exclude_combinations=exclude_combinations,
    statCorrType=statCorrType,
)

# %% ~ Plots ~

# %% V1 Depth Profiles
# Now make plots based on the previous analysis

roi_type = 'in_tgt'

save_as_df(avgDepthDiffs, roi_type, 'task', figDir, statTestType)
save_as_df(avgDepthDiffs, roi_type, 'task', figDir, statTestType)  # preserving original call order

cm = 1/2.54
for analysis in ['task', 'loc']:
    fig = plt.figure(figsize=(6*cm, 4*cm))
    fig.set_size_inches((6*cm, 4*cm))
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

    ylim   = [-0.02, 1.02]
    xlim   = [-1, 6]
    Ntext  = [4, 0.05]
    STATS, DIFFS = return_stats_diffs(statDetails, diffDetails,
                                       stat_analyses, diff_analyses, analysis)
    xticks = [0, 2, 4, 6]
    plot_avg_depth_profile(p1, avgDepthProfiles[roi_type][analysis],
                           STATS['labels'], STATS['colors'],
                           ylim, xlim, dx, dy, Ntext, lcolor, fsize, xticks=xticks)

    p2 = fig.add_axes([.7, .27, .25, .7])
    fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
    xlim = [-0.3, 1.8]
    plot_avg_diff_profile(p2, avgDepthDiffs[roi_type][analysis],
                          DIFFS['statIDs'].keys(), DIFFS['colors'],
                          ylim, xlim, dx, dy, Ntext, lcolor, fsize, useSI)

    if savefigs:
        if use_decon:
            fig.savefig(os.path.join(figDir, 'avg_profiles_%s_%s_deconv.%s'
                                     % (analysis, roi_type, fig_format)))
        else:
            fig.savefig(os.path.join(figDir, 'avg_profiles_%s_%s.%s'
                                     % (analysis, roi_type, fig_format)))

# %% Plot context modulation effect separately

def save_statistical_results_local(data_dict, alpha, statTestType, Npermutations=None,
                                    output_csv='output.csv', binType='norm_depths'):
    save_statistical_results(data_dict, alpha, statTestType, Npermutations,
                              output_csv, binType)

for roi_type in ['in_tgt']:
    for analysis in ['task', 'loc']:
        STATS, DIFFS = return_stats_diffs(statDetails, diffDetails,
                                           stat_analyses, diff_analyses, analysis)
        for iDiff, diff in enumerate(DIFFS['statIDs'].keys()):

            if fig_size == "small":
                fig = plt.figure(figsize=(6*cm, 4*cm))
                fig.set_size_inches((6*cm, 4*cm))
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

            ylim   = [-0.02, 1.02]
            xlim   = [-1, 6]
            Ntext  = [4, 0.05]

            stat_labels   = DIFFS['statIDs'][diff]
            stat_labels_i = [STATS['labels'].index(item) for item in stat_labels]
            stat_colors   = [STATS['colors'][i] for i in stat_labels_i]

            xticks = [0, 2, 4, 6]
            plot_avg_depth_profile(p1, avgDepthProfiles[roi_type][analysis],
                                   stat_labels, stat_colors,
                                   ylim, xlim, dx, dy, Ntext, lcolor, fsize, xticks=xticks)

            if analysis == 'loc':
                xlim = [-0.5, 2.5]
            else:
                xlim = [-0.5, 1.5]
            p2   = fig.add_axes([.7, .27, .25, .7])
            fix_axes(p2, lcolor=lcolor, fcolor=fcolor)
            if 'corrected p-vals' in avgDepthDiffs[roi_type][analysis][diff].keys():
                plot_avg_diff_profile(p2, avgDepthDiffs[roi_type][analysis],
                                      [diff], [DIFFS['colors'][iDiff]],
                                      ylim, xlim, dx, dy, Ntext, lcolor, fsize, useSI,
                                      showSig=True, pthresh=pthresh,
                                      statCorrType=avgDepthDiffs[roi_type][analysis][diff]['corrected p-vals'])
            else:
                plot_avg_diff_profile(p2, avgDepthDiffs,
                                      [diff], [DIFFS['colors'][iDiff]],
                                      ylim, xlim, dx, dy, Ntext, lcolor, fsize, useSI,
                                      showSig=False)
            if savefigs:
                if use_decon:
                    fig.savefig(os.path.join(figDir, 'avg_profiles_%s_%s_%s_deconv.%s'
                                             % (analysis, roi_type, diff, fig_format)))
                    if 'corrected p-vals' in avgDepthDiffs[roi_type][analysis][diff].keys():
                        if not os.path.exists(os.path.join(statsDir, 'depth')):
                                os.makedirs(os.path.join(statsDir, 'depth'))
                        save_statistical_results(
                            avgDepthDiffs[roi_type][analysis][diff],
                            pthresh, statTestType, Npermutations=Npermutations,
                            output_csv=os.path.join(statsDir, 'depth',
                                                     'avg_profiles_%s_%s_%s_%s_deconv.csv'
                                                     % (analysis, roi_type, diff, statTestType)),
                        )
                else:
                    fig.savefig(os.path.join(figDir, 'avg_profiles_%s_%s_%s.%s'
                                             % (analysis, roi_type, diff, fig_format)))
                    if 'corrected p-vals' in avgDepthDiffs[roi_type][analysis][diff].keys():
                        if not os.path.exists(os.path.join(statsDir, 'depth')):
                            os.makedirs(os.path.join(statsDir, 'depth'))
                        save_statistical_results(
                            avgDepthDiffs[roi_type][analysis][diff],
                            pthresh, statTestType, Npermutations=Npermutations,
                            output_csv=os.path.join(statsDir, 'depth',
                                                     'avg_profiles_%s_%s_%s_%s.csv'
                                                     % (analysis, roi_type, diff, statTestType)),
                        )

# %% Individual subject depth profiles (QC)

roi_types = ['in_tgt']
plot_individual_condition_profiles(
    depthProfiles, avgDepthProfiles,
    statDetails, diffDetails, stat_analyses, diff_analyses,
    use_decon, "large", savefigs, figDir, fig_format, fcolor, lcolor,
    roi_types
)

plot_individual_diff_profiles(
    diffProfiles, avgDepthDiffs,
    statDetails, diffDetails, stat_analyses, diff_analyses,
    use_decon, "large", pthresh, savefigs, figDir, fig_format, fcolor, lcolor,
    roi_types
)

# %% Loc Profiles across surface

kernel        = smooth_kernel
smooth_factor = 0.3
nRadii        = 20
ymax          = 7
ymin          = -7
highlight     = False

for analysis in ['task', 'loc']:
    STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                        stat_analyses, diff_analyses, analysis)
    for s_i, stat in enumerate(STATS['labels']):
        included_data = return_included_subj(subj_analyses, analysis)
        masks_rad = {label: (included_data[label]['no_vein'] &
                              included_data[label]['sig'] &
                              (included_data[label]['Visual Region'] == 'V1'))
                     for label in included_data.keys()}
        fig = plot_smoothed_radial_profile(
            included_data, analysis, stat, kernel,
            mask=masks_rad, vline=roiRad, radMax=maxRad,
            ymin=ymin, ymax=ymax, statColor=STATS['colors'][s_i],
            depth_labels=rad_depth_labels, depthBoundaries=rad_depthBoundaries,
        )
        if savefigs:
            fig.savefig(os.path.join(figDir,
                        f"radial_profiles_{analysis}_{stat}.{fig_format}"))


# %% Compute differences and put them back in all_data

for analysis in ['task', 'loc']:
    included_data = return_included_subj(subj_analyses, analysis)
    STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                        stat_analyses, diff_analyses, analysis)
    for diff in DIFFS['statIDs'].keys():
        for iR, label in enumerate(included_data.keys()):
            stat1 = DIFFS['statIDs'][diff][0]
            stat2 = DIFFS['statIDs'][diff][1]
            included_data[label][diff] = included_data[label][stat1] - included_data[label][stat2]

# %% Add Binned Radial Data

figsize       = (9*cm, 5*cm)
ax_width      = 0.7
ax_height     = 0.3
ax_height_bar = 0.3
ax_spacing    = 0.1
ax_left       = 0.2
ax_bottom     = 0.25
ax_subspacing = 0.1
for analysis in ['task']:
    STATS, DIFFS = return_stats_diffs(statDetails, diffDetails,
                                       stat_analyses, diff_analyses, analysis)
    for diff in ['iso-sur']:
        included_data = return_included_subj(subj_analyses, analysis)
        STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                            stat_analyses, diff_analyses, analysis)
        diff_id    = list(DIFFS['statIDs'].keys()).index(diff)
        diff_color = DIFFS['colors'][diff_id]

        comparisons = {}
        for d in rad_depth_labels:
            if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                    comparisons[d] = (
                        avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff]['corrected p-vals']
                    )

        masks_rad = {label: (included_data[label]['no_vein'] &
                              included_data[label]['sig'] &
                              (included_data[label]['Visual Region'] == 'V1'))
                     for label in included_data.keys()}
        fig = plot_smoothed_radial_profile_wbins(
            included_data, avgRadialDiff[analysis], analysis, diff, kernel,
            mask=masks_rad, ymin=-2, ymax=2, ymin_bar=-1, ymax_bar=1,
            vline=roiRad, statColor=diff_color, pval_threshold=pthresh,
            radMax=maxRad, nRad=nRad, comparisons=comparisons,
            depth_labels=rad_depth_labels, depthBoundaries=rad_depthBoundaries,
            figsize=figsize, ax_width=ax_width, ax_height=ax_height,
            ax_height_bar=ax_height_bar, ax_spacing=ax_spacing,
            ax_left=ax_left, ax_bottom=ax_bottom, ax_subspacing=ax_subspacing,
        )
        if savefigs:
            fig.savefig(os.path.join(figDir,
                        f"radial_profiles_{analysis}_{diff}_wbins.{fig_format}"))
            if 'corrected p-vals' in avgRadialDiff[analysis][d][diff].keys():
                for d in rad_depth_labels:
                    avgRadialDiff[analysis][d][diff]['rad (sigma)'] = np.arange(
                        maxRad/(2*nRad), maxRad, maxRad/nRad)
                    if not os.path.exists(os.path.join(statsDir, 'radial')):
                            os.makedirs(os.path.join(statsDir, 'radial'))
                    save_statistical_results(
                        avgRadialDiff[analysis][d][diff],
                        pthresh, statTestType, Npermutations=Npermutations,
                        output_csv=os.path.join(statsDir, 'radial',
                                                f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}.csv"),
                        binType='rad (sigma)',
                    )
                    if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                        if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                            if not os.path.exists(os.path.join(statsDir, 'radial')):
                                    os.makedirs(os.path.join(statsDir, 'radial'))
                            save_2samp_results(
                                avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff],
                                np.arange(maxRad/(2*nRad), maxRad, maxRad/nRad),
                                pthresh, statTestType, Npermutations=Npermutations,
                                output_csv=os.path.join(statsDir, 'radial',
                                                        f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}_multiComp.csv"),
                                binType='rad (sigma)',
                            )

figsize       = (8, 7)
ax_width      = 0.7
ax_height     = 0.4
ax_height_bar = 0.3
ax_spacing    = 0.3
ax_left       = 0.15
ax_bottom     = 0.12
skip_diffs    = {'task': ["iso-sur"], 'loc': []}
for analysis in ['task', 'loc']:
    STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                        stat_analyses, diff_analyses, analysis)
    diff_list = list(DIFFS['statIDs'].keys())
    for diff_key in skip_diffs[analysis]:
        diff_list.remove(diff_key)

    for diff in diff_list:
        included_data = return_included_subj(subj_analyses, analysis)
        STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                            stat_analyses, diff_analyses, analysis)
        diff_id    = list(DIFFS['statIDs'].keys()).index(diff)
        diff_color = DIFFS['colors'][diff_id]

        comparisons = {}
        for d in rad_depth_labels:
            if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                    comparisons[d] = (
                        avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff]['corrected p-vals']
                    )

        masks_rad = {label: (included_data[label]['no_vein'] &
                              included_data[label]['sig'] &
                              (included_data[label]['Visual Region'] == 'V1'))
                     for label in included_data.keys()}
        fig = plot_smoothed_radial_profile_wbins(
            included_data, avgRadialDiff[analysis], analysis, diff, kernel,
            mask=masks_rad, ymin=-2, ymax=2, ymin_bar=-1, ymax_bar=1,
            vline=roiRad, statColor=diff_color, pval_threshold=pthresh,
            radMax=maxRad, nRad=nRad, comparisons=comparisons,
            depth_labels=rad_depth_labels, depthBoundaries=rad_depthBoundaries,
            figsize=figsize, ax_width=ax_width, ax_height=ax_height,
            ax_height_bar=ax_height_bar, ax_spacing=ax_spacing,
            ax_left=ax_left, ax_bottom=ax_bottom, ax_subspacing=ax_subspacing,
        )
        if savefigs:
            fig.savefig(os.path.join(figDir,
                        f"radial_profiles_{analysis}_{diff}_wbins.{fig_format}"))
            if 'corrected p-vals' in avgRadialDiff[analysis][d][diff].keys():
                for d in rad_depth_labels:
                    avgRadialDiff[analysis][d][diff]['rad (sigma)'] = np.arange(
                        maxRad/(2*nRad), maxRad, maxRad/nRad)
                    if not os.path.exists(os.path.join(statsDir, 'radial')):
                        os.makedirs(os.path.join(statsDir, 'radial'))
                    save_statistical_results(
                        avgRadialDiff[analysis][d][diff],
                        pthresh, statTestType, Npermutations=Npermutations,
                        output_csv=os.path.join(statsDir, 'radial',
                                                f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}.csv"),
                        binType='rad (sigma)',
                    )
                    if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                        if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                            if not os.path.exists(os.path.join(statsDir, 'radial')):
                                os.makedirs(os.path.join(statsDir, 'radial'))
                            save_2samp_results(
                                avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff],
                                np.arange(maxRad/(2*nRad), maxRad, maxRad/nRad),
                                pthresh, statTestType, Npermutations=Npermutations,
                                output_csv=os.path.join(statsDir, 'radial',
                                                        f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}_multiComp.csv"),
                                binType='rad (sigma)',
                            )

# %% ~ Save Analyzed Data ~

if not os.path.exists('analyzed_data'):
    os.makedirs('analyzed_data')

for key, df in all_data.items():
    df.to_csv(f'analyzed_data/{key}.csv', index=False)

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

vArea        = 'V1'
subjIDs_V1   = [subj for subj in all_data.keys()
                if all_data[subj]['Visual Region'].str.contains(vArea).any()]

for task_dict in depthProfiles['in_tgt']:
    for cond_dict in depthProfiles['in_tgt'][task_dict]:
        depthProfiles['in_tgt'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in diffProfiles['in_tgt']:
    for cond_dict in diffProfiles['in_tgt'][task_dict]:
        diffProfiles['in_tgt'][task_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in radialProfiles:
    for depth_dict in radialProfiles[task_dict]:
        for cond_dict in radialProfiles[task_dict][depth_dict]:
            radialProfiles[task_dict][depth_dict][cond_dict]['subjIDs'] = subjIDs_V1
for task_dict in radialDiffProfiles:
    for depth_dict in radialDiffProfiles[task_dict]:
        for cond_dict in radialDiffProfiles[task_dict][depth_dict]:
            radialDiffProfiles[task_dict][depth_dict][cond_dict]['subjIDs'] = subjIDs_V1

with open("analyzed_data/DepthProfiles.json", 'w') as outfile:
    json.dump(convert_numpy_to_list(depthProfiles), outfile)

with open("analyzed_data/DepthDiffProfiles.json", 'w') as outfile:
    json.dump(convert_numpy_to_list(diffProfiles), outfile)

with open("analyzed_data/avgDepthProfiles.json", 'w') as outfile:
    json.dump(convert_numpy_to_list(avgDepthProfiles), outfile)

with open("analyzed_data/avgDepthDiffProfiles.json", 'w') as outfile:
    json.dump(convert_numpy_to_list(avgDepthDiffs), outfile)

with open("analyzed_data/RadialProfiles.json", 'w') as outfile:
    json.dump(convert_numpy_to_list(radialProfiles), outfile)

with open("analyzed_data/RadialDiffProfiles.json", 'w') as outfile:
    json.dump(convert_numpy_to_list(radialDiffProfiles), outfile)

with open("analyzed_data/avgRadialProfiles.json", 'w') as outfile:
    json.dump(convert_numpy_to_list(avgRadialProfiles), outfile)

with open("analyzed_data/avgRadialDiffProfiles.json", 'w') as outfile:
    json.dump(convert_numpy_to_list(avgRadialDiff), outfile)

# %% Run depth_visualizations.py

depth_viz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'depth_visualizations.py')
subprocess.run(
    [sys.executable, depth_viz_path],
    check=True,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)

# %%
