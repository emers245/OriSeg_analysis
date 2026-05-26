#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OriSeg Full Analysis

Runs the complete laminar depth analysis pipeline and produces all figures
for the manuscript. QC/exploratory visualizations are delegated to
QC_visualizations.py. Deveining logic lives in deveining.py.
Statistical correction and result-saving utilities live in run_statistics.py.
Condition/contrast configuration is loaded from analysis_config.json.
Depth visualizations are produced by depth_visualizations.py, which is
executed as a subprocess at the end of this script.
"""
# %%
import os
import glob
import sys
import subprocess
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

# Add func dir to path
funcs_dir = '..'
if funcs_dir not in sys.path:
    sys.path.append(funcs_dir)

from oriseg_funcs import *
from deveining import devein_voxels
from run_statistics import (apply_fdr_correction, save_statistical_results,
                             save_2samp_results, save_as_df)
from QC_visualizations import (
    plot_t1w_profiles, plot_ellipse_qc, plot_pval_histograms,
    plot_depth_histograms, plot_deveining_qc, plot_mnv_threshold_comparison,
    plot_voxel_count_depth_radius, plot_bold_per_subject, plot_bold_summary,
    plot_centroids_qc, plot_individual_condition_profiles,
    plot_individual_diff_profiles, plot_individual_radial_profiles,
    plot_diff_radial_profiles_subplot, plot_radial_analysis,
)

plt.close('all')

fcolor = 'white'
lcolor = 'black'
savefigs = True
mainDir = '../..'
figDir  = mainDir + '/figs/subjAvg/'
fig_format = 'svg'
fig_size   = "small"  # "small" or "large"

if savefigs and not os.path.exists(figDir):
    os.makedirs(figDir)
if 'stats' not in os.listdir(figDir):
    os.makedirs(os.path.join(figDir, 'stats'))

np.random.seed(68752)

# %% ~ Load Config ~

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_config.json')) as f:
    _cfg = json.load(f)
statDetails  = _cfg['statDetails']
diffDetails  = _cfg['diffDetails']
stat_analyses = _cfg['stat_analyses']
diff_analyses = _cfg['diff_analyses']

# %% ~ Set Parameters ~

# Radial distance parameters
roiRad     = 1.
centerRad  = 1.
borderRad  = [1., 3.]
surRad     = [3.]
nRad       = 5
maxRad     = 5
rad_depth_labels   = ['all depths']
rad_depthBoundaries = np.array([[0, 1]])
radBin_comparisons  = [[0, 4]]

# Depth parameters
nDepths      = 7
nDepths_rings = 3
nDepths_gPPI  = 3

# Mask parameters
use_fullmodel_mask  = True
use_loc_mask        = False
useSI               = False
pthresh_fullmodel   = 0.01
pthresh_loc         = 0.01

# Statistics parameters
pthresh          = 0.05
prop_err         = False
showSig          = True
compareRadtoNull = False
statCorrType     = 'fdr_bh'
statTestType     = 'permutation'
Npermutations    = 10000

# Depth deconvolution
use_decon = {'task': True, 'loc': True, 'gPPI': False}
p2t_model     = 6.3
Nbins_model   = 10
normalize_psf = False

# Subject inclusion
subj_analyses = {
    'task': 'all',
    'loc':  'all',
    'gPPI': ['pnr256', 'pnr328', 'pnr495', 'pnr510', 'pnr579', 'pnr668',
             'pnr685', 'pnr713', 'pnr739', 'pnr756', 'pnr822'],
}


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

mainDir      = '.'
datasets_V1  = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual',
                                       'pnr???_??_???_??.csv'))
datasets_V23 = glob.glob(os.path.join(mainDir, 'roi_data_manualSeg/V23_roi_manual',
                                       'pnr???_???_???_??.csv'))

exclude_V1 = [
    'pnr143_V1_tgt_rh',
    'pnr161_V1_tgt_lh', 'pnr161_V1_tgt_rh',
    'pnr352_V1_tgt_lh', 'pnr352_V1_tgt_rh',
    'pnr579_V1_tgt_lh',
    'pnr668_V1_tgt_rh',
]
exclude_V23 = []
for e_i, excl in enumerate(exclude_V1):
    datasets_V1.remove(
        os.path.join(mainDir, 'roi_data_manualSeg/target_roi_manual', excl+'.csv'))
datasets_V1.sort()
Ndsets_V1 = len(datasets_V1)
print(f"{Ndsets_V1} V1 ROIs")
for e_i, excl in enumerate(exclude_V23):
    datasets_V23.remove(
        os.path.join(mainDir, 'roi_data_manualSeg/V23_roi_manual', excl+'.csv'))
datasets_V23.sort()
Ndsets_V23 = len(datasets_V23)
print(f"{Ndsets_V23} V2/3 ROIs")

datasets = datasets_V1 + datasets_V23
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

plot_t1w_profiles(all_data, roiRad, savefigs, figDir, fig_format)

# %% Ellipse computation (modifies all_data; needed for downstream analysis)

from matplotlib.patches import Ellipse

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

            df     = df_all[(df_all['Visual Region'] == 'V1') &
                            (df_all['Subregion'] == 'tgt') &
                            (df_all['hemi'] == hemi)]
            tgt_df = df[df['ctr-sur'] > 0]
            cov    = np.cov(tgt_df['x'][df['scale_xy_dist'] < 2.2],
                            tgt_df['y'][df['scale_xy_dist'] < 2.2])
            com    = (np.mean(tgt_df['x'][df['scale_xy_dist'] < 2.2]),
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
            df['ellipse_a']    = a
            df['ellipse_b']    = b
            df['ellipse_theta']  = theta
            df['ellipse_comX']   = com[0]
            df['ellipse_comY']   = com[1]
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
    ellipse_df.to_csv(os.path.join(figDir, 'stats', 'ROIstats.csv'))
    ellipse_stats.to_csv(os.path.join(figDir, 'stats', 'ROIsummary.csv'))

plot_ellipse_qc(all_data, ellipse_df, roiRad, Ndsets_V1, savefigs, figDir, fig_format)

# %% p-value and depth histograms (QC)

subject_visArea_combinations = [
    (subjID, vis_region)
    for subjID, df in all_data.items()
    for vis_region in df['Visual Region'].unique()
]

plot_pval_histograms(all_data, roiRad, subject_visArea_combinations,
                     savefigs, figDir, fig_format)
plot_depth_histograms(all_data, roiRad, nDepths, subject_visArea_combinations,
                      savefigs, figDir, fig_format)

# %% ~ Analysis Preprocessing ~

# %% Deveining

lmnv_dict = devein_voxels(
    all_data,
    depth_var='d_norm',
    deep_pct=10,
    sd_thresh=2,
    out_dir=os.path.join(figDir, 'stats'),
)

# Deveining QC plots
plot_deveining_qc(all_data, lmnv_dict, nDepths,
                  savefigs=savefigs, figDir=figDir, fig_format=fig_format)
plot_mnv_threshold_comparison(all_data, lmnv_dict,
                               savefigs=savefigs, figDir=figDir, fig_format=fig_format)

# %% Apply full model p-val mask if desired

Nsubj = len([subj for subj in all_data.keys()
             if all_data[subj]['Visual Region'].str.contains('V23').any()])

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
    fullmodel_thresh_df.to_csv(os.path.join(figDir, 'stats', 'task_pval_mask.csv'))
    fullmodel_thresh_stats.to_csv(os.path.join(figDir, 'stats', 'task_pval_mask_summary.csv'))

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
    loc_thresh_df.to_csv(os.path.join(figDir, 'stats', 'task_pval_mask.csv'))
    loc_thresh_stats.to_csv(os.path.join(figDir, 'stats', 'task_pval_mask_summary.csv'))

# Voxel count QC
plot_voxel_count_depth_radius(all_data, nDepths_rings, savefigs, figDir, fig_format)

# %% Compute iso for localizer condition

for key in list(all_data.keys()):
    all_data[key]['iso0_loc'] = all_data[key]['iso0'] - all_data[key]['sur']

# %% ~ Get Depth Profiles ~

profile_method = 'bin'
for key in all_data.keys():
    df = all_data[key][all_data[key]['Visual Region'] == 'V1']
    all_data[key].loc[all_data[key]['Visual Region'] == 'V1', 'in_tgt'] = df['scale_xy_dist'] < roiRad
    all_data[key].loc[all_data[key]['Visual Region'] == 'V1', 'in_ctr'] = df['scale_xy_dist'] < centerRad
    all_data[key].loc[all_data[key]['Visual Region'] == 'V1', 'in_bor'] = (
        (df['scale_xy_dist'] >= borderRad[0]) & (df['scale_xy_dist'] < borderRad[1])
    )
    if len(surRad) == 1:
        sur_mask = df['scale_xy_dist'] > surRad[0]
    elif len(surRad) > 1:
        sur_mask = ((df['scale_xy_dist'] >= surRad[0]) & (df['scale_xy_dist'] <= surRad[1]))
    all_data[key].loc[all_data[key]['Visual Region'] == 'V1', 'in_sur'] = sur_mask

for key in all_data.keys():
    all_data[key]['in_V23'] = all_data[key]['Visual Region'] == 'V23'

masks = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}
for roi in masks.keys():
    masks[roi] = {key: all_data[key][roi]*all_data[key]['sig']*all_data[key]['no_vein']
                  for key in all_data.keys()}

depthProfiles = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}
diffProfiles  = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}

for roi in depthProfiles.keys():
    if roi == 'in_tgt':
        nD           = nDepths
        radialParam  = 'scale_xy_dist'
    elif roi == 'in_V23':
        nD           = nDepths_gPPI
        radialParam  = None
    else:
        nD           = nDepths_rings
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
            csv_filepath = os.path.join(figDir, 'stats', csv_filename)
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

print(f"CSV files have been saved in '{os.path.join(figDir, 'stats')}' directory.")

# %% Total % BOLD change (QC)

plot_bold_per_subject(all_data, statDetails, savefigs, figDir, fig_format)
plot_bold_summary(all_data, statDetails, savefigs, figDir, fig_format)

# %% Deconvolution

for roi in depthProfiles.keys():

    if roi == 'in_tgt':
        nD = nDepths
    elif roi == 'in_V23':
        nD = nDepths_gPPI
    else:
        nD = nDepths_rings

    for analysis in ['task', 'loc']:
        if roi != 'in_V23':
            included_data = return_included_subj(subj_analyses, analysis)
        else:
            included_data = return_included_subj(subj_analyses, 'gPPI')

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

avgDepthProfiles = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}
avgDepthDiffs    = {'in_tgt': {}, 'in_ctr': {}, 'in_bor': {}, 'in_sur': {}, 'in_V23': {}}

for roi in depthProfiles.keys():
    if roi == 'in_tgt':
        nD = nDepths
    elif roi == 'in_V23':
        nD = nDepths
    else:
        nD = nDepths_rings

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

# %% Centroid plots (QC)

for analysis in ['task', 'loc']:
    included_data = return_included_subj(subj_analyses, ['task', 'loc'])
    STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                        stat_analyses, diff_analyses, analysis)
    plot_centroids_qc(included_data, masks['in_tgt'], STATS, DIFFS,
                      roiRad, nDepths, savefigs, figDir, fig_format, analysis)

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

# %% Get null radial profiles

radialProfiles_null     = {analysis_type: {} for analysis_type in subj_analyses.keys()}
radialDiffProfiles_null = {analysis_type: {} for analysis_type in subj_analyses.keys()}
avgRadialProfiles_null  = {analysis_type: {} for analysis_type in subj_analyses.keys()}
avgRadialDiff_null      = {analysis_type: {} for analysis_type in subj_analyses.keys()}

for analysis in subj_analyses.keys():
    included_data = return_included_subj(subj_analyses, analysis)
    STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                        stat_analyses, diff_analyses, analysis)

    layer_mask_dict = {l: {} for l in rad_depth_labels}
    included_data[d_label]['scale_xy_dist_scrambled'] = None
    for iR, d_label in enumerate(included_data.keys()):
        for iB, dB in enumerate(rad_depthBoundaries):
            df       = all_data[d_label]
            lmask    = (df['d'] > dB[0]) & (df['d'] < dB[1])
            roi_mask = (included_data[d_label]['no_vein'] & included_data[d_label]['sig'])
            df[rad_depth_labels[iB]] = lmask
            layer_mask_dict[rad_depth_labels[iB]][d_label] = lmask & roi_mask

            vArea_mask = included_data[d_label]['Visual Region'] == 'V1'
            included_data[d_label].loc[lmask & roi_mask & vArea_mask,
                                       'scale_xy_dist_scrambled'] = np.random.permutation(
                included_data[d_label].loc[lmask & roi_mask & vArea_mask, 'scale_xy_dist']
            )

    for il, l in enumerate(rad_depth_labels):
        radialProfiles_null[analysis][l] = compute_all_rad_profiles(
            included_data, STATS, 'bin', nRad, layer_mask_dict[l],
            radParam='scale_xy_dist_scrambled', spec_Drange=[0, maxRad], radMax=maxRad,
        )
        radialDiffProfiles_null[analysis][l] = compute_rad_diff_profiles(
            included_data, STATS, DIFFS['statIDs'], 'bin', nRad, prop_err,
            layer_mask_dict[l],
            radParam='scale_xy_dist_scrambled', spec_Drange=[0, maxRad], radMax=maxRad,
        )
        avgRadialProfiles_null[analysis][l], avgRadialDiff_null[analysis][l] = compute_avg_rad_profile(
            radialProfiles_null[analysis][l], STATS, DIFFS['statIDs'],
            STATS['labels'], list(DIFFS['statIDs'].keys()),
            prop_err, useSI, statTestType=statTestType, npermSamples=Npermutations,
        )

# %% Compare radial profiles to null

if compareRadtoNull:
    experiment        = 'task'
    cortical_depths   = ['deep', 'middle', 'superficial']
    condition_contrasts = ['odss', 'fgm']
    radial_bins_tested  = [0, 1, 2, 3, 4]

    for depth in cortical_depths:
        for condition in condition_contrasts:
            for b_i in range(nRad):
                if b_i in radial_bins_tested:
                    data_task = radialDiffProfiles[experiment][depth][condition]['avg']
                    null_task = radialDiffProfiles_null[experiment][depth][condition]['avg']
                    data_distribution = [subject_data[b_i] for subject_data in data_task]
                    null_distribution = [subject_null[b_i] for subject_null in null_task]

                    def statistic(data, null, axis):
                        return np.mean(data, axis=axis) - np.mean(null, axis=axis)

                    perm_result = stats.permutation_test(
                        (data_distribution, null_distribution),
                        statistic,
                        permutation_type='independent',
                        n_resamples=Npermutations,
                        alternative='two-sided',
                    )
                    p_value = perm_result.pvalue

                    if 'p-vals vs null' not in avgRadialDiff[experiment][depth][condition]:
                        avgRadialDiff[experiment][depth][condition]['p-vals vs null'] = np.array([])
                    avgRadialDiff[experiment][depth][condition]['p-vals vs null'][b_i] = p_value
                else:
                    if 'p-vals vs null' not in avgRadialDiff[experiment][depth][condition]:
                        avgRadialDiff[experiment][depth][condition]['p-vals vs null'] = np.array([])
                    avgRadialDiff[experiment][depth][condition]['p-vals vs null'][b_i] = np.nan

    print("Permutation tests completed and p-values saved.")

# %% Multisample comparisons

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

exclude_indiv_ttest     = True
exclude_analysis_types  = ['gPPI']
exclude_rois            = []
exclude_cond = ['iso0', 'iso90', 'orth', 'sur',
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
    ('in_bor', 'gPPI', 'all'),
    ('in_sur', 'gPPI', 'all'),
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

for roi_type in ['in_tgt', 'in_V23']:
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
                        save_statistical_results(
                            avgDepthDiffs[roi_type][analysis][diff],
                            pthresh, statTestType, Npermutations=Npermutations,
                            output_csv=os.path.join(figDir, 'stats',
                                                     'avg_profiles_%s_%s_%s_%s_deconv.csv'
                                                     % (analysis, roi_type, diff, statTestType)),
                        )
                else:
                    fig.savefig(os.path.join(figDir, 'avg_profiles_%s_%s_%s.%s'
                                             % (analysis, roi_type, diff, fig_format)))
                    if 'corrected p-vals' in avgDepthDiffs[roi_type][analysis][diff].keys():
                        save_statistical_results(
                            avgDepthDiffs[roi_type][analysis][diff],
                            pthresh, statTestType, Npermutations=Npermutations,
                            output_csv=os.path.join(figDir, 'stats',
                                                     'avg_profiles_%s_%s_%s_%s.csv'
                                                     % (analysis, roi_type, diff, statTestType)),
                        )

# %% Individual subject depth profiles (QC)

plot_individual_condition_profiles(
    depthProfiles, avgDepthProfiles,
    statDetails, diffDetails, stat_analyses, diff_analyses,
    use_decon, fig_size, savefigs, figDir, fig_format, fcolor, lcolor,
)

plot_individual_diff_profiles(
    diffProfiles, avgDepthDiffs,
    statDetails, diffDetails, stat_analyses, diff_analyses,
    use_decon, fig_size, pthresh, savefigs, figDir, fig_format, fcolor, lcolor,
)

# %% Loc Profiles across surface

def gaussian_kernel(x, k, xloc):
    kernel = (1/(np.sqrt(2*np.pi)*k)) * np.exp(-((x-xloc)**2) / (2*k**2))
    return kernel / np.sum(kernel)

def plot_smoothed_radial_profile(data, analysis_type, condition, kernel,
                                  mask=None, smooth_factor=0.3, radMax=4,
                                  nRadii=20, ymin=-5, ymax=5, fontsize=8,
                                  fcolor='white', lcolor='black',
                                  depth_labels=['deep', 'middle', 'superficial'],
                                  depthBoundaries=np.array([[0, 1/3], [1/3, 2/3], [2/3, 1]]),
                                  plot_indiv=True, statColor='gray', vline=2):
    all_profiles = {depth_label: {} for depth_label in depth_labels}

    for iR, label in enumerate(data.keys()):
        df = data[label]
        if mask:
            df = df[mask[label]]
        for iD, depth_label in enumerate(depth_labels):
            depth_df = df[(df['d'] >= depthBoundaries[iD, 0]) &
                          (df['d'] <  depthBoundaries[iD, 1])]
            coef = depth_df[condition].values
            x    = depth_df['scale_xy_dist'].values
            coef_smooth, x_smooth = smoothen(coef, x, kernel=kernel,
                                              smooth_factor=smooth_factor, radMax=maxRad)
            all_profiles[depth_label][label] = coef_smooth

    fig = plt.figure(figsize=(5, 6))
    fig.patch.set_facecolor(fcolor)
    for iD, depth_label in enumerate(depth_labels):
        p = fig.add_axes([.15, .1 + iD * .3, .7, .2])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        layer_profiles_list = list(all_profiles[depth_label].values())
        stat_avg = np.mean(np.vstack(layer_profiles_list), axis=0)
        stat_std = np.std(np.vstack(layer_profiles_list),  axis=0)

        p.plot(x_smooth, stat_avg, color=statColor, label=depth_label)
        p.fill_between(x_smooth,
                       stat_avg - stat_std / np.sqrt(len(all_profiles[depth_label])),
                       stat_avg + stat_std / np.sqrt(len(all_profiles[depth_label])),
                       alpha=0.4, color=statColor)

        p.set_ylim([ymin, ymax])
        p.set_ylabel('BOLD % change', fontsize=fontsize, color=lcolor)
        p.set_title(depth_label)
        if plot_indiv:
            for iR, label in enumerate(data.keys()):
                p.plot(x_smooth.T, np.array(all_profiles[depth_label][label]).T,
                       color=statColor, alpha=0.2)
        if iD == 0:
            p.set_xlabel('relative distance from ROI center ($\\sigma$)',
                         fontsize=fontsize, color=lcolor)
        p.plot([vline, vline], [-6, 6], '--', color='black')
        p.plot([0, radMax], [0, 0], '--', color='black')
        p.legend([condition])

    return fig

kernel        = smooth_kernel
smooth_factor = 0.3
nRadii        = 20
ymax          = 7
ymin          = -7
highlight     = False

for analysis in ['task', 'loc', 'gPPI']:
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

# %% Individual subject radial profiles (QC)

plot_individual_radial_profiles(
    all_data, subj_analyses, statDetails, diffDetails,
    stat_analyses, diff_analyses,
    kernel, rad_depth_labels, rad_depthBoundaries, maxRad,
    savefigs, figDir, fig_format,
    return_included_subj_fn=return_included_subj,
)

# %% Compute differences and put them back in all_data

for analysis in ['task', 'loc', 'gPPI']:
    included_data = return_included_subj(subj_analyses, analysis)
    STATS, DIFFS  = return_stats_diffs(statDetails, diffDetails,
                                        stat_analyses, diff_analyses, analysis)
    for diff in DIFFS['statIDs'].keys():
        for iR, label in enumerate(included_data.keys()):
            stat1 = DIFFS['statIDs'][diff][0]
            stat2 = DIFFS['statIDs'][diff][1]
            included_data[label][diff] = included_data[label][stat1] - included_data[label][stat2]

# %% Diff radial profiles (QC)

plot_diff_radial_profiles_subplot(
    all_data, subj_analyses, statDetails, diffDetails,
    stat_analyses, diff_analyses,
    kernel, rad_depth_labels, rad_depthBoundaries,
    savefigs, figDir, fig_format,
    return_included_subj_fn=return_included_subj,
)

# %% Add Binned Radial Data

def plot_smoothed_radial_profile_wbins(data, avgRadialProfiles, analysis_type,
                                        condition, kernel, mask=None,
                                        smooth_factor=0.3, radMax=4, nRadii=20,
                                        ymin=-5, ymax=5, ymin_bar=-0.5, ymax_bar=3.0,
                                        fontsize=8, fcolor='white', lcolor='black',
                                        depth_labels=['deep', 'middle', 'superficial'],
                                        depthBoundaries=np.array([[0, 1/3], [1/3, 2/3], [2/3, 1]]),
                                        plot_indiv=True, statColor='gray', vline=2,
                                        pval_threshold=0.05, nRad=4, comparisons=None,
                                        figsize=(8, 10), ax_width=0.7, ax_height=0.15,
                                        ax_height_bar=0.06, ax_spacing=0.3, ax_left=0.15,
                                        ax_bottom=0.15, ax_subspacing=0.05):
    import numpy as np
    import matplotlib.pyplot as plt

    all_profiles = {depth_label: {} for depth_label in depth_labels}

    for iR, label in enumerate(data.keys()):
        df = data[label]
        if mask:
            df = df[mask[label]]
        for iD, depth_label in enumerate(depth_labels):
            depth_df = df[(df['d'] >= depthBoundaries[iD, 0]) &
                          (df['d'] <  depthBoundaries[iD, 1])]
            coef = depth_df[condition].values
            x    = depth_df['scale_xy_dist'].values
            coef_smooth, x_smooth = smoothen(coef, x, kernel=kernel,
                                              smooth_factor=smooth_factor, radMax=radMax)
            all_profiles[depth_label][label] = coef_smooth

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(fcolor)
    for iD, depth_label in enumerate(depth_labels):
        p = fig.add_axes([ax_left,
                          ax_bottom + ax_height_bar + iD*ax_spacing + ax_subspacing,
                          ax_width, ax_height])
        fix_axes(p, lcolor=lcolor, fcolor=fcolor)
        layer_profiles_list = list(all_profiles[depth_label].values())
        stat_avg = np.mean(np.vstack(layer_profiles_list), axis=0)
        stat_std = np.std(np.vstack(layer_profiles_list),  axis=0)

        p.plot(x_smooth, stat_avg, color=statColor, label=depth_label)
        p.fill_between(x_smooth,
                       stat_avg - stat_std / np.sqrt(len(all_profiles[depth_label])),
                       stat_avg + stat_std / np.sqrt(len(all_profiles[depth_label])),
                       alpha=0.4, color=statColor)

        p.set_ylim([ymin, ymax])
        p.set_ylabel('BOLD % change', fontsize=fontsize, color=lcolor)
        p.set_title(depth_label, fontsize=fontsize)
        if plot_indiv:
            for iR, label in enumerate(data.keys()):
                p.plot(x_smooth.T, np.array(all_profiles[depth_label][label]).T,
                       color=statColor, alpha=0.2)
        p.plot([vline, vline], [ymin, ymax], '--', color='black')
        p.plot([0, radMax], [0, 0], '--', color='black')
        p.set_xlim([0, radMax])
        p.set_xticks(np.linspace(0, maxRad, nRad+1), [])
        p.legend([condition])

        bar_p = fig.add_axes([ax_left, ax_bottom + iD*ax_spacing, ax_width, ax_height_bar])
        fix_axes(bar_p, lcolor=lcolor, fcolor=fcolor)
        bar_data    = avgRadialProfiles[depth_label][condition]['avg']
        bar_std     = (avgRadialProfiles[depth_label][condition]['stdev'] /
                       np.sqrt(avgRadialProfiles[depth_label][condition]['Nsamp']))
        bar_pvals   = avgRadialProfiles[depth_label][condition].get('p-vals', None)
        corrected_pvals = avgRadialProfiles[depth_label][condition].get('corrected p-vals', None)

        x_ticks = np.linspace(0+(radMax/(2*nRad)), radMax-(radMax/(2*nRad)), nRad)
        bar_p.bar(x_ticks, bar_data, width=0.9, yerr=bar_std, capsize=5,
                  color=statColor, alpha=0.6)
        bar_p.set_ylabel('Binned Avg', fontsize=fontsize, color=lcolor)
        bar_p.set_ylim([ymin_bar, ymax_bar])
        bar_p.set_xlim([0, radMax])
        if iD == 0:
            bar_p.set_xlabel('relative distance from ROI center ($\\sigma$)',
                             fontsize=fontsize, color=lcolor)
            bar_p.set_xticks(np.linspace(0, maxRad, nRad+1))
        else:
            bar_p.set_xticks(np.linspace(0, maxRad, nRad+1), [])

        if comparisons and depth_label in comparisons:
            comp_matrix = comparisons[depth_label]
            for row in range(comp_matrix.shape[0]):
                for col in range(row + 1, comp_matrix.shape[1]):
                    pval = comp_matrix[row, col]
                    if np.isfinite(pval) and pval < pval_threshold:
                        x1, x2 = x_ticks[row], x_ticks[col]
                        y_max   = max(bar_data[row] + bar_std[row],
                                      bar_data[col] + bar_std[col]) + 0.5
                        bar_p.plot([x1, x1, x2, x2],
                                   [y_max, y_max+0.1, y_max+0.1, y_max], lw=1.5, color='k')
                        bar_p.text((x1+x2)/2, y_max+0.05, '*', ha='center',
                                   color='k', fontsize=fontsize+4)

        if type(bar_pvals) == np.ndarray:
            bar_pvals_list = list(bar_pvals)
        else:
            bar_pvals_list = list(bar_pvals.pvalue)
        for i, pval in enumerate(bar_pvals_list):
            if corrected_pvals is not None and corrected_pvals[i] < pval_threshold:
                bar_p.text(x_ticks[i], bar_data[i] + bar_std[i] + 0, '*',
                           ha='center', va='bottom', color='k', fontsize=fontsize+4)
            elif pval < pval_threshold:
                bar_p.text(x_ticks[i], bar_data[i] + bar_std[i] + 0, '*',
                           ha='center', va='bottom', color='k', fontsize=fontsize+4)
        bar_p.plot([0, radMax], [0, 0], '--', color='black')

    return fig

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
                    save_statistical_results(
                        avgRadialDiff[analysis][d][diff],
                        pthresh, statTestType, Npermutations=Npermutations,
                        output_csv=os.path.join(figDir, 'stats',
                                                f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}.csv"),
                        binType='rad (sigma)',
                    )
                    if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                        if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                            save_2samp_results(
                                avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff],
                                np.arange(maxRad/(2*nRad), maxRad, maxRad/nRad),
                                pthresh, statTestType, Npermutations=Npermutations,
                                output_csv=os.path.join(figDir, 'stats',
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
skip_diffs    = {'task': ["iso-sur"], 'loc': [], 'gPPI': []}
for analysis in ['task', 'loc', 'gPPI']:
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
                    save_statistical_results(
                        avgRadialDiff[analysis][d][diff],
                        pthresh, statTestType, Npermutations=Npermutations,
                        output_csv=os.path.join(figDir, 'stats',
                                                f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}.csv"),
                        binType='rad (sigma)',
                    )
                    if 'withinCondition' in avgRadialDiff_comparisons[analysis]:
                        if diff in avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d]:
                            save_2samp_results(
                                avgRadialDiff_comparisons[analysis]['withinCondition']['across_rad'][d][diff],
                                np.arange(maxRad/(2*nRad), maxRad, maxRad/nRad),
                                pthresh, statTestType, Npermutations=Npermutations,
                                output_csv=os.path.join(figDir, 'stats',
                                                        f"radial_profiles_{analysis}_{diff}_{d}_wbins_{statTestType}_multiComp.csv"),
                                binType='rad (sigma)',
                            )

# %% gPPI matrices

def return_gPPI_matrices(data_dict, analysis_type, roi_label, seed, condition,
                          isDiff=False, transpose=False):
    depth_bins  = ['deep', 'middle', 'superficial']
    avg_matrix  = np.zeros((3, 3))
    pval_matrix = np.zeros((3, 3))

    for i, seed_depth in enumerate(depth_bins):
        if not isDiff:
            key = (f"{seed}_{seed_depth}_deveined" if not condition
                   else f"{seed}_{seed_depth}_deveined_{condition}")
        else:
            key = (f"gPPI_{seed_depth}_{seed}" if not condition
                   else f"{condition}_gPPI_{seed_depth}_{seed}")

        avg  = data_dict[roi_label][analysis_type][key]['avg']
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
            avg_matrix[i, :]  = avg
            pval_matrix[i, :] = pval
        else:
            avg_matrix[:, i]  = avg
            pval_matrix[:, i] = pval

    return avg_matrix, pval_matrix


def plot_gPPI_mat(data_mat, p_mat, ROIx, ROIy, seed=None, target=None, Nvox=None,
                  cbar_lims=[-2.0, 2.0], title_add='', fig=None, ax=None,
                  fontsize=12, invert_yaxis=True, cbar=True):
    if not seed:
        seed = ROIx
    if not target:
        target = ROIy
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    depth_labels = ['deep', 'middle', 'superficial']
    heatmap = ax.imshow(data_mat, cmap='bwr', interpolation='nearest',
                        vmin=cbar_lims[0], vmax=cbar_lims[1])

    for i in range(data_mat.shape[0]):
        for j in range(data_mat.shape[1]):
            if p_mat[i][j] < 0.05 and p_mat[i][j] >= 0.01:
                ax.text(j, i, f"{data_mat[i,j]:.2f}%*\np={p_mat[i,j]:.3f}",
                        ha='center', va='center', color='black',
                        fontsize=0.6*fontsize, weight='bold')
            elif p_mat[i][j] < 0.01 and p_mat[i][j] >= 0.001:
                ax.text(j, i, f"{data_mat[i,j]:.2f}%**\np={p_mat[i,j]:.3f}",
                        ha='center', va='center', color='black',
                        fontsize=0.6*fontsize, weight='bold')
            elif p_mat[i][j] < 0.001:
                ax.text(j, i, f"{data_mat[i,j]:.2f}%**\np<0.001",
                        ha='center', va='center', color='black',
                        fontsize=0.6*fontsize, weight='bold')
            else:
                ax.text(j, i, f"{data_mat[i,j]:.2f}%\np={p_mat[i,j]:.2f}",
                        ha='center', va='center', color='black', fontsize=0.6*fontsize)

    if cbar:
        cbar_h = plt.colorbar(heatmap, ax=ax)
        cbar_h.set_label("% BOLD Change", fontsize=0.6*fontsize)
        cbar_h.ax.tick_params(labelsize=0.6*fontsize)
    ax.set_xticks(np.arange(data_mat.shape[1]))
    ax.set_yticks(np.arange(data_mat.shape[0]))
    if Nvox is None:
        ax.set_xticklabels(depth_labels, fontsize=0.6*fontsize)
        ax.set_yticklabels(depth_labels, fontsize=0.6*fontsize)
    else:
        ax.set_xticklabels(
            ['deep \n N=%d' % Nvox['seed'][0],
             'middle \n N=%d' % Nvox['seed'][1],
             'superficial \n N=%d' % Nvox['seed'][2]], fontsize=0.6*fontsize)
        ax.set_yticklabels(
            ['deep \n N=%d' % Nvox['targ'][0],
             'middle \n N=%d' % Nvox['targ'][1],
             'superficial \n N=%d' % Nvox['targ'][2]], fontsize=0.6*fontsize)
    ax.set_xlabel(ROIx, fontsize=0.8*fontsize)
    ax.set_ylabel(ROIy, fontsize=0.8*fontsize)
    ax.set_title('%s -> %s %s' % (seed, target, title_add), fontsize=fontsize)
    if invert_yaxis:
        ax.invert_yaxis()
    plt.show()
    return (heatmap, fig, ax)


plot_baseline = False
seedROI   = 'V23'
targetROI = 'V1'
roi_label = 'in_ctr'
ROIx      = 'V23'
ROIy      = 'V1'

transpose = seedROI != ROIx

if plot_baseline:
    fig, ax = plt.subplots(1, 5)
else:
    fig, ax = plt.subplots(1, 4)
fig.set_figwidth(14)
fig.set_figheight(6)
fontsize = 12

if plot_baseline:
    data_mat, p_mat = return_gPPI_matrices(avgDepthProfiles, 'gPPI', roi_label,
                                            seedROI, None, transpose=transpose)
    plot_gPPI_mat(data_mat, p_mat, ROIx, ROIy, seed=seedROI, target=targetROI,
                  fig=fig, ax=ax[0], cbar=False)

for iC, condition in enumerate(['iso0', 'iso90', 'orth', 'sur']):
    data_mat, p_mat = return_gPPI_matrices(avgDepthProfiles, 'gPPI', roi_label,
                                            seedROI, condition, transpose=transpose)
    plot_gPPI_mat(data_mat, p_mat, ROIx, ROIy, seed=seedROI, target=targetROI,
                  fig=fig, ax=ax[iC], cbar=False, title_add=condition)

plt.tight_layout()
if savefigs:
    fig.savefig(os.path.join(figDir,
                'gPPI_conditions_mat_seed=%s_target=%s.%s' % (seedROI, targetROI, fig_format)),
                format=fig_format)

plot_baseline = False
seedROI   = 'V23'
targetROI = 'V1'
roi_label = 'in_ctr'
ROIx      = 'V23'
ROIy      = 'V1'

transpose = seedROI != ROIx

if plot_baseline:
    fig, ax = plt.subplots(1, 3)
else:
    fig, ax = plt.subplots(1, 2)
fig.set_figwidth(14)
fig.set_figheight(6)
fontsize = 12

if plot_baseline:
    data_mat, p_mat = return_gPPI_matrices(avgDepthDiffs, 'gPPI', roi_label,
                                            seedROI, None, isDiff=True, transpose=transpose)
    plot_gPPI_mat(data_mat, p_mat, ROIx, ROIy, seed=seedROI, target=targetROI,
                  fig=fig, ax=ax[0], cbar=False)

for iC, condition in enumerate(['odss', 'fgm']):
    data_mat, p_mat = return_gPPI_matrices(avgDepthDiffs, 'gPPI', roi_label,
                                            seedROI, condition, isDiff=True, transpose=transpose)
    plot_gPPI_mat(data_mat, p_mat, ROIx, ROIy, seed=seedROI, target=targetROI,
                  fig=fig, ax=ax[iC], cbar=False, title_add=condition)

plt.tight_layout()
if savefigs:
    fig.savefig(os.path.join(figDir,
                'gPPI_diffs_mat_seed=%s_target=%s.%s' % (seedROI, targetROI, fig_format)),
                format=fig_format)

# %% Individual subjects radial analysis (QC)

plot_radial_analysis(all_data, 'ctr-sur_unwarp', nDepths_rings=nDepths_rings, plot_type='bar')

# %% ~ Save Analyzed Data ~

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
vArea        = 'V23'
subjIDs_V23  = [subj for subj in all_data.keys()
                if all_data[subj]['Visual Region'].str.contains(vArea).any()]

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
