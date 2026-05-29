#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC and exploratory visualizations for the OriSeg depth analysis.

All functions here are called from oriSeg_fullAnalysis.py and produce figures
used for data quality control and exploratory analysis that do not appear in
the publication. The computation in oriSeg_fullAnalysis.py is not altered by
calling or skipping any of these functions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import pandas as pd
import seaborn as sns

funcs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if funcs_dir not in sys.path:
    sys.path.append(funcs_dir)

from oriseg_funcs import (get_lmnv, get_deep_layer_dist, get_mnv_mask,
                           plot_mnv_histograms, plot_depth_maps,
                           plot_depth_voxel_loss, fix_axes,
                           plot_centroids, plot_centroids_diff,
                           plot_avg_depth_profile, plot_avg_diff_profile,
                           smooth_kernel, smoothen, makeProfile1D)


# ---------------------------------------------------------------------------
# T1w stria profiles
# ---------------------------------------------------------------------------

def plot_t1w_profiles(all_data, roiRad, savefigs, figDir, fig_format, nDepths=7):
    fig = plt.figure(num=1)
    fig.clf()
    for iR, label in enumerate(all_data.keys()):
        df = all_data[label]
        df = df[df['Visual Region'] == 'V1']
        df = df[df['Subregion'] == 'tgt']

        roi = df[df['scale_xy_dist'] < roiRad]
        roi = roi[roi['scale_xy_dist'] > 0]
        dataDict = makeProfile1D(roi['d'].values,
                                 nDepths,
                                 roi['t1'].values,
                                 np.min(roi['d'].values),
                                 np.max(roi['d'].values),
                                 True)

        plt.subplot(int(np.ceil(len(all_data.keys())/2.)), 2, 1 + iR)
        plt.plot(dataDict['profile']['depth'],
                 dataDict['profile']['avg'][0])
        plt.title('%s (%d vox)' % (label, len(roi)), fontsize=8)

    if savefigs:
        fig.savefig(os.path.join(figDir, 't1w_profiles.%s' % fig_format))


# ---------------------------------------------------------------------------
# Ellipse fit visualization (reads pre-computed params from all_data)
# ---------------------------------------------------------------------------

def plot_ellipse_qc(all_data, ellipse_df, roiRad, Ndsets_V1, savefigs, figDir, fig_format):
    frad = plt.figure(figsize=(6.5, 8))
    floc = plt.figure(figsize=(6.5, 8))
    iR   = 0

    cmap     = plt.cm.get_cmap('viridis')
    cmap_rev = cmap.reversed()

    for iS, label in enumerate(all_data.keys()):
        df_all = all_data[label]

        for iH, hemi in enumerate(df_all['hemi'].unique()):

            if np.sum((df_all['Visual Region'] == 'V1') &
                      (df_all['Subregion'] == 'tgt') &
                      (df_all['hemi'] == hemi)) != 0:

                df = df_all[(df_all['Visual Region'] == 'V1') &
                            (df_all['Subregion'] == 'tgt') &
                            (df_all['hemi'] == hemi)]

                a     = df['ellipse_a'].values[0]
                b     = df['ellipse_b'].values[0]
                theta = df['ellipse_theta'].values[0]
                com   = (df['ellipse_comX'].values[0], df['ellipse_comY'].values[0])

                ellipse = Ellipse(com,
                                  width=2*roiRad*np.sqrt(a),
                                  height=2*roiRad*np.sqrt(b),
                                  angle=180*theta/np.pi,
                                  zorder=100, alpha=1., edgecolor='r', facecolor='None')

                ax1 = frad.add_subplot(int(np.ceil(Ndsets_V1/2)), 2, iR+1)
                pcm = ax1.scatter(df['x'], df['y'], c=df['scale_xy_dist'], s=0.05, cmap=cmap_rev)
                plt.colorbar(pcm, ax=ax1)
                ax1.add_patch(ellipse)
                ax1.patch.set_facecolor('r')
                ax1.set_title(label + " " + hemi + " radius = %.1f$\\sigma$: SD<%.1f Nvox = %d"
                              % (roiRad, roiRad, np.sum(df['scale_xy_dist'] < roiRad)), fontsize=6)
                ax1.axis('off')
                ax1.set_aspect('equal')

                ellipse2 = Ellipse(com,
                                   width=2*roiRad*np.sqrt(a),
                                   height=2*roiRad*np.sqrt(b),
                                   angle=180*theta/np.pi,
                                   zorder=100, alpha=1., edgecolor='r', facecolor='None')
                ax2 = floc.add_subplot(int(np.ceil(Ndsets_V1/2)), 2, iR+1)
                pcm = ax2.scatter(df['x'], df['y'], c=df['ctr-sur_unwarp'], s=0.05,
                                  cmap=cmap, vmin=-2, vmax=2)
                plt.colorbar(pcm, ax=ax2)
                ax2.add_patch(ellipse2)
                ax2.patch.set_facecolor('r')
                ax2.set_title(label + " " + hemi + " localizer: SD<%.1f Nvox = %d"
                              % (roiRad, np.sum(df['scale_xy_dist'] < roiRad)), fontsize=6)
                ax2.axis('off')
                ax2.set_aspect('equal')

                iR += 1

    frad.tight_layout(pad=0.5)
    floc.tight_layout(pad=0.5)
    if savefigs:
        frad.savefig(os.path.join(figDir, 'xy_map_rad.%s'  % fig_format))
        floc.savefig(os.path.join(figDir, 'xy_map_loc.%s'  % fig_format))


# ---------------------------------------------------------------------------
# p-value histograms
# ---------------------------------------------------------------------------

def plot_pval_histograms(all_data, roiRad, subject_visArea_combinations, savefigs, figDir, fig_format):
    # loc p-values
    fig_loc, axes_loc = plt.subplots(
        2, int(np.ceil(len(subject_visArea_combinations)/2)),
        figsize=(5 * len(subject_visArea_combinations), 12)
    )
    if len(subject_visArea_combinations) == 1:
        axes_loc = [axes_loc]
    for i, (subjID, vis_region) in enumerate(subject_visArea_combinations):
        r, c = np.unravel_index(i, (2, int(np.ceil(len(subject_visArea_combinations)/2))))
        ax   = axes_loc[r, c]
        df   = all_data[subjID]
        roi  = df[(df['Visual Region'] == vis_region) &
                  ((df['scale_xy_dist'] < roiRad) | (df['scale_xy_dist'].isna()))]
        if 'loc pval' in roi.keys():
            roi = roi.rename(columns={'loc pval': 'loc p-val'})
        color  = 'b' if vis_region == 'V1' else ('r' if vis_region == 'V23' else 'gray')
        minp   = 10**-3
        maxp   = 1
        logbins = np.logspace(np.log(minp), np.log(maxp), 20)
        ax.hist(roi['loc p-val'].values, bins=logbins, density=True, alpha=0.5, color=color)
        ax.set_xscale('log')
        p_less_05 = (100 * (roi['loc p-val'] <= 0.05).sum() / len(roi['loc p-val'])
                     if len(roi['loc p-val']) > 0 else 0)
        ax.set_title(f"{subjID} - {vis_region} loc p-val \n (p < 0.05: {p_less_05:.2f}%)", fontsize=12)
        ax.set_xlabel("pval", fontsize=10)
        ax.set_ylim([0, 100])
        ax.set_xlim([minp, maxp])
        ax.tick_params(axis='x', labelsize=10)
    fig_loc.tight_layout(pad=0.5)

    # task p-values
    fig_task, axes_task = plt.subplots(
        2, int(np.ceil(len(subject_visArea_combinations)/2)),
        figsize=(5 * len(subject_visArea_combinations), 12)
    )
    if len(subject_visArea_combinations) == 1:
        axes_task = [axes_task]
    for i, (subjID, vis_region) in enumerate(subject_visArea_combinations):
        r, c = np.unravel_index(i, (2, int(np.ceil(len(subject_visArea_combinations)/2))))
        ax   = axes_task[r, c]
        df   = all_data[subjID]
        roi  = df[(df['Visual Region'] == vis_region) &
                  ((df['scale_xy_dist'] < roiRad) | (df['scale_xy_dist'].isna()))]
        if 'task pval' in roi.keys():
            roi = roi.rename(columns={'task pval': 'task p-val'})
        color   = 'b' if vis_region == 'V1' else ('r' if vis_region == 'V23' else 'gray')
        minp    = 10**-3
        maxp    = 1
        logbins = np.logspace(np.log(minp), np.log(maxp), 20)
        ax.hist(roi['task p-val'].values, bins=logbins, density=True, alpha=0.5, color=color)
        ax.set_xscale('log')
        p_less_05 = (100 * (roi['task p-val'] <= 0.05).sum() / len(roi['task p-val'])
                     if len(roi['task p-val']) > 0 else 0)
        ax.set_title(f"{subjID} - {vis_region} task p-val \n (p < 0.05: {p_less_05:.2f}%)", fontsize=12)
        ax.set_xlabel("pval", fontsize=10)
        ax.set_ylim([0, 100])
        ax.set_xlim([minp, maxp])
        ax.tick_params(axis='x', labelsize=10)
    fig_task.tight_layout(pad=0.5)

    if savefigs:
        fig_loc.savefig(os.path.join(figDir,  f'pvals_loc.{fig_format}'))
        fig_task.savefig(os.path.join(figDir, f'pvals_task.{fig_format}'))


# ---------------------------------------------------------------------------
# Depth histograms
# ---------------------------------------------------------------------------

def plot_depth_histograms(all_data, roiRad, nDepths, subject_visArea_combinations,
                          savefigs, figDir, fig_format):
    fig_dhist, axes_dhist = plt.subplots(
        2, int(np.ceil(len(subject_visArea_combinations)/2)),
        figsize=(5 * len(subject_visArea_combinations), 12)
    )
    if len(subject_visArea_combinations) == 1:
        axes_dhist = [axes_dhist]

    for i, (subjID, vis_region) in enumerate(subject_visArea_combinations):
        r, c = np.unravel_index(i, (2, int(np.ceil(len(subject_visArea_combinations)/2))))
        ax   = axes_dhist[r, c]
        df   = all_data[subjID]
        roi  = df[(df['Visual Region'] == vis_region) &
                  ((df['scale_xy_dist'] < roiRad) | (df['scale_xy_dist'].isna()))]
        color = 'b' if vis_region == 'V1' else ('r' if vis_region == 'V23' else 'gray')
        ax.hist(roi['d'].values, bins=nDepths, density=False, alpha=0.5, color=color)
        ax.set_title(f"{subjID} - {vis_region} depth", fontsize=6)
        ax.set_xlabel("Normalize Depth WM -> GM")
        ax.set_ylabel("Num. Voxels")
        ax.set_xlim([0, 1])
        plt.legend(['N='+str(len(roi)),], fontsize=6)
    fig_dhist.tight_layout(pad=0.5)

    if savefigs:
        fig_dhist.savefig(os.path.join(figDir, f'pvals_loc.{fig_format}'))


# ---------------------------------------------------------------------------
# Depth box-and-whisker plots
# ---------------------------------------------------------------------------

def plot_depth_boxplots(all_data, nDepths, masks=None,
                        savefigs=False, figDir=None, fig_format='svg', plottype="box"):
    """
    Plot distribution of voxel counts at each depth bin across subjects.
    """
    fig_dbox, axes_dbox = plt.subplots(1, 1, figsize=(8, 6))
    depth_bins = np.linspace(0, 1, nDepths + 1)
    depth_counts = [[] for _ in range(nDepths)]
    # Gather voxel counts by depth bin
    for i, label in enumerate(all_data.keys()):
        df = all_data[label]
        if masks is not None and label in masks:
            mask = masks[label]
        else:
            mask = df.index
        roi = df[mask==1]

        for d_i in range(nDepths):
            count = roi['d_bin'].value_counts().get(d_i, 0)
            depth_counts[d_i].append(count)
    if plottype == "box":
        axes_dbox.boxplot(depth_counts, positions=np.arange(nDepths), widths=0.6)
    elif plottype == "violin":
        sns.violinplot(data=depth_counts, ax=axes_dbox, inner='box', color='lightblue')
        sns.swarmplot(data=depth_counts, ax=axes_dbox, color='k', size=3)
    else:
        raise ValueError("Invalid plottype. Use 'box' or 'violin'.")
    axes_dbox.set_xticks(np.arange(nDepths))
    axes_dbox.set_xticklabels([f"{depth_bins[d_i]:.2f}-{depth_bins[d_i+1]:.2f}" for d_i in range(nDepths)],
                                rotation=45, fontsize=6)
    axes_dbox.set_xlabel("Depth bins", fontsize=8)
    axes_dbox.set_ylabel("Voxel count", fontsize=8)
    axes_dbox.set_title(f"Depth distribution", fontsize=10)
    fig_dbox.tight_layout(pad=0.5)
    if savefigs:
        fig_dbox.savefig(os.path.join(figDir, f'depth_boxplots.{fig_format}'))


# ---------------------------------------------------------------------------
# Deveining diagnostic plots
# ---------------------------------------------------------------------------

def plot_deveining_qc(all_data, lmnv_dict, nDepths, depth_var='d_norm',
                      deep_pct=10, sd_thresh=2, x_var='x', y_var='y',
                      savefigs=False, figDir=None, fig_format='svg',
                      vAreas=['V1']):
    depth_groups = {'deep': [0.0, 1/3], 'middle': [1/3, 2/3], 'superficial': [2/3, 1.0]}
    depth_labels = ['superficial', 'middle', 'deep']
    Ngroups      = len(depth_labels)
    fsize        = 12

    # Pass 1: collect all ROI records across both vAreas so figures can be
    # pre-allocated with the correct number of columns.
    roi_records = []
    for vArea in vAreas:
        subjIDs = [subj for subj in all_data.keys()
                   if all_data[subj]['Visual Region'].str.contains(vArea).any()]
        for label in subjIDs:
            df_all = all_data[label]
            for hemi in df_all['hemi'].unique():
                if np.sum((df_all['Visual Region'] == vArea) &
                          (df_all['Subregion'] == 'tgt') &
                          (df_all['hemi'] == hemi)) == 0:
                    continue
                df   = df_all[(df_all['Visual Region'] == vArea) &
                               (df_all['Subregion'] == 'tgt') &
                               (df_all['hemi'] == hemi)]
                lmnv = get_lmnv(df, key='stdev_xerrts')
                mnv  = np.exp(lmnv)
                z    = df[depth_var]
                [_, _, deep]          = get_deep_layer_dist(df, depth_var, deep_pct)
                [mnv_mask, _]         = get_mnv_mask(df, depth_var, deep_pct, sd_thresh)
                roi_records.append({
                    'label': label, 'hemi': hemi, 'vArea': vArea,
                    'df': df, 'lmnv': lmnv, 'mnv': mnv, 'z': z,
                    'deep': deep, 'mnv_mask': mnv_mask,
                })

    Ndsets  = len(roi_records)
    col_w   = 2.5

    # Pass 2: create 4 figures with proper grids, then fill subplot by subplot.
    fig_hist,   axes_hist   = plt.subplots(2,          Ndsets,
                                            figsize=(col_w * Ndsets, 5))
    fig_dmap,   axes_dmap   = plt.subplots(Ngroups * 2, Ndsets,
                                            figsize=(col_w * Ndsets, 3 * Ngroups))
    fig_thresh, axes_thresh = plt.subplots(Ngroups * 2, Ndsets,
                                            figsize=(col_w * Ndsets, 3 * Ngroups))
    fig_depth,  axes_depth  = plt.subplots(2,          Ndsets,
                                            figsize=(col_w * Ndsets, 5))

    # Ensure axes are always 2-D even when Ndsets == 1
    axes_hist   = np.atleast_2d(axes_hist)
    axes_dmap   = np.atleast_2d(axes_dmap)
    axes_thresh = np.atleast_2d(axes_thresh)
    axes_depth  = np.atleast_2d(axes_depth)

    for k_i, roi in enumerate(roi_records):
        title = '%s %s %s' % (roi['label'], roi['vArea'], roi['hemi'])

        plot_mnv_histograms(roi['lmnv'], roi['lmnv'][roi['deep']], roi['mnv_mask'],
                            deep_pct, title,
                            axes_hist[0, k_i], axes_hist[1, k_i], fsize)

        scatter_axes = [axes_dmap[2 * d_i,     k_i] for d_i in range(Ngroups)]
        hist_axes    = [axes_dmap[2 * d_i + 1, k_i] for d_i in range(Ngroups)]
        plot_depth_maps(roi['df'], depth_var, depth_groups, depth_labels,
                        x_var, y_var, roi['mnv'], [2, 5], fsize,
                        scatter_axes, hist_axes)

        scatter_axes_t = [axes_thresh[2 * d_i,     k_i] for d_i in range(Ngroups)]
        hist_axes_t    = [axes_thresh[2 * d_i + 1, k_i] for d_i in range(Ngroups)]
        plot_depth_maps(roi['df'], depth_var, depth_groups, depth_labels,
                        x_var, y_var, roi['mnv'], [2, 5], fsize,
                        scatter_axes_t, hist_axes_t, mask=roi['mnv_mask'])

        plot_depth_voxel_loss(roi['z'], roi['mnv_mask'], nDepths, title,
                               axes_depth[0, k_i], axes_depth[1, k_i], fsize)

    for fig in [fig_hist, fig_dmap, fig_thresh, fig_depth]:
        fig.tight_layout()

    if savefigs and figDir is not None:
        fig_hist.savefig(  os.path.join(figDir, 'mnv_hist.%s'            % fig_format))
        fig_dmap.savefig(  os.path.join(figDir, 'mnv_depth_map.%s'       % fig_format))
        fig_thresh.savefig(os.path.join(figDir, 'mnv_depth_map_thresh.%s' % fig_format))
        fig_depth.savefig( os.path.join(figDir, 'mnv_depth_hist.%s'      % fig_format))


# ---------------------------------------------------------------------------
# Threshold comparison violin plots
# ---------------------------------------------------------------------------

def plot_mnv_threshold_comparison(all_data, lmnv_dict, depth_var='d_norm',
                                   deep_pct=10, savefigs=False, figDir=None,
                                   fig_format='svg', vAreas=['V1']):
    spreadF = 2
    for iR, vArea in enumerate(vAreas):
        k_i     = 0
        subjIDs = [subj for subj in all_data.keys()
                   if all_data[subj]['Visual Region'].str.contains(vArea).any()]
        f       = plt.figure()
        dsets   = []

        for iS, label in enumerate(subjIDs):
            df_all = all_data[label]

            for iH, hemi in enumerate(df_all['hemi'].unique()):

                if np.sum((df_all['Visual Region'] == vArea) &
                          (df_all['Subregion'] == 'tgt') &
                          (df_all['hemi'] == hemi)) != 0:

                    df   = df_all[(df_all['Visual Region'] == vArea) &
                                  (df_all['Subregion'] == 'tgt') &
                                  (df_all['hemi'] == hemi)]
                    lmnv = get_lmnv(df, key='stdev_xerrts')
                    [deep_mean, deep_std, deep] = get_deep_layer_dist(df, depth_var, deep_pct)

                    violin_parts = plt.violinplot(lmnv, positions=[spreadF*k_i], showmedians=True)
                    for pc in violin_parts['bodies']:
                        pc.set_facecolor('b')
                        pc.set_edgecolor('b')
                    for pc in violin_parts:
                        if not isinstance(violin_parts[pc], list):
                            violin_parts[pc].set_edgecolor('b')

                    violin_parts = plt.violinplot(lmnv[deep], positions=[spreadF*(k_i+0.2)],
                                                   showmedians=True)
                    for pc in violin_parts['bodies']:
                        pc.set_facecolor('orange')
                        pc.set_edgecolor('orange')
                    for pc in violin_parts:
                        if not isinstance(violin_parts[pc], list):
                            violin_parts[pc].set_edgecolor('orange')

                    plt.hlines(lmnv_dict[label]['thresh'],
                               spreadF*k_i - 0.5, spreadF*(k_i+0.2) + 0.5, color='r')
                    plt.title(vArea)

                    dsets.append(label + hemi)
                    k_i += 1

        plt.xticks(np.arange(0, spreadF*k_i, spreadF), dsets, rotation=30, fontsize=6)
        plt.ylabel("log(MNV)")

        if savefigs and figDir is not None:
            f.savefig(os.path.join(figDir, 'mnv_summary_violin_%s.%s' % (vArea, fig_format)))


# ---------------------------------------------------------------------------
# Voxel count across depth and radius after masking
# ---------------------------------------------------------------------------

def plot_voxel_count_depth_radius(all_data, nDepths_rings, savefigs, figDir, fig_format, figsize=(10, 5)):
    dbins = np.linspace(0, 1, nDepths_rings + 1)
    for subj in all_data.keys():
        fig, ax = plt.subplots(nDepths_rings, 3, figsize=figsize)
        if nDepths_rings == 1:
            ax = np.atleast_2d(ax)
        fig.suptitle(f"{subj}")
        for d_i in range(nDepths_rings):
            ax_i      = nDepths_rings - d_i - 1
            test_data = all_data[subj][
                (all_data[subj]['sig'] & all_data[subj]['no_vein'] &
                 (all_data[subj]['d_norm'] >= dbins[d_i]) &
                 (all_data[subj]['d_norm'] <  dbins[d_i + 1]))
            ]['scale_xy_dist']
            max_dist = np.max(all_data[subj]['scale_xy_dist'])
            ax[ax_i, 0].hist(test_data, bins=np.arange(0, np.ceil(max_dist), 1))
            if d_i == 0:
                ax[ax_i, 0].set_xlabel("Radial Distance ($\\sigma$)")
            ax[ax_i, 0].set_ylabel("Voxel Count")
            ax[ax_i, 0].set_title(f"Depth bin = {d_i}")
            ax[ax_i, 0].set_ylim([0, 1500])
            ax[ax_i, 0].set_xlim([0, np.ceil(max_dist)])
            ax[ax_i, 0].plot([0, np.ceil(max_dist)], [10, 10], '--r', label='10 voxels')
            ax[ax_i, 0].legend()

            for h_i, hemi in enumerate(['lh', 'rh']):
                df_V1 = all_data[subj][all_data[subj]['Visual Region'] == 'V1']
                if hemi in np.unique(df_V1['hemi']):
                    mask_h = (df_V1['sig'] & df_V1['no_vein'] &
                              (df_V1['d_norm'] >= dbins[d_i]) &
                              (df_V1['d_norm'] <  dbins[d_i + 1]) &
                              (df_V1['hemi'] == hemi))
                    a     = df_V1[mask_h]['ellipse_a'].values[0]
                    b     = df_V1[mask_h]['ellipse_b'].values[0]
                    theta = df_V1[mask_h]['ellipse_theta'].values[0]
                    comX  = df_V1[mask_h]['ellipse_comX'].values[0]
                    comY  = df_V1[mask_h]['ellipse_comY'].values[0]
                    com   = [comX, comY]

                    df    = df_V1[mask_h]
                    pcm   = ax[ax_i, h_i + 1].scatter(
                        df['x'], df['y'], c=df['ctr-sur_unwarp'],
                        s=4, cmap='plasma', vmin=-5, vmax=5
                    )
                    cbar = plt.colorbar(pcm, ax=ax[ax_i, h_i + 1])
                    cbar.set_label('Ctr - Sur', fontsize=10)

                    max_dist2 = np.max(df['scale_xy_dist'])
                    for scale in range(1, int(np.ceil(max_dist2)) + 1):
                        alpha_value = 1.0 - (scale / np.ceil(max_dist2))
                        width  = scale * 2 * np.sqrt(a)
                        height = scale * 2 * np.sqrt(b)
                        ellipse = Ellipse(
                            com, width=width, height=height,
                            angle=180 * theta / np.pi,
                            alpha=alpha_value, edgecolor='r', facecolor='none',
                            label=f'Scale {scale}' if scale == 1 else None
                        )
                        ax[ax_i, h_i + 1].add_patch(ellipse)

                        theta_rad = np.deg2rad(180 * theta / np.pi)
                        dx = (height / 2) * np.sin(theta_rad)
                        dy = (height / 2) * np.cos(theta_rad)
                        ax[ax_i, h_i + 1].text(
                            com[0] + dx, com[1] - dy, f'{scale}$\\sigma$',
                            color='darkred', fontsize=6, ha='center', va='center',
                            rotation=theta
                        )

                    ax[ax_i, h_i + 1].set_title(f"{hemi} Depth bin = {d_i}")
                    ax[ax_i, h_i + 1].set_xlim([np.min(df['x'])-1, np.max(df['x'])+1])
                    ax[ax_i, h_i + 1].set_ylim([np.min(df['y'])-1, np.max(df['y'])+1])
                    ax[ax_i, h_i + 1].axis('off')
                    ax[ax_i, h_i + 1].set_aspect('equal')

        plt.tight_layout()
        if savefigs:
            fig.savefig(os.path.join(figDir,
                        'radial_distance_histograms_%s_V1.%s' % (subj, fig_format)))


# ---------------------------------------------------------------------------
# Total % BOLD change per subject
# ---------------------------------------------------------------------------

def plot_bold_per_subject(all_data, statDetails, savefigs, figDir, fig_format, plot_V23=False):

    for subj in all_data.keys():
        df     = all_data[subj]
        df_tgt = df[(df['Visual Region'] == 'V1') & (df['in_tgt'] == True) &
                    (df['sig'] == True) & (df['no_vein'] == True)]
        if plot_V23:
            df_V23 = df[(df['Visual Region'] == 'V23') & (df['sig'] == True) &
                        (df['no_vein'] == True)]
        if plot_V23:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax = [ax] #make subscriptable
        fig.suptitle(f"{subj}")

        if not df_tgt.empty:
            conditions = ['sur', 'iso90', 'orth', 'iso0']
            means      = [df_tgt[cond].mean() for cond in conditions]
            sns.violinplot(data=df_tgt[conditions], ax=ax[0],
                           palette=statDetails['colors'][:len(conditions)], inner=None)
            sns.swarmplot(data=df_tgt[conditions], ax=ax[0], color='k', alpha=0.7)
            for i, cond in enumerate(conditions):
                ax[0].plot([i-0.2, i+0.2], [means[i], means[i]],
                           color=statDetails['colors'][i], linewidth=2)
            ax[0].set_title('V1 tgt')
            ax[0].set_ylabel('Average BOLD Response')
            ax[0].set_xticklabels(conditions, rotation=45, ha='right')

        if plot_V23 and not df_V23.empty:
            conditions_V23 = [
                'V23_superficial_deveined_orth', 'V23_middle_deveined_orth', 'V23_deep_deveined_orth',
                'V23_superficial_deveined_iso90', 'V23_middle_deveined_iso90', 'V23_deep_deveined_iso90',
                'V23_superficial_deveined_iso0', 'V23_middle_deveined_iso0', 'V23_deep_deveined_iso0',
                'V23_superficial_deveined_sur', 'V23_middle_deveined_sur', 'V23_deep_deveined_sur',
            ]
            concatenated_conditions = ['V23_deveined_orth', 'V23_deveined_iso90',
                                       'V23_deveined_iso0', 'V23_deveined_sur']
            df_V23_concat = pd.DataFrame()
            for i, base_cond in enumerate(['orth', 'iso90', 'iso0', 'sur']):
                df_V23_concat[concatenated_conditions[i]] = pd.concat([
                    df_V23[f'V23_superficial_deveined_{base_cond}'],
                    df_V23[f'V23_middle_deveined_{base_cond}'],
                    df_V23[f'V23_deep_deveined_{base_cond}']
                ])
            means = [df_V23_concat[cond].mean() for cond in concatenated_conditions]
            sns.violinplot(data=df_V23_concat, ax=ax[1],
                           palette=statDetails['colors'][:len(concatenated_conditions)], inner=None)
            for i, cond in enumerate(concatenated_conditions):
                ax[1].plot([i-0.2, i+0.2], [means[i], means[i]],
                           color=statDetails['colors'][i], linewidth=2)
            ax[1].set_title('V23')
            ax[1].set_ylabel('Average BOLD Response')
            ax[1].set_xticklabels(concatenated_conditions, rotation=45, ha='right')

        plt.tight_layout()
        if savefigs:
            fig.savefig(os.path.join(figDir, 'avg_bold_response_%s.%s' % (subj, fig_format)))


def _annotate_swarm_by_facecolor(ax, start_collection_idx, palette, x_offset=0.02, fontsize=6):
    subj_names = list(palette.keys())
    subj_rgb   = np.array([palette[s] for s in subj_names])

    for coll in ax.collections[start_collection_idx:]:
        offsets = coll.get_offsets()
        if offsets is None or len(offsets) == 0:
            continue
        facecolors = coll.get_facecolors()
        if len(facecolors) == 0:
            continue
        if len(facecolors) == 1 and len(offsets) > 1:
            facecolors = np.repeat(facecolors, len(offsets), axis=0)
        for (x, y), fc in zip(offsets, facecolors):
            rgb    = np.array(fc[:3])
            i_subj = np.argmin(np.linalg.norm(subj_rgb - rgb, axis=1))
            subj   = subj_names[i_subj]
            ax.text(x + x_offset, y, subj, color=palette[subj],
                    fontsize=fontsize, va="center")


def plot_bold_summary(all_data, statDetails, savefigs, figDir, fig_format, statsDir, plot_V23=False):

    V1_means  = {}
    V23_means = {}

    for subj in all_data.keys():
        df     = all_data[subj]
        df_tgt = df[(df['Visual Region'] == 'V1') & (df['in_tgt'] == True) &
                    (df['sig'] == True) & (df['no_vein'] == True)]
        if plot_V23:
            df_V23 = df[(df['Visual Region'] == 'V23') & (df['sig'] == True) &
                        (df['no_vein'] == True)]

        if not df_tgt.empty:
            conditions = ['sur', 'iso90', 'orth', 'iso0']
            V1_means[subj] = dict(zip(conditions, [df_tgt[c].mean() for c in conditions]))

        if plot_V23 and not df_V23.empty:
            concatenated_conditions = ['V23_deveined_orth', 'V23_deveined_iso90',
                                       'V23_deveined_iso0', 'V23_deveined_sur']
            df_V23_concat = pd.DataFrame()
            for i, base_cond in enumerate(['orth', 'iso90', 'iso0', 'sur']):
                df_V23_concat[concatenated_conditions[i]] = pd.concat([
                    df_V23[f'V23_superficial_deveined_{base_cond}'],
                    df_V23[f'V23_middle_deveined_{base_cond}'],
                    df_V23[f'V23_deep_deveined_{base_cond}']
                ])
            V23_means[subj] = dict(zip(concatenated_conditions,
                                       [df_V23_concat[c].mean() for c in concatenated_conditions]))

    if plot_V23:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax = [ax]  # make subscriptable
    fig.suptitle("Average BOLD Response Across Subjects")

    if V1_means:
        V1_df   = pd.DataFrame(V1_means).T
        V1_long = (V1_df.reset_index().rename(columns={"index": "subject"})
                   .melt(id_vars="subject", var_name="condition", value_name="value").dropna())
        subj_order  = list(V1_df.index)
        plot_colors = sns.color_palette("husl", len(subj_order))
        palette     = dict(zip(subj_order, plot_colors))
        n_before    = len(ax[0].collections)
        sns.swarmplot(data=V1_long, x="condition", y="value", hue="subject",
                      hue_order=subj_order, palette=palette, dodge=False,
                      alpha=0.7, ax=ax[0], legend=False)
        _annotate_swarm_by_facecolor(ax[0], n_before, palette, x_offset=0.02, fontsize=6)
        ax[0].set_title('V1 tgt')
        ax[0].set_ylabel('Average % BOLD Change')
        ax[0].set_xticklabels(V1_df.columns, rotation=45, ha='right')

    if V23_means:
        V23_df   = pd.DataFrame(V23_means).T
        V23_long = (V23_df.reset_index().rename(columns={"index": "subject"})
                    .melt(id_vars="subject", var_name="condition", value_name="value").dropna())
        subj_order  = list(V23_df.index)
        plot_colors = sns.color_palette("husl", len(subj_order))
        palette     = dict(zip(subj_order, plot_colors))
        n_before    = len(ax[1].collections)
        sns.swarmplot(data=V23_long, x="condition", y="value", hue="subject",
                      hue_order=subj_order, palette=palette, dodge=False,
                      alpha=0.7, ax=ax[1], legend=False)
        _annotate_swarm_by_facecolor(ax[1], n_before, palette, x_offset=0.02, fontsize=6)
        ax[1].set_title('V23 tgt')
        ax[1].set_ylabel('Average % BOLD Change')
        ax[1].set_xticklabels(V23_df.columns, rotation=45, ha='right')

    plt.tight_layout()
    if savefigs:
        fig.savefig(os.path.join(figDir, f"avg_bold_response_across_subjects.{fig_format}"))

    for region, means in zip(['V1', 'V23'], [V1_means, V23_means]):
        df      = pd.DataFrame(means).T
        zscored = (df - df.mean()) / df.std()
        print(f"Summary stats for {region}:")
        print("\n")
        print(zscored)
        csv_filename = f"{region}_zscored_means.csv"
        csv_filepath = os.path.join(statsDir, 'QC', csv_filename)
        zscored.to_csv(csv_filepath, index_label='Subject')
        print(f"Z-scored means saved to '{csv_filepath}'")


# ---------------------------------------------------------------------------
# Centroid plots
# ---------------------------------------------------------------------------

def plot_centroids_qc(included_data, masks_in_tgt, STATS, DIFFS, roiRad, nDepths,
                      savefigs, figDir, fig_format, analysis):
    plot_centroids(included_data, masks_in_tgt, STATS, roiRad, nDepths=nDepths)
    plot_centroids_diff(included_data, masks_in_tgt, STATS, DIFFS, roiRad, nDepths)

    if savefigs:
        for l in STATS['labels']:
            plt.figure(l)
            plt.savefig(os.path.join(figDir, 'centroids_%s.%s' % (l, fig_format)))
        for l in DIFFS['statIDs'].keys():
            plt.figure(l)
            plt.savefig(os.path.join(figDir, 'centroids_%s.%s' % (l, fig_format)))


# ---------------------------------------------------------------------------
# Individual subject depth profiles overlaid on average
# ---------------------------------------------------------------------------

def plot_individual_condition_profiles(depthProfiles, avgDepthProfiles,
                                        statDetails, diffDetails,
                                        stat_analyses, diff_analyses,
                                        use_decon, fig_size,
                                        savefigs, figDir, fig_format,
                                        fcolor, lcolor, roi_types):
    cm = 1/2.54
    dx = 4.
    dy = 0.7

    for roi_type in roi_types:
        for analysis in ['task', 'loc']:
            STATS = _get_stats(statDetails, diffDetails, stat_analyses, diff_analyses, analysis)
            for iStat, stat in enumerate(STATS['labels']):

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

                p1 = fig.add_axes([.2, .25, .3, .7])
                fix_axes(p1, lcolor=lcolor, fcolor=fcolor)

                ylim = [-0.02, 1.02]
                if analysis == 'task':
                    xlim = [0, 12.5]
                elif analysis == 'loc':
                    xlim = [-3, 7]
                Ntext = [4, 0.05]

                if use_decon[analysis]:
                    if type(depthProfiles[roi_type][analysis][stat]['avg_decon']) == list:
                        depthProfiles[roi_type][analysis][stat]['avg_decon'] = np.vstack(
                            depthProfiles[roi_type][analysis][stat]['avg_decon']
                        )
                else:
                    if type(depthProfiles[roi_type][analysis][stat]['avg']) == list:
                        depthProfiles[roi_type][analysis][stat]['avg'] = np.vstack(
                            depthProfiles[roi_type][analysis][stat]['avg']
                        )

                xticks = [0, 4, 8, 12]
                plot_avg_depth_profile(
                    p1, avgDepthProfiles[roi_type][analysis], [stat],
                    [STATS['colors'][iStat]], ylim, xlim, dx, dy,
                    Ntext, lcolor, fsize,
                    plot_indiv=depthProfiles[roi_type][analysis],
                    use_decon=use_decon[analysis], xticks=xticks
                )

                if savefigs:
                    if use_decon[analysis]:
                        fig.savefig(os.path.join(
                            figDir, 'avg_profiles_%s_%s_%s_deconv.%s'
                            % (analysis, roi_type, stat, fig_format)))
                    else:
                        fig.savefig(os.path.join(
                            figDir, 'avg_profiles_%s_%s_%s.%s'
                            % (analysis, roi_type, stat, fig_format)))


def plot_individual_diff_profiles(diffProfiles, avgDepthDiffs,
                                   statDetails, diffDetails,
                                   stat_analyses, diff_analyses,
                                   use_decon, fig_size, pthresh,
                                   savefigs, figDir, fig_format,
                                   fcolor, lcolor, roi_types):
    cm = 1/2.54
    dx = 4.
    dy = 0.7

    for roi_type in roi_types:
        for analysis in ['task', 'loc']:
            STATS, DIFFS = _get_stats_diffs(statDetails, diffDetails,
                                             stat_analyses, diff_analyses, analysis)
            for iDiff, diff in enumerate(DIFFS['statIDs'].keys()):

                if fig_size == "small":
                    fig = plt.figure(figsize=(6*cm, 4*cm))
                    fig.set_size_inches((6*cm, 4*cm))
                elif fig_size == "large":
                    fig = plt.figure(figsize=(6, 4))
                    fig.set_size_inches((6, 4))
                fig.patch.set_facecolor(fcolor)
                fig.clf()
                fsize = 7 if fig_size == "small" else 10

                p2 = fig.add_axes([.7, .25, .25, .7])
                fix_axes(p2, lcolor=lcolor, fcolor=fcolor)

                ylim = [-0.02, 1.02]
                if analysis == 'task':
                    xlim = [-1.0, 5.0]
                elif analysis == 'loc':
                    xlim = [-1.0, 5.0]
                Ntext = [4, 0.05]

                plot_avg_diff_profile(
                    p2, avgDepthDiffs[roi_type][analysis], [diff],
                    [DIFFS['colors'][iDiff]], ylim, xlim, dx, dy,
                    Ntext, lcolor, fsize, useSI=False,
                    plot_indiv=diffProfiles[roi_type][analysis],
                    showSig=True, pthresh=pthresh,
                    statCorrType=avgDepthDiffs[roi_type][analysis][diff]['corrected p-vals'],
                    use_decon=use_decon[analysis]
                )
                if fig_size == "large":
                    p2.set_xticks([0.0, 2.5, 5.0])

                if savefigs:
                    if use_decon[analysis]:
                        fig.savefig(os.path.join(
                            figDir, 'avg_diffs_%s_%s_%s_deconv.%s'
                            % (analysis, roi_type, diff, fig_format)))
                    else:
                        fig.savefig(os.path.join(
                            figDir, 'avg_diffs_%s_%s_%s.%s'
                            % (analysis, roi_type, diff, fig_format)))


# ---------------------------------------------------------------------------
# Per-subject radial profiles subplot
# ---------------------------------------------------------------------------

def plot_radial_profiles_subplot(data, analysis_type, condition, kernel,
                                  mask=None, smooth_factor=0.3, radMax=4,
                                  nRadii=20, ymin=-5, ymax=5, fontsize=8,
                                  fcolor='white', lcolor='black',
                                  depth_labels=['deep', 'middle', 'superficial'],
                                  depthBoundaries=np.array([[0, 1/3], [1/3, 2/3], [2/3, 1]]),
                                  statColor='gray', vline=2):
    if isinstance(condition, str):
        condition = [condition]
    if isinstance(statColor, str):
        statColor = [statColor]

    all_profiles = {depth_label: {cond: {} for cond in condition}
                    for depth_label in depth_labels}

    for iR, label in enumerate(data.keys()):
        df = data[label]
        if mask:
            df = df[mask[label]]
        for iD, depth_label in enumerate(depth_labels):
            depth_df = df[(df['d'] >= depthBoundaries[iD, 0]) &
                          (df['d'] <  depthBoundaries[iD, 1])]
            x = depth_df['scale_xy_dist'].values
            for cond in condition:
                coef            = depth_df[cond].values
                coef_smooth, x_smooth = smoothen(coef, x, kernel=kernel,
                                                  smooth_factor=smooth_factor,
                                                  radMax=radMax)
                all_profiles[depth_label][cond][label] = coef_smooth

    n_rows = len(depth_labels)
    n_cols = len(data.keys())
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    fig.patch.set_facecolor(fcolor)

    for iD, depth_label in enumerate(depth_labels):
        for iR, label in enumerate(data.keys()):
            ax = axs[iD, iR] if n_rows > 1 and n_cols > 1 else axs[max(iD, iR)]
            fix_axes(ax, lcolor=lcolor, fcolor=fcolor)

            for iC, cond in enumerate(condition):
                stat_avg = all_profiles[depth_label][cond][label]
                ax.plot(x_smooth, stat_avg, label=f'{cond}', alpha=0.7, color=statColor[iC])

            ax.set_ylim([ymin, ymax])
            ax.set_title(f'{depth_label} - {label}', fontsize=fontsize, color=lcolor)
            ax.plot([vline, vline], [ymin, ymax], '--', color='black')
            ax.plot([0, 4], [0, 0], '--', color='black')
            if iD == n_rows - 1:
                ax.set_xlabel('relative distance from ROI center ($\\sigma$)',
                              fontsize=fontsize, color=lcolor)
            if iR == 0:
                ax.set_ylabel('BOLD % change', fontsize=fontsize, color=lcolor)
            ax.legend(fontsize=fontsize - 2)

    plt.tight_layout()
    return fig


def plot_individual_radial_profiles(all_data, subj_analyses, statDetails, diffDetails,
                                     stat_analyses, diff_analyses,
                                     kernel, rad_depth_labels, rad_depthBoundaries,
                                     maxRad, savefigs, figDir, fig_format,
                                     return_included_subj_fn):
    for analysis in ['task', 'loc', 'gPPI']:
        included_data = return_included_subj_fn(subj_analyses, analysis)
        STATS, DIFFS  = _get_stats_diffs(statDetails, diffDetails,
                                          stat_analyses, diff_analyses, analysis)
        stat       = STATS['labels']
        stat_color = []
        for s in stat:
            stat_id = STATS['labels'].index(s)
            stat_color.append(STATS['colors'][stat_id])

        masks = {label: (included_data[label]['no_vein'] &
                         included_data[label]['sig'] &
                         (included_data[label]['Visual Region'] == 'V1'))
                 for label in included_data.keys()}

        fig = plot_radial_profiles_subplot(
            included_data, analysis, stat, kernel, mask=masks,
            ymin=-2, ymax=10, statColor=stat_color, vline=1,
            depth_labels=rad_depth_labels, depthBoundaries=rad_depthBoundaries
        )
        if savefigs:
            fig.savefig(os.path.join(figDir, "radial_profiles_%s.%s" % (analysis, fig_format)))


def plot_diff_radial_profiles_subplot(all_data, subj_analyses, statDetails, diffDetails,
                                       stat_analyses, diff_analyses,
                                       kernel, rad_depth_labels, rad_depthBoundaries,
                                       savefigs, figDir, fig_format,
                                       return_included_subj_fn):
    for analysis in ['task', 'loc', 'gPPI']:
        included_data = return_included_subj_fn(subj_analyses, analysis)
        STATS, DIFFS  = _get_stats_diffs(statDetails, diffDetails,
                                          stat_analyses, diff_analyses, analysis)
        diff       = DIFFS['statIDs']
        diff_color = []
        for d in diff:
            diff_id = list(DIFFS['statIDs'].keys()).index(d)
            diff_color.append(DIFFS['colors'][diff_id])

        masks = {label: (included_data[label]['no_vein'] &
                         included_data[label]['sig'] &
                         (included_data[label]['Visual Region'] == 'V1'))
                 for label in included_data.keys()}

        fig = plot_radial_profiles_subplot(
            included_data, analysis, diff, kernel, mask=masks,
            ymin=-2, ymax=2, statColor=diff_color, vline=1,
            depth_labels=rad_depth_labels, depthBoundaries=rad_depthBoundaries
        )
        if savefigs:
            fig.savefig(os.path.join(figDir,
                        "radial_diff_profiles_%s.%s" % (analysis, fig_format)))


# ---------------------------------------------------------------------------
# Per-subject binned radial analysis
# ---------------------------------------------------------------------------

def plot_radial_analysis(all_data, column_name, nDepths_rings=3, plot_type="violin"):
    dbins = np.linspace(0, 1, nDepths_rings + 1)

    for subj in all_data.keys():
        fig, ax = plt.subplots(nDepths_rings, 3, figsize=(10, 12))
        fig.suptitle(f"{subj}")

        for d_i in range(nDepths_rings):
            ax_i     = nDepths_rings - d_i - 1
            df_V1    = all_data[subj][all_data[subj]['Visual Region'] == 'V1']
            depth_bin_data = df_V1[
                (df_V1['sig'] & df_V1['no_vein'] &
                 (df_V1['d_norm'] >= dbins[d_i]) &
                 (df_V1['d_norm'] <  dbins[d_i + 1]))
            ]

            max_dist    = np.ceil(np.max(depth_bin_data['scale_xy_dist'])) if len(depth_bin_data) > 0 else 1
            radial_bins = np.arange(0, max_dist + 1, 1)
            avg_values  = []
            std_values  = []
            radial_bin_values = []

            for r_i in range(len(radial_bins) - 1):
                radial_bin_data = depth_bin_data[
                    (depth_bin_data['scale_xy_dist'] >= radial_bins[r_i]) &
                    (depth_bin_data['scale_xy_dist'] <  radial_bins[r_i + 1])
                ]
                values = radial_bin_data[column_name].values
                avg_values.append(values.mean() if len(values) > 0 else 0)
                std_values.append(values.std()  if len(values) > 1 else 0)
                radial_bin_values.append(values if len(values) > 0 else [0])

            if plot_type == "violin":
                ax[ax_i, 0].violinplot(radial_bin_values,
                                       positions=np.arange(len(radial_bins) - 1),
                                       showmeans=True)
            elif plot_type == "bar":
                ax[ax_i, 0].bar(radial_bins[:-1], avg_values, width=1, color='b', align='edge')
                ax[ax_i, 0].errorbar(radial_bins[:-1] + 0.5, avg_values,
                                     yerr=std_values, fmt='o', color='k', capsize=3)
            else:
                raise ValueError("Invalid plot_type. Choose 'violin' or 'bar'.")

            if d_i == 0:
                ax[ax_i, 0].set_xlabel("Radial Distance ($\\sigma$)")
            ax[ax_i, 0].set_ylabel("Value")
            ax[ax_i, 0].set_title(f"Depth bin = {d_i}")
            ax[ax_i, 0].set_xlim([-0.5, len(radial_bins) - 1])

            for h_i, hemi in enumerate(['lh', 'rh']):
                if hemi in np.unique(df_V1['hemi']):
                    ellipse_data = depth_bin_data[depth_bin_data['hemi'] == hemi]
                    if len(ellipse_data) > 0:
                        a     = ellipse_data['ellipse_a'].values[0]
                        b     = ellipse_data['ellipse_b'].values[0]
                        theta = ellipse_data['ellipse_theta'].values[0]
                        comX  = ellipse_data['ellipse_comX'].values[0]
                        comY  = ellipse_data['ellipse_comY'].values[0]
                        com   = [comX, comY]

                        pcm  = ax[ax_i, h_i + 1].scatter(
                            ellipse_data['x'], ellipse_data['y'],
                            c=ellipse_data[column_name], s=4,
                            cmap='plasma', vmin=-5, vmax=5
                        )
                        cbar = plt.colorbar(pcm, ax=ax[ax_i, h_i + 1])
                        cbar.set_label(column_name, fontsize=10)

                        max_dist2 = np.max(ellipse_data['scale_xy_dist']) if len(ellipse_data) > 0 else 1
                        for scale in range(1, int(np.ceil(max_dist2)) + 1):
                            alpha_value = 1.0 - (scale / np.ceil(max_dist2))
                            width  = scale * 2 * np.sqrt(a)
                            height = scale * 2 * np.sqrt(b)
                            ellipse = Ellipse(
                                com, width=width, height=height,
                                angle=180 * theta / np.pi,
                                alpha=alpha_value, edgecolor='r', facecolor='none',
                                label=f'Scale {scale}' if scale == 1 else None
                            )
                            ax[ax_i, h_i + 1].add_patch(ellipse)
                            theta_rad = np.deg2rad(180 * theta / np.pi)
                            dx = (height / 2) * np.sin(theta_rad)
                            dy = (height / 2) * np.cos(theta_rad)
                            ax[ax_i, h_i + 1].text(
                                com[0] + dx, com[1] - dy, f'{scale}$\\sigma$',
                                color='darkred', fontsize=6, ha='center', va='center',
                                rotation=theta
                            )

                        ax[ax_i, h_i + 1].set_title(f"{hemi} Depth bin = {d_i}")
                        ax[ax_i, h_i + 1].set_xlim([np.min(ellipse_data['x'])-1,
                                                     np.max(ellipse_data['x'])+1])
                        ax[ax_i, h_i + 1].set_ylim([np.min(ellipse_data['y'])-1,
                                                     np.max(ellipse_data['y'])+1])
                        ax[ax_i, h_i + 1].axis('off')
                        ax[ax_i, h_i + 1].set_aspect('equal')

        plt.tight_layout()


# ---------------------------------------------------------------------------
# Internal helpers (not for external use)
# ---------------------------------------------------------------------------

def _get_stats(statDetails, diffDetails, stat_analyses, diff_analyses, analysis):
    if len(stat_analyses[analysis]) > 1:
        labels = statDetails['labels'][stat_analyses[analysis][0]:stat_analyses[analysis][1]]
        colors = statDetails['colors'][stat_analyses[analysis][0]:stat_analyses[analysis][1]]
    else:
        labels = statDetails['labels'][stat_analyses[analysis][0]:]
        colors = statDetails['colors'][stat_analyses[analysis][0]:]
    return {'labels': labels, 'colors': colors}


def _get_stats_diffs(statDetails, diffDetails, stat_analyses, diff_analyses, analysis):
    STATS = _get_stats(statDetails, diffDetails, stat_analyses, diff_analyses, analysis)
    if len(stat_analyses[analysis]) > 1:
        DIFFS = {
            'statIDs': {key: diffDetails['statIDs'][key]
                        for key in diff_analyses[analysis]['list']},
            'colors':  diffDetails['colors'][diff_analyses[analysis]['ids'][0]:
                                             diff_analyses[analysis]['ids'][1]],
        }
    else:
        DIFFS = {
            'statIDs': {key: diffDetails['statIDs'][key]
                        for key in diff_analyses[analysis]['list']},
            'colors':  diffDetails['colors'][diff_analyses[analysis]['ids'][0]:],
        }
    return STATS, DIFFS
