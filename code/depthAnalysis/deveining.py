#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vein exclusion (deveining) for laminar fMRI depth analysis.

The deveining step identifies and removes voxels contaminated by draining veins
using mean-normalized variance (MNV). Deep cortical layers (far from the surface)
are used as a reference distribution because they have the lowest vascular density
and therefore the lowest expected MNV. Voxels whose log(MNV) exceeds the deep-layer
mean by more than sd_thresh standard deviations are flagged as vein-contaminated.

Importable function: devein_voxels(all_data, ...)
"""

import os
import sys
import numpy as np
import pandas as pd

funcs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if funcs_dir not in sys.path:
    sys.path.append(funcs_dir)

from oriseg_funcs import get_lmnv, get_deep_layer_dist, get_mnv_mask


def save_dropout_statistics_to_csv(vArea, subjIDs, hemi, dropout, voxels_before, voxels_after, out_dir='output'):
    os.makedirs(out_dir, exist_ok=True)

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

    mean_data = df.mean(numeric_only=True)
    std_data = df.std(numeric_only=True)

    mean_data['Subject ID'] = 'Mean'
    std_data['Subject ID'] = 'Std Dev'
    mean_data = pd.DataFrame([mean_data])
    std_data = pd.DataFrame([std_data])

    df = pd.concat([df, mean_data, std_data], ignore_index=True)

    filename = f"{vArea}_dropout_statistics.csv"
    filepath = os.path.join(out_dir, filename)
    df.to_csv(filepath, index=False)

    print(f"Saved dropout statistics to {filepath}")


def devein_voxels(all_data, depth_var='d_norm', deep_pct=10, sd_thresh=2, out_dir=None):
    """
    Normalizes cortical depth to 'd_norm', applies vein exclusion based on
    mean-normalized variance (MNV), and stores the boolean mask as 'no_vein'
    in each subject's DataFrame (in-place).

    Parameters
    ----------
    all_data : dict
        Per-subject DataFrames keyed by subject ID.
    depth_var : str
        Column name to use as the depth variable (default 'd_norm').
    deep_pct : int
        Percentile threshold to define the deep-layer reference distribution.
    sd_thresh : float
        Number of standard deviations above the deep-layer mean to use as
        the vein-exclusion threshold.
    out_dir : str or None
        If provided, dropout statistics CSVs are saved to this directory.

    Returns
    -------
    lmnv_dict : dict
        Per-subject threshold information: mean, std, thresh, deep_mean, deep_std.
    """
    # Normalize 'd' within each subject, visual region, subregion, and hemisphere
    for subjID, df in all_data.items():
        all_data[subjID]['d_norm'] = df.groupby(['Visual Region', 'Subregion', 'hemi'])['d'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )

    depth_groups = {'deep': [0.0, 1/3], 'middle': [1/3, 2/3], 'superficial': [2/3, 1.0]}
    lmnv_dict = {key: {'mean': 0, 'std': 0, 'thresh': 0, 'deep_mean': 0, 'deep_std': 0}
                 for key in all_data.keys()}

    for iR, vArea in enumerate(['V1', 'V23']):
        k_i = 0
        subjIDs = [subj for subj in all_data.keys()
                   if all_data[subj]['Visual Region'].str.contains(vArea).any()]
        dsets = []
        for iH, hemi in enumerate(['lh', 'rh']):
            dsets = dsets + [
                subj + hemi for subj in all_data.keys()
                if ((all_data[subj]['Visual Region'].str.contains(vArea)) &
                    (all_data[subj]['hemi'].str.contains(hemi))).any()
            ]
        Ndsets = len(dsets)
        dropout = {
            'superficial': np.zeros(Ndsets),
            'middle':      np.zeros(Ndsets),
            'deep':        np.zeros(Ndsets),
            'total':       np.zeros(Ndsets),
        }

        voxels_before = []
        voxels_after  = []
        all_hemis     = []
        all_subj      = []

        for iS, label in enumerate(subjIDs):
            df_all = all_data[label]

            for iH, hemi in enumerate(df_all['hemi'].unique()):

                if np.sum((df_all['Visual Region'] == vArea) &
                          (df_all['Subregion'] == 'tgt') &
                          (df_all['hemi'] == hemi)) != 0:

                    df = df_all[(df_all['Visual Region'] == vArea) &
                                (df_all['Subregion'] == 'tgt') &
                                (df_all['hemi'] == hemi)]

                    lmnv = get_lmnv(df, key='stdev_xerrts')
                    mnv  = np.exp(lmnv)

                    z = df[depth_var]
                    [deep_mean, deep_std, deep] = get_deep_layer_dist(df, depth_var, deep_pct)
                    lmnv_dict[label]['deep_mean'] = deep_mean
                    lmnv_dict[label]['deep_std']  = deep_std

                    [mnv_mask, lmnv_thresh] = get_mnv_mask(df, depth_var, deep_pct, sd_thresh)
                    lmnv_dict[label]['thresh'] = lmnv_thresh
                    lmnv_dict[label]['mean']   = np.mean(lmnv)
                    lmnv_dict[label]['std']    = np.std(lmnv)

                    voxels_before.append(np.size(mnv))
                    voxels_after.append(np.sum(mnv_mask))
                    all_hemis.append(hemi)
                    all_subj.append(label)

                    print("%d/%d Voxels Survive for %s %s %s"
                          % (np.sum(mnv_mask), np.size(mnv), label, vArea, hemi))
                    superficial_mask = (z >= depth_groups['superficial'][0])
                    middle_mask      = ((z >= depth_groups['middle'][0]) &
                                        (z <  depth_groups['middle'][1]))
                    deep_mask        = (z < depth_groups['deep'][1])
                    print("\t %d/%d Voxels Survive for superficial %s"
                          % (np.sum(mnv_mask * superficial_mask), np.sum(superficial_mask), label))
                    print("\t %d/%d Voxels Survive for middle %s"
                          % (np.sum(mnv_mask * middle_mask), np.sum(middle_mask), label))
                    print("\t %d/%d Voxels Survive for deep %s"
                          % (np.sum(mnv_mask * deep_mask), np.sum(deep_mask), label))
                    dropout['superficial'][k_i] = 1 - np.sum(mnv_mask * superficial_mask) / np.sum(superficial_mask)
                    dropout['middle'][k_i]      = 1 - np.sum(mnv_mask * middle_mask)      / np.sum(middle_mask)
                    dropout['deep'][k_i]        = 1 - np.sum(mnv_mask * deep_mask)        / np.sum(deep_mask)
                    dropout['total'][k_i]       = 1 - np.sum(mnv_mask)                    / np.size(mnv)

                    k_i += 1

                    condition = (
                        (all_data[label]['Visual Region'] == vArea) &
                        (all_data[label]['Subregion'] == 'tgt') &
                        (all_data[label]['hemi'] == hemi)
                    )
                    all_data[label].loc[condition, 'no_vein'] = mnv_mask

        print("Average total dropout rate: %s +/- %s"
              % (np.mean(dropout['total']), np.std(dropout['total'])))
        print("\t Superficial: %s +/- %s"
              % (np.mean(dropout['superficial']), np.std(dropout['superficial'])))
        print("\t Middle: %s +/- %s"
              % (np.mean(dropout['middle']), np.std(dropout['middle'])))
        print("\t Deep: %s +/- %s"
              % (np.mean(dropout['deep']), np.std(dropout['deep'])))

        if out_dir is not None:
            save_dropout_statistics_to_csv(
                vArea, all_subj, all_hemis, dropout,
                voxels_before, voxels_after, out_dir=out_dir
            )

    return lmnv_dict
