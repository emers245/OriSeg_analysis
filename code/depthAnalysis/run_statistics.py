#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical correction and results-saving utilities for the OriSeg depth analysis.

Key entry point: apply_fdr_correction(...)
  Gathers all p-values from the analysis dictionaries, applies FDR correction,
  and writes corrected p-values back in-place under the 'corrected p-vals' key.
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# p-value gathering helpers
# ---------------------------------------------------------------------------

def gather_pvals(dictionary, keys_path, all_pvals, path_to_pvals,
                 exclude_analysis_types, exclude_rois, exclude_cond,
                 exclude_combinations):
    for key, value in dictionary.items():
        if key in exclude_analysis_types or key in exclude_rois or key in exclude_cond:
            continue

        new_keys_path = keys_path + [key]
        if isinstance(value, dict):
            gather_pvals(value, new_keys_path, all_pvals, path_to_pvals,
                         exclude_analysis_types, exclude_rois, exclude_cond,
                         exclude_combinations)
        elif 'p-vals' == key:
            exclude = False
            for combination in exclude_combinations:
                match = True
                for i, exclude_key in enumerate(combination):
                    if exclude_key != 'all' and (i >= len(new_keys_path) or
                                                  new_keys_path[i] != exclude_key):
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


def gather_avgRadialDiff_comparisons(dictionary, all_p_values, path_to_pvals):
    for key, value in dictionary.items():
        if isinstance(value, dict) and 'p-vals' not in value:
            gather_avgRadialDiff_comparisons(value, all_p_values, path_to_pvals)
        elif 'p-vals' in value:
            p_values = value['p-vals']
            if isinstance(p_values, np.ndarray):
                all_p_values.append(p_values.flatten())
                path_to_pvals.append((dictionary[key], 'p-vals'))


# ---------------------------------------------------------------------------
# Main FDR correction entry point
# ---------------------------------------------------------------------------

def apply_fdr_correction(dictionaries_to_search, avgRadialDiff_comparisons,
                         exclude_analysis_types=None, exclude_rois=None,
                         exclude_cond=None, exclude_combinations=None,
                         statCorrType='fdr_bh'):
    """
    Gather all p-values from the provided dictionaries, apply FDR correction,
    and write 'corrected p-vals' back into each dictionary in-place.

    Parameters
    ----------
    dictionaries_to_search : list of dict
        Analysis result dictionaries to search for 'p-vals' entries.
        Typically [avgDepthProfiles, avgDepthDiffs, avgRadialProfiles, avgRadialDiff]
        when exclude_indiv_ttest=True, or additionally [depthProfiles, diffProfiles]
        when exclude_indiv_ttest=False.
    avgRadialDiff_comparisons : dict
        Multi-sample radial bin comparison p-values.
    exclude_analysis_types : list, optional
    exclude_rois : list, optional
    exclude_cond : list, optional
    exclude_combinations : list of tuple, optional
    statCorrType : str
        Correction method passed to statsmodels multipletests (default 'fdr_bh').
    """
    if exclude_analysis_types is None:
        exclude_analysis_types = []
    if exclude_rois is None:
        exclude_rois = []
    if exclude_cond is None:
        exclude_cond = []
    if exclude_combinations is None:
        exclude_combinations = []

    all_pvals      = []
    path_to_pvals  = []

    for dictionary in dictionaries_to_search:
        gather_pvals(dictionary, [], all_pvals, path_to_pvals,
                     exclude_analysis_types, exclude_rois, exclude_cond,
                     exclude_combinations)

    gather_avgRadialDiff_comparisons(avgRadialDiff_comparisons, all_pvals, path_to_pvals)

    flattened_p_values   = np.concatenate(all_pvals)
    nan_mask             = np.isnan(flattened_p_values)
    non_nan_p_values     = flattened_p_values[~nan_mask]

    _, corrected_non_nan_p_values, _, _ = multipletests(
        non_nan_p_values, alpha=0.05, method=statCorrType
    )

    corrected_p_values             = np.full_like(flattened_p_values, np.nan)
    corrected_p_values[~nan_mask]  = corrected_non_nan_p_values

    idx = 0
    for dictionary, key in path_to_pvals:
        p_values = dictionary['p-vals']
        if hasattr(p_values, 'pvalue'):
            dictionary['corrected p-vals'] = (
                corrected_p_values[idx:idx + len(p_values.pvalue)]
                .reshape(p_values.pvalue.shape)
            )
            idx += len(p_values.pvalue)
        elif isinstance(p_values, np.ndarray):
            corrected_subset = (
                corrected_p_values[idx:idx + p_values.size]
                .reshape(p_values.shape)
            )
            idx += p_values.size
            valid_mask = ~np.isnan(p_values)
            dictionary['corrected p-vals'] = np.where(valid_mask, corrected_subset, np.nan)


# ---------------------------------------------------------------------------
# Results-saving helpers
# ---------------------------------------------------------------------------

def save_statistical_results(data_dict, alpha, statTestType, Npermutations=None,
                             output_csv='output.csv', binType='norm_depths'):
    avg       = data_dict.get('avg')
    sd        = data_dict.get('stdev')
    norm_bins = data_dict.get(binType)
    Nsamp     = data_dict.get('Nsamp')
    p_vals    = data_dict.get('corrected p-vals')

    data = {
        binType:  norm_bins,
        'avg':    avg,
        'sd':     sd,
        'alpha':  [alpha] * len(avg),
    }

    if statTestType == 't-test' and isinstance(p_vals, ttest_ind.__class__):
        data['df']             = p_vals.df
        data['test statistic'] = p_vals.statistic
        data['p-vals']         = p_vals.pvalue
    elif statTestType == 'permutation' and isinstance(p_vals, np.ndarray):
        data['df']             = [Nsamp] * len(avg)
        data['test statistic'] = avg
        data['p-vals']         = p_vals
        Npermutations_array    = Npermutations * np.ones(len(avg))
        Npermutations_array[Npermutations_array > 2**Nsamp] = 2**Nsamp
        data['Npermutations']  = Npermutations_array
    else:
        raise ValueError("Invalid input for 'statTestType' or 'corrected p-vals'")

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)


def save_2samp_results(data_dict, bins, alpha, statTestType, Npermutations=None,
                       output_csv='output.csv', binType='norm_depths'):
    p_vals = data_dict.get('corrected p-vals').flatten()
    df     = data_dict.get('df').flatten()
    p_vals = data_dict.get('corrected p-vals').flatten()

    X, Y       = np.meshgrid(bins, bins)
    all_pairs  = np.vstack([X.ravel(), Y.ravel()])

    data = {
        binType + ' 0': all_pairs[0, :],
        binType + ' 1': all_pairs[1, :],
        'df':    df,
        'alpha': [alpha] * len(df),
        'p-vals': p_vals,
    }

    if statTestType == 'permutation':
        Npermutations_array = Npermutations * np.ones(len(df))
        Npermutations_array[Npermutations_array < 2**(df+1)] = (
            2**(df[Npermutations_array < 2**(df+1)]+1)
        )
        data['Npermutations'] = Npermutations_array

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)


def save_as_df(profiles, roi_type, analysis_type, figDir, statTestType):
    for diff, data in profiles[roi_type][analysis_type].items():
        depth_bins = np.arange(len(data['avg']))
        if 'corrected p-vals' in data.keys():
            if statTestType == 't-test':
                df = pd.DataFrame({
                    'depth bin':       depth_bins,
                    'avg':             data['avg'],
                    'stdev':           data['stdev'],
                    'norm_depths':     data['norm_depths'],
                    't-statistic':     data['p-vals'].statistic,
                    'p-value':         data['p-vals'].pvalue,
                    'corrected p-value': data['corrected p-vals'],
                    'df':              data['p-vals'].df,
                    'N':               [data['Nsamp']] * len(data['avg']),
                })
                df.to_csv(os.path.join(figDir, f"{diff}_{roi_type}.csv"), index=False)
            elif statTestType == 'permutation':
                df = pd.DataFrame({
                    'depth bin':         depth_bins,
                    'avg':               data['avg'],
                    'stdev':             data['stdev'],
                    'norm_depths':       data['norm_depths'],
                    'p-value':           data['p-vals'],
                    'corrected p-value': data['corrected p-vals'],
                    'df':                data['Nsamp'] - 1,
                    'N':                 [data['Nsamp']] * len(data['avg']),
                })
                df.to_csv(os.path.join(figDir, f"{diff}_{roi_type}.csv"), index=False)
