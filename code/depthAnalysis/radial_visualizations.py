#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Radial visualizations

This contains functions for the main radial analysis visualizations, including:
- Plotting the average radial profiles for each condition and group
- Plotting the average difference radial profiles for each contrast of interest
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from oriseg_funcs import *

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
    """
    Plots the smoothed radial profile for each depth layer and condition.
    Parameters:
    - data: dict of DataFrames, each containing the data for a single run/subject
    - analysis_type: str, either 'task' or 'loc'
    - condition: str, the column name in the DataFrame to plot
    - kernel: str, the type of kernel to use for smoothing (e.g., 'gaussian')
    - mask: dict of boolean arrays, optional, to filter the data for each run/subject
    - smooth_factor: float, the bandwidth for the smoothing kernel
    - radMax: float, the maximum radial distance to plot
    - nRadii: int, the number of radial bins to use for the x-axis
    - ymin, ymax: float, the y-axis limits for the plot
    - fontsize: int, the font size for labels and titles
    - fcolor: str, the face color for the figure
    - lcolor: str, the color for labels and axes
    - depth_labels: list of str, the labels for each depth layer
    - depthBoundaries: array, the boundaries for each depth layer in terms of relative depth
    - plot_indiv: bool, whether to plot individual profiles for each run/subject
    - statColor: str, the color for the average profile and shaded error region
    - vline: float, the x-value at which to plot a vertical dashed line (e.g., to indicate the boundary of the ROI)
    Returns:
    - fig: the matplotlib figure object containing the plot"""

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

def plot_smoothed_radial_profile_wbins(data, avgRadialProfiles, analysis_type,
                                        condition, kernel, mask=None,
                                        smooth_factor=0.3, radMax=5, nRadii=20,
                                        ymin=-5, ymax=5, ymin_bar=-0.5, ymax_bar=3.0,
                                        fontsize=8, fcolor='white', lcolor='black',
                                        depth_labels=['deep', 'middle', 'superficial'],
                                        depthBoundaries=np.array([[0, 1/3], [1/3, 2/3], [2/3, 1]]),
                                        plot_indiv=True, statColor='gray', vline=2,
                                        pval_threshold=0.05, nRad=4, comparisons=None,
                                        figsize=(8, 10), ax_width=0.7, ax_height=0.15,
                                        ax_height_bar=0.06, ax_spacing=0.3, ax_left=0.15,
                                        ax_bottom=0.15, ax_subspacing=0.05):
    """
    Plots the smoothed radial profile for each depth layer and condition, along with bar plots of the average radial profiles in specified bins.
    Parameters:
    - data: dict of DataFrames, each containing the data for a single run/subject
    - avgRadialProfiles: dict containing the average radial profile data for each depth layer and condition, including means, standard deviations, sample sizes, and p-values
    - analysis_type: str, either 'task' or 'loc'
    - condition: str, the column name in the DataFrame to plot
    - kernel: str, the type of kernel to use for smoothing (e.g., 'gaussian')
    - mask: dict of boolean arrays, optional, to filter the data for each run/subject
    - smooth_factor: float, the bandwidth for the smoothing kernel
    - radMax: float, the maximum radial distance to plot
    - nRadii: int, the number of radial bins to use for the x-axis
    - ymin, ymax: float, the y-axis limits for the line plots
    - ymin_bar, ymax_bar: float, the y-axis limits for the bar plots
    - fontsize: int, the font size for labels and titles
    - fcolor: str, the face color for the figure
    - lcolor: str, the color for labels and axes
    - depth_labels: list of str, the labels for each depth layer
    - depthBoundaries: array, the boundaries for each depth layer in terms of relative depth
    - plot_indiv: bool, whether to plot individual profiles for each run/subject
    - statColor: str, the color for the average profile and shaded error region
    - vline: float, the x-value at which to plot a vertical dashed line (e.g., to indicate the boundary of the ROI)
    - pval_threshold: float, the threshold for significance when plotting p-value annotations
    - nRad: int, the number of radial bins to use for the bar plots
    - comparisons: dict, optional, containing pairwise comparison p-values for each depth layer, to annotate significant differences between bins
    - figsize: tuple, the size of the figure
    - ax_width, ax_height: float, the width and height of the line plot axes
    - ax_height_bar: float, the height of the bar plot axes
    - ax_spacing: float, the vertical spacing between the line plot axes
    - ax_left: float, the left position of the axes
    - ax_bottom: float, the bottom position of the first line plot axis
    - ax_subspacing: float, the additional vertical spacing between the line plot and bar plot for each depth layer
    Returns:
    - fig: the matplotlib figure object containing the plot
    """

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
        p.set_xticks(np.linspace(0, radMax, nRad+1), [])
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
            bar_p.set_xticks(np.linspace(0, radMax, nRad+1))
        else:
            bar_p.set_xticks(np.linspace(0, radMax, nRad+1), [])

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