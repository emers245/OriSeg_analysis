#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Edited by JHE: 09/11/2025
# Modified to add one-sample t-tests (per depth x radius x condition),
# Benjamini-Hochberg correction and plotting of significance markers
# (JHE: 2025-09-16)

import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import nibabel
# Now do the classification
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score
import subprocess as sub # for reading output of 3dinfo
import scipy.stats as scipy_stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

def fix_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 8))

def pick_voxels(roi, nTotal, stat=None, radii=None, depth=None, depthBin='all', randomize=False, stat_criterion=0.05, stat_type='p', rng=None):
    ''' We want a function for returning a list of Booleans that will sample
    from the rows of our X matrix. The X matrix will be nSamples x nVoxels.
    This function returns a Boolean to index the column dimension of X.
    '''
    #set RNG
    if not rng:
        rng = np.random
    
    # first, restrit our ROI to voxels that will be useful, in the desired rad
    if depth is None:
        depth = np.ones(roi.shape)
    restricted_roi = (roi > radii[0])*(roi <= radii[1])*(depth > 0)
    # depth is a little annoying, because we don't know where the splits are
    all_depths = depth[restricted_roi > 0]
    d_idx = np.argsort(all_depths)
    
    if depthBin == 'deep':
        depthRange = [0, all_depths[d_idx[int(len(all_depths)/3.)]]]
    elif depthBin == 'middle':
        depthRange = [all_depths[d_idx[int(len(all_depths)/3.)]],
                      all_depths[d_idx[int(2*len(all_depths)/3.)]]]
    elif depthBin == 'superficial':
        depthRange = [all_depths[d_idx[int(2*len(all_depths)/3.)]], 1]
    else:
        depthRange = [0, 1]
        
    restricted_roi *= (depth > depthRange[0])*(depth <= depthRange[1])
    print('%d voxels available in %s bin, %d-%d radius' %(np.sum(restricted_roi),
                                                          depthBin,
                                                          radii[0],
                                                          radii[1]))
    if np.sum(restricted_roi): # we have voxels!
        # next, pick out our nTotal voxels
        if nTotal < np.sum(restricted_roi):
            try:
                if not randomize: #take the the top nKeep voxels sorted by stat
                    stat_roi = stat[restricted_roi > 0]
                    sorted_stat = stat_roi.copy()
                    sorted_stat.sort()
                    if stat_type == 'F' or stat_type == 'T':
                        cutoff_stat = sorted_stat[len(sorted_stat) - nKeep - 1]
                        use_voxels = (stat > cutoff_stat)*restricted_roi
                    elif stat_type == 'p':
                        cutoff_stat = sorted_stat[nKeep]
                        use_voxels = (stat < cutoff_stat)*restricted_roi
                    else:
                        print("pick_voxels: Unknown stat_criterion")
                else:
                    if stat_type == 'T' or stat_type == 'F':
                        stat_mask = (stat >= stat_criterion)
                    elif stat_type == 'p':
                        stat_mask = (stat <= stat_criterion)
                    else:
                        print("pick_voxels: Unknown stat_criterion")
                    true_indices = np.argwhere(stat_mask * restricted_roi)
                    chosen_indices = true_indices[rng.choice(true_indices.shape[0], nKeep, replace=False)]
                    new_mask = np.zeros_like(stat_mask, dtype=bool)
                    for idx in chosen_indices:
                        new_mask[tuple(idx)] = True
                    use_voxels = new_mask*restricted_roi
                        
            except: # awful logic to handle stat=None --> random
                use_voxels = np.zeros(restricted_roi.shape) > 0
                idx = np.where(restricted_roi)
                random_picks = np.random.choice(np.arange(len(idx[0])),
                                                nTotal,
                                                replace=False)
                for iP in random_picks:
                    use_voxels[idx[0][iP], idx[1][iP], idx[2][iP]] = True
        else:
            use_voxels = restricted_roi > 0
    else:
        use_voxels = np.zeros(roi.shape)

    return use_voxels[roi > 0] 

def get_pvals(glmFile, Fstats, Fbrik):
    ''' given a file name for a GLM (so we can use 3dinfo to pull out
    degrees of freedom) and a vector of Fs, return a vector of ps
    
    Inputs:
        glmFile (str): path to GLM nifti file
        Fstats (array-like): an array of F-statistics
        Fbrik (int): position of F-statistic in stat array; must be > 0
    Returns:
        pvalues (array-like): p-values formatted as the shape of the Fstats
    '''
    info = sub.check_output('3dinfo -verbose %s' %glmFile, shell=True, text=True)
    lines = info.split('\n')
    for li, l in enumerate(lines):
        if 'sub-brick #%d' %(Fbrik) in l:
            statpar_i = lines[li+1].find('statpar =')
            statpar = lines[li+1][statpar_i:]
            temp = statpar[10::] # changed from statpar[10:-1]
            DOF = [int(d) for d in temp.split(' ')]
    # now that we have DOF, create a p
    if Fbrik >= 0:
        # this will work for everything except Benson atlas info
        pvals = 1-scipy_stats.f.cdf(Fstats,
                                    DOF[0],
                                    DOF[1])
    else:
        pvals = np.zeros(np.shape(Fstats))
    return pvals     

# Set random seed
seed = 4938
np.random.seed(seed)

# Set Variables
mainDir = '/home/scat-raid3/data/oriSeg_redo'
savedir = os.path.join(mainDir, 'orientation_classification')
figdir = os.path.join(savedir, 'figures')
figFileType = 'svg'

subjIDs_included = ["pnr143", "pnr256", "pnr328", "pnr495", "pnr510", "pnr579", "pnr668", "pnr685", "pnr713", "pnr739", "pnr756", "pnr822"]
subjDirs = [os.path.join(mainDir, subjIDs_included[i]) for i in range(len(subjIDs_included))] #glob.glob(os.path.join(mainDir, 'pnr???'))
subjDirs.sort()
roiNames = ['V1_tgt_lh_rad10_filled_deveined', 'V1_tgt_rh_rad10_filled_deveined']
dataType = 'unwarp'
scanType = 'task'
if scanType == 'localizer':
    conditions = ['center', 'surround']
    scanNames = ['run00_loc.nii', 'run03_loc.nii', 'run06_loc.nii']
elif scanType == 'task':
    #conditions = ['iso0', 'iso90', 'orth', 'sur']
    conditions = ['iso0', 'sur']
    scanNames = ['run01_task.nii', 'run02_task.nii', 'run04_task.nii', 'run05_task.nii']

sort_by = 'p' # 'T', 'F', 'random', 'p'
randomize = True # if true, select voxels randomly above some statistical criterion
stat_criterion = 0.01 # statistical criterion for T or F stat above/below which to select voxels randomly
nKeep = 10
nSamples = 50 # if randomizing voels used in SVM, number of resamples to do

depth_labels = ['deep', 'middle', 'superficial']
#radius_bands = [[0, 2], [2, 4], [4, 6]]
radius_bands = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
nDepths = len(depth_labels)
nRadii = len(radius_bands)

#%% pick up the data and pull out the data for N TRs after that
results = {'subj': [], 'roi': [], 'nVoxels': []}
for condition in conditions:
    results[condition] = []
plt.close('all')
for subjDir in tqdm(subjDirs, desc="Subject Number:"):
    plt.figure()
    for iROI, roiName in enumerate(roiNames):
        roi_file = os.path.join(mainDir, subjDir, 'rois', roiName + '.nii')
        depth_file = os.path.join(mainDir, subjDir, 'depth', 'laynii_manual.nii')
        glm_file = os.path.join(mainDir, subjDir, 'glm', 'stats_task_%s.nii' %dataType)
        
        if (os.path.exists(roi_file)) and (os.path.exists(depth_file)) and os.path.exists(glm_file):
            results['subj'].append(os.path.split(subjDir)[1])
            results['roi'].append(roiName) 
            classification_accuracy = {cond: {'nVoxels': np.zeros((nDepths, nRadii, nSamples)),
                                              'accuracy': np.zeros((nDepths, nRadii, nSamples)),
                                              'acc_std': np.zeros((nDepths, nRadii, nSamples))} for cond in conditions}

            #pick up data
            depth = nibabel.load(depth_file).get_fdata()
            roi = nibabel.load(roi_file).get_fdata()
            if np.sum(roi): # pnr510 has an empty deveined file ...
                stats = nibabel.load(glm_file).get_fdata()
                
                nVoxels = int(np.sum(roi > 0)) # JHE: Changed to reflect actual number of voxels
                
                if sort_by == 'F':
                    statBrik = stats[:, :, :, 0, -1]
                elif sort_by == 'T':
                    statBrik = stats[:, :, :, 0, -2]
                elif sort_by == 'p':
                    FBrik = stats[:, :, :, 0, -1]
                    stat_len = np.shape(stats)[4]
                    stat_ind = stat_len-1
                    statBrik = get_pvals(glm_file, FBrik, stat_ind)
                else:
                    statBrik = None
                #%% Loop through conditions (then depths and radii) to test classification accuracy
                
                # fig = plt.figure(num=1, figsize=(6, 9))
                # fig.clf()
                
                for iC, condition in enumerate(conditions):
                    theta_labels = ['0', '22', '45', '67', '90', '112', '135', '157']
                    nThetas = len(theta_labels)
                    
                    # TRs to grab after each stim onset
                    TR_range = [2, 6]
                    # pick up the timing files to get onset times for each condition
                    onsets = {}
                    for theta in theta_labels:
                        cond_label = '%s_%s' %(condition, theta)
                        timing_file = os.path.join(mainDir,
                                                   subjDir,
                                                   'experiment',
                                                   '%s.txt' %cond_label)
                        with open(timing_file, 'r') as fid:
                            lines = fid.readlines()
                        # this logic skips duplicates, which only happens for 1 surr theta
                        onsets[cond_label] = [float(l.strip().split()[0]) for l in lines]
                        if len(onsets[cond_label]) != len(scanNames):
                            print('Oh no! We do not have the right number of lines in %s' %cond_label)
                
                    columns = ['scan', 'stim', 'TR', 'roi']
                    
                    extracted_data = {col: [] for col in columns}
                    for iS, scan in enumerate(scanNames):
                        fileName = os.path.join(mainDir, subjDir, 'func', dataType, 'scaled', scan)
                        data = nibabel.load(fileName).get_fdata()
                        for theta in theta_labels:
                            cond_label = '%s_%s' %(condition, theta)
                            onset_TR = round(onsets[cond_label][iS]/2)
                            for iT in range(onset_TR + TR_range[0], onset_TR + TR_range[1]):
                                extracted_data['scan'].append(scan)
                                extracted_data['stim'].append(cond_label)
                                extracted_data['TR'].append(iT)
                                extracted_data['roi'].append(data[:, :, :, iT][roi > 0.])
                                    
                    
                    X = np.asarray(extracted_data['roi'])
                    y = extracted_data['stim']
                    # scramble labels for reality check
                    #y = [y[iScr] for iScr in np.random.permutation(len(y))]
                    
                    for iSamp in range(nSamples):
                        rng = np.random.RandomState(iSamp)
                        for iD, depth_label in enumerate(depth_labels):
                            for iR, radii in enumerate(radius_bands):
                        
                                print('Taking %d %s voxels from r=%d to r=%d after sorting by %s' %(np.min((nKeep, nVoxels)),
                                                                                  depth_label,
                                                                                  radii[0],
                                                                                  radii[1],
                                                                                  sort_by))
                                use_voxels = pick_voxels(roi, 
                                                         nKeep, 
                                                         stat=statBrik, 
                                                         radii=radii, 
                                                         depth=depth, 
                                                         depthBin=depth_label, 
                                                         randomize=randomize, 
                                                         stat_criterion=stat_criterion,
                                                         stat_type=sort_by,
                                                         rng=rng)
                    
                                if np.sum(use_voxels):
                                    B = X[:, use_voxels]
                                    
                                    # Create a svm Classifier
                                    clf = svm.SVC(probability=True) 
                                    
                                    # Create the rules for n_splits train/test splits for cross validation
                                    nFold = 10
                                    cv = StratifiedKFold(n_splits=nFold, shuffle=True, random_state=42)
                                    
                                    # Use sklearn's cros_val_score to run the fit/test the SVC n_splits times
                                    acc_scores = cross_val_score(clf, B, y, cv=cv, scoring="accuracy")
                                    print(f"  {nFold:d}-fold CV accuracy: {acc_scores.mean():.3f} ± {acc_scores.std()/np.sqrt(nFold):.3f}")
                                    classification_accuracy[condition]['nVoxels'][iD, iR, iSamp] = np.sum(use_voxels)
                                    classification_accuracy[condition]['accuracy'][iD, iR, iSamp] = acc_scores.mean()
                                    classification_accuracy[condition]['acc_std'][iD, iR, iSamp] = acc_scores.std()
                                else:
                                    classification_accuracy[condition]['nVoxels'][iD, iR, iSamp] = 0
                                    classification_accuracy[condition]['accuracy'][iD, iR, iSamp] = np.nan
                                    classification_accuracy[condition]['acc_std'][iD, iR, iSamp] = np.nan
                                
                    plt.subplot(len(conditions) + 1, 2, 1 + iC*2 + iROI)
                    plt.imshow(classification_accuracy[condition]['accuracy'].mean(axis=-1), vmin=0, vmax=1)
                    plt.ylim([-.5, 2.5])
                    plt.yticks([0, 1, 2], depth_labels)
                    plt.xticks(range(len(radius_bands)), [])
                    plt.title('Classification accuracy, %s' %condition)
                    results[condition].append(classification_accuracy[condition]['accuracy'].mean(axis=-1))
                plt.subplot(len(conditions) + 1, 2, len(conditions)*2 + 1 + iROI)
                plt.imshow(classification_accuracy[condition]['nVoxels'].mean(axis=-1), vmin=0, vmax=nKeep)
                plt.ylim([-.5, 2.5])
                plt.title('actual # voxels')
                plt.suptitle('%s: %d voxels picked by %s with %s repeats' %(os.path.split(subjDir)[1],
                                                            nKeep,
                                                            sort_by,
                                                            nSamples))
                results['nVoxels'].append(classification_accuracy[conditions[0]]['nVoxels'].mean(axis=-1))
        else:
            print('Missing data for %s, %s' %(subjDir, roiName))
#%% expand out the data
results_to_write = {'subj': [],
                    'roi': [],
                    'cond': [],
                    'depth': [],
                    'radius_band': [],
                    'accuracy': [],
                    'nVoxels': []}
for iS, subj in enumerate(results['subj']):
    for iD, depth in enumerate(depth_labels):
        for iR, radii in enumerate(radius_bands):
            try:
                for condition in conditions:
                    results_to_write['accuracy'].append(results[condition][iS][iD, iR])
                    results_to_write['subj'].append(subj)
                    results_to_write['roi'].append(results['roi'][iS])
                    results_to_write['cond'].append(condition)
                    results_to_write['depth'].append(depth)
                    results_to_write['radius_band'].append('%d-%d' %(radii[0], radii[1]))
                    results_to_write['nVoxels'].append(results['nVoxels'][iS][iD, iR])
            except:
                print('no data for %s' %subj)
# write these data down
if sort_by == 'random':
    selection_method = 'random'
else:
    if not randomize:
        selection_method = 'top'
    else:
        selection_method = 'randomTop'
results_csv = os.path.join(mainDir,
                           'orientation_classification',
                           'accuracy_%s_%dbands_%d_%s_voxels.csv' %(scanType,
                                                                    len(radius_bands),
                                                                    nKeep,
                                                                    selection_method))
df = pd.DataFrame(results_to_write)
df.to_csv(results_csv)
#%% Skip this cell 
load_csv = False
if load_csv:
    # set the csv name here, and this chunk of code will pick up the specified
    # accuracy*.csv and reshape it to a results structure to use the plotting
    # code below
    if sort_by == 'random':
        csv_filename = f'accuracy_{scanType}_{len(radius_bands)}bands_{nKeep}_random_voxels.csv'
    elif sort_by == 'F' or sort_by == 'T' or sort_by == 'p':
        if randomize:
            csv_filename = f'accuracy_{scanType}_{len(radius_bands)}bands_{nKeep}_randomTop_voxels.csv'
        else:
            csv_filename = f'accuracy_{scanType}_{len(radius_bands)}bands_{nKeep}_top_voxels.csv'
    else:
        print("Unknown sort_by")
    results_csv = os.path.join(mainDir,
                               'orientation_classification',
                               csv_filename)
    df = pd.read_csv(results_csv)
    conditions = list(np.unique(df['cond']))
    results = {'subj': [], 'roi': [], 'nVoxels': []}
    for condition in conditions:
        results[condition] = []
    subj_list = np.unique(df.subj)
    for subj in subj_list:
        subj_df = df[df.subj == subj]
        roi_list = np.unique(subj_df.roi)
        for roi in roi_list:
            roi_df = subj_df[subj_df.roi == roi]
            results['subj'].append(subj)
            results['roi'].append(roi)
            results['nVoxels'].append(np.zeros((len(depth_labels),
                                                len(radius_bands))))
            for condition in conditions:
                results[condition].append(np.zeros((len(depth_labels),
                                                    len(radius_bands))))
                cond_df = roi_df[roi_df['cond'] == condition]
                for iR, radii in enumerate(radius_bands):
                    rad_str = '%d-%d' %(radii[0], radii[1])
                    rad_df = cond_df[cond_df['radius_band'] == rad_str]
                    for iD, depth in enumerate(depth_labels):
                        results[condition][-1][iD, iR] = rad_df[rad_df['depth'] == depth]['accuracy'].iloc[0]
                        results['nVoxels'][-1][iD, iR] = rad_df[rad_df['depth'] == depth]['nVoxels'].iloc[0]
#%%
# -------------------------- NEW / MODIFIED PLOTTING & STAT TESTS ---------------------------- #
class_max = .5
class_chance = 0.125
for hemi in [0,1]:
    plt.figure(figsize=(9, 9))
    plt.suptitle('%d voxels picked by %s' %(nKeep, sort_by))
    
    if hemi == 0:
        hemi_name = 'lh'
    elif hemi == 1:
        hemi_name = 'rh'
    else:
        print("Unknown hemi index")
    exclude_small = False
    
    # prepare colors for depths and mapping
    depth_colors = ['tab:blue', 'tab:orange', 'tab:green']  # deep, middle, superficial (keeps colors consistent)
    
    # We'll compute p-values for every condition x depth x radius (excluding 'nVoxels'),
    # then apply BH across all tests for all conditions (as user requested).
    all_tests = []   # list of (condition, depth_idx, radius_idx, p_raw)
    # To store p-values in structured dict for later mapping
    pvals_dict = {cond: np.full((nDepths, nRadii), np.nan) for cond in conditions}
    
    # First gather and test (across subjects) using the same 'use_data' selection you used for plotting
    # Note: 'use_data' in your plotting is a list of per-ROI arrays; we follow same indexing (hemi stride)
    # We'll build per-condition data arrays shaped (nSubjects_used, nDepths, nRadii)
    per_condition_data = {}
    for condition in conditions:
        if exclude_small:
            use_data = [results[condition][idx] for idx in range(hemi, len(results[condition]), 2) if np.sum(results['nVoxels'][idx] == nKeep) == nRadii*nDepths]
        else:
            use_data = [results[condition][idx] for idx in range(hemi, len(results[condition]), 2)]
        data_array = np.asarray(use_data)  # shape (nSubjects, nDepths, nRadii)
        per_condition_data[condition] = data_array
        # for each cell, run 1-sample t-test vs class_chance
        if data_array.size == 0:
            continue
        nSubj = data_array.shape[0]
        for iD in range(nDepths):
            for iR in range(nRadii):
                arr = data_array[:, iD, iR]
                # drop nans
                arr_clean = arr[~np.isnan(arr)]
                if arr_clean.size < 2:
                    p = np.nan
                else:
                    tstat, p = scipy_stats.ttest_1samp(arr_clean, popmean=class_chance)
                pvals_dict[condition][iD, iR] = p
                all_tests.append((condition, iD, iR, p))
               
    # Do correction for multiple comparisons using FDR-BH correction
    raw_pvals = [t[3] for t in all_tests]
    rej, pvals_corr, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
    pvals_adj_dict = {cond: np.full((nDepths, nRadii), np.nan) for cond in conditions}
    for idx, (condition, iD, iR, p_raw) in enumerate(all_tests):
        pvals_adj_dict[condition][iD, iR] = pvals_corr[idx]
    
    # Now plotting (with significance stars)
    for iC, condition in enumerate(conditions + ['nVoxels']):
        plt.subplot(len(conditions) + 1, 2, 1 + iC*2)
        if exclude_small:
            use_data = [results[condition][idx] for idx in range(hemi, len(results[condition]), 2) if np.sum(results['nVoxels'][idx] == nKeep) == nRadii*nDepths]
        else:
            use_data = [results[condition][idx] for idx in range(hemi, len(results[condition]), 2)]
        if iC < len(conditions):
            vmax = class_max
            vmin = class_chance
        else:
            vmax = nKeep
            vmin = 0
        plt.imshow(np.nanmean(np.asarray(use_data), axis=0), vmin=vmin, vmax=vmax) 
        plt.ylim([-.5, 2.5])
        plt.yticks([0, 1, 2], depth_labels)
        plt.xticks(range(len(radius_bands)), [])
        if condition != 'nVoxels':
            plt.title('Classification accuracy, %s' %condition)
        else:
            plt.title('voxels available in sub-ROI')
            plt.xticks(range(len(radius_bands)),
                       ['%d-%dmm' %(r[0], r[1]) for r in radius_bands],
                       rotation=45)
        # line graph
        ax = plt.subplot(len(conditions) + 1, 2, 2 + iC*2)
        # extract same data_array
        if condition != 'nVoxels':
            data_array = per_condition_data[condition]  # (nSubjects, nDepths, nRadii)
        else:
            # build an array of nVoxels counts
            if exclude_small:
                use_nv = [results['nVoxels'][idx] for idx in range(hemi, len(results['nVoxels']), 2) if np.sum(results['nVoxels'][idx] == nKeep) == nRadii*nDepths]
            else:
                use_nv = [results['nVoxels'][idx] for idx in range(hemi, len(results['nVoxels']), 2)]
            data_array = np.asarray(use_nv)  # shape (nSubjects, nDepths, nRadii)
        if data_array.size == 0:
            continue
        nSubjects_used = data_array.shape[0]
    
        # compute means and sems across subjects for plotting
        means = np.nanmean(data_array, axis=0)  # (nDepths, nRadii)
        sems = np.nanstd(data_array, axis=0) / np.sqrt(np.sum(~np.isnan(data_array), axis=0))  # handle nan counting
    
        # plot each depth with consistent color
        for iD in range(nDepths):
            plt.errorbar(np.arange(len(radius_bands)),
                         means[iD, :], 
                         sems[iD, :], 
                         label=depth_labels[iD],
                         color=depth_colors[iD],
                         marker='o')
        plt.ylim([0, class_max if condition != 'nVoxels' else nKeep*1.1])
        plt.xticks(range(len(radius_bands)), [])
        plt.ylabel('SVM accuracy' if condition != 'nVoxels' else '# voxels')
        if condition != 'nVoxels':
            plt.plot([0, len(radius_bands)-1], [class_chance, class_chance], 'k--')
            if iC == 1:
                plt.text(1, 0.02, 'n=%d %s ROIs' %(nSubjects_used, ['lh', 'rh'][hemi]))
    
        # Plot significance stars for each depth x radius (only for accuracy conditions)
        if condition != 'nVoxels':
            # p_adj map for this condition
            p_adj_map = pvals_adj_dict[condition]  # shape (nDepths, nRadii)
            # For each (iD, iR) show star if adjusted p < thresholds
            for iD in range(nDepths):
                for iR in range(nRadii):
                    p_adj = p_adj_map[iD, iR]
                    if np.isnan(p_adj):
                        continue
                    # choose symbol according to thresholds
                    if p_adj < 0.001:
                        stars = '***'
                    elif p_adj < 0.01:
                        stars = '**'
                    elif p_adj < 0.05:
                        stars = '*'
                    else:
                        stars = None
                    if stars:
                        # compute y position: mean + sem + small offset
                        y = np.max(means[:, iR])
                        y_i = np.argmax(means[:,iR])
                        yerr = sems[y_i, iR] if not np.isnan(sems[y_i, iR]) else 0
                        offset = 0.02 * class_max
                        y_star = y + yerr + offset + iD*(0.03 * class_max)
                        # place text
                        ax.text(iR, y_star, stars, ha='center', va='bottom', color=depth_colors[iD], fontsize=10, fontweight='bold')
    
    plt.legend(depth_labels)
    # final adjustments for the last subplot (nVoxels)
    plt.ylim([0, nKeep*1.1])
    plt.ylabel('# voxels used to classify')
    plt.xticks(range(len(radius_bands)),
               ['%d-%d$\sigma$' %(r[0], r[1]) for r in radius_bands],
               rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if sort_by == 'random':
        fig_filename = f'svm_accuracy_{scanType}_{len(radius_bands)}bands_{nKeep}_random_voxels_{hemi_name}.{figFileType}'
    elif sort_by == 'F' or sort_by == 'T' or sort_by == 'p':
        if randomize:
            fig_filename = f'accuracy_{scanType}_{len(radius_bands)}bands_{nKeep}_randomTop_voxels_{hemi_name}.{figFileType}'
        else:
            fig_filename = f'accuracy_{scanType}_{len(radius_bands)}bands_{nKeep}_top_voxels_{hemi_name}.{figFileType}'
    else:
        print("Unknown sort_by")
    plt.savefig(os.path.join(figdir,fig_filename))
# ------------------------------------------------------------------------------------------- #
# Save statistics to a CSV file
stats_to_write = {'condition': [],
                  'depth': [],
                  'radius_band': [],
                  'avg_accuracy': [],
                  'std_accuracy': [],
                  'nSubjects': [],
                  'p_raw': [],
                  'p_adj': []}
for condition in conditions:
    data_array = per_condition_data[condition]  # (nSubjects, nDepths, nRadii)
    if data_array.size == 0:
        continue
    nSubjects_used = data_array.shape[0]
    for iD in range(nDepths):
        for iR in range(nRadii):
            arr = data_array[:, iD, iR]
            arr_clean = arr[~np.isnan(arr)]
            avg_acc = np.nanmean(arr_clean)
            std_acc = np.nanstd(arr_clean)
            p_raw = pvals_dict[condition][iD, iR]
            p_adj = pvals_adj_dict[condition][iD, iR]
            stats_to_write['condition'].append(condition)
            stats_to_write['depth'].append(depth_labels[iD])
            stats_to_write['radius_band'].append('%d-%d' %(radius_bands[iR][0], radius_bands[iR][1]))
            stats_to_write['avg_accuracy'].append(avg_acc)
            stats_to_write['std_accuracy'].append(std_acc)
            stats_to_write['nSubjects'].append(len(arr_clean))
            stats_to_write['p_raw'].append(p_raw)
            stats_to_write['p_adj'].append(p_adj)
stats_csv = os.path.join(mainDir,
                         'orientation_classification',
                         'summary_accuracy_stats_%s_%dbands_%d_%s_voxels.csv' %(scanType,
                                                          len(radius_bands),
                                                          nKeep,
                                                          hemi_name))
df_stats = pd.DataFrame(stats_to_write)
df_stats.to_csv(stats_csv)