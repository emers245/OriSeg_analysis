#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:31:44 2024

@author: Joe
"""

import numpy as np

def permute_1samp(data, test_stat, null_stat=0, n_permutations=1000, test_type='two-sided', axis=0):
    # Compute the actual test statistic for the given data
    actual_stat = test_stat(data, axis=axis)
    
    # Determine the maximum number of unique permutations
    max_permutations = 2 ** data.shape[axis]
    if n_permutations > max_permutations:
        n_permutations = max_permutations
    
    # Initialize a set to store unique sign flip patterns and an array to store the permutation test statistics
    unique_sign_patterns = set()
    perm_stats = np.zeros((n_permutations,) + actual_stat.shape)
    
    # Generate unique permutations by randomly flipping the signs of the observations
    i = 0
    while i < n_permutations:
        sign_flips = tuple(np.random.choice([-1, 1], size=data.shape[axis], replace=True))
        if sign_flips not in unique_sign_patterns:
            unique_sign_patterns.add(sign_flips)
            expanded_sign_flips = np.expand_dims(np.array(sign_flips), axis=tuple(range(1, data.ndim)))
            permuted_data = data * expanded_sign_flips
            perm_stats[i] = test_stat(permuted_data, axis=axis)
            i += 1
    
    # Compute p-values based on the null distribution
    if test_type == 'two-sided':
        p_values = np.mean(np.abs(perm_stats - null_stat) >= np.abs(actual_stat - null_stat), axis=0)
    elif test_type == 'one-sided-positive':
        p_values = np.mean(perm_stats >= actual_stat, axis=0)
    elif test_type == 'one-sided-negative':
        p_values = np.mean(perm_stats <= actual_stat, axis=0)
    else:
        raise ValueError("Invalid test_type. Choose from 'two-sided', 'one-sided-positive', or 'one-sided-negative'.")
    
    return p_values

# Example usage
if __name__ == "__main__":
    data = np.random.randn(11, 7)  # Example data array with shape (11 subjects, 7 depth bins)
    test_stat = np.mean  # Define the test statistic (mean)
    p_values = permute_1samp(data, test_stat, null_stat=0, n_permutations=5000, test_type='two-sided', axis=0)
    print("P-values:", p_values)
    
    #%% demo
    import matplotlib.pyplot as plt
    
    # Set variables
    axis = 0
    null_stat = 0
    n_permutations = 5000
    test_type = 'two-sided'
    
    # Compute the actual test statistic for the given data
    actual_stat = test_stat(data, axis=axis)
    
    # Determine the maximum number of unique permutations
    max_permutations = 2 ** data.shape[axis]
    if n_permutations > max_permutations:
        n_permutations = max_permutations
    
    # Initialize a set to store unique sign flip patterns and an array to store the permutation test statistics
    unique_sign_patterns = set()
    perm_stats = np.zeros((n_permutations,) + actual_stat.shape)
    
    # Generate unique permutations by randomly flipping the signs of the observations
    i = 0
    while i < n_permutations:
        sign_flips = tuple(np.random.choice([-1, 1], size=data.shape[axis], replace=True))
        if sign_flips not in unique_sign_patterns:
            unique_sign_patterns.add(sign_flips)
            expanded_sign_flips = np.expand_dims(np.array(sign_flips), axis=tuple(range(1, data.ndim)))
            permuted_data = data * expanded_sign_flips
            perm_stats[i] = test_stat(permuted_data, axis=axis)
            i += 1
    
    # Compute p-values based on the null distribution
    if test_type == 'two-sided':
        p_values = np.mean(np.abs(perm_stats - null_stat) >= np.abs(actual_stat - null_stat), axis=0)
    elif test_type == 'one-sided-positive':
        p_values = np.mean(perm_stats >= actual_stat, axis=0)
    elif test_type == 'one-sided-negative':
        p_values = np.mean(perm_stats <= actual_stat, axis=0)
    else:
        raise ValueError("Invalid test_type. Choose from 'two-sided', 'one-sided-positive', or 'one-sided-negative'.")
    
    # Plot
    plt.figure()
    plt.hist(data[:,0])
    plt.plot((np.mean(data[:,0]),np.mean(data[:,0])),[0,5],label='mean')
    plt.legend()
    plt.xlabel('beta')
    plt.ylabel("N subjects")
    plt.title('Example Data N = 11')
    
    plt.figure()
    plt.hist(permuted_data[:,0])
    plt.plot((np.mean(permuted_data[:,0]),np.mean(permuted_data[:,0])),[0,5],label='mean')
    plt.legend()
    plt.xlabel('beta')
    plt.ylabel("N subjects")
    plt.title('Example Permuted Data N = 11')
    
    plt.figure()
    plt.hist(perm_stats[:,0],density=True)
    plt.plot((np.mean(data[:,0]),np.mean(data[:,0])),[0,1.2],label='sample mean')
    plt.legend()
    plt.xlabel('mean beta')
    plt.ylabel("Probability Density")
    plt.title(f'Null Perm Distribution: p={p_values[0]:.3f}')
    
    #%% demo real data
    import scipy.stats as stats
    
    # This is real data from 12 subjects in oriSeg
    # The data show the average contrast between iso90 and iso0 at 7 depth bins
    # within 1 sigma of the ROI center.
    # data = np.array([[ 0.11385393,  0.85136237,  0.89494879,  0.29626056,  0.04201962,
    #         -0.21610284,  1.33824563],
    #         [0.67280523, 0.90490531, 1.28134039, 1.13945317, 1.53784265,
    #         2.11568239, 1.90729537],
    #         [0.2912803 , 0.34224837, 0.63692313, 0.65164465, 0.40375908,
    #         0.3630618 , 0.32213609],
    #         [-0.07124485,  0.25011545,  0.31451511,  0.60802978,  0.92885662,
    #          0.94714795,  1.00025038],
    #         [-0.44573492, -0.05678238,  0.18242548,  0.27020932,  0.08328955,
    #          0.54976705,  0.69116779],
    #         [0.1096833 , 0.32511606, 0.35476731, 0.36805928, 0.59781865,
    #         0.3313378 , 0.59154126],
    #         [0.69060173, 0.59761145, 1.48238951, 1.51871169, 2.17546352,
    #         2.58659705, 2.17808414],
    #         [0.8315821 , 0.42623093, 0.70112576, 1.20562157, 1.33024523,
    #         1.06513032, 1.3995008 ],
    #         [ 0.17366902,  0.09817047,  0.10739324,  0.40830749, -0.48013624,
    #          0.23249378,  0.24773589],
    #         [0.40993351, 0.72389037, 0.41133479, 0.44710974, 0.74765731,
    #         1.00230691, 0.99664372],
    #         [0.60680285, 0.41172636, 1.36764488, 1.16747916, 0.84448963,
    #         1.88699425, 2.04467738],
    #         [-0.11520159,  0.19955331,  0.64088602,  0.52890755,  0.46021221,
    #          0.59590349,  0.92959655]])
    # An alternative dataset for orth - iso90 contrast
    data = np.array([[ 0.06324374,  0.33905942, -0.43247313,  0.67531022,  1.67253675,
                      3.43659727, -0.36438486],
                     [0.67372082, 0.22916906, 0.61217881, 0.29059113, 0.59437771,
                      1.04501978, 1.28423834],
                     [-0.14524032, -0.01270742, -0.17679373, -0.07686081, -0.14975905,
                      0.13545734,  0.0864793 ],
                     [0.36571019, 0.13116327, 0.4206699 , 0.24347861, 0.64896101,
                      1.48480472, 0.82874181],
                     [ 0.10693128,  0.44012572, -0.22512313,  0.1839792 ,  0.33380663,
                      0.25254567,  0.42639646],
                     [-0.27391247, -0.15772023, -0.28516969, -0.12392212, -0.19845545,
                      0.14413644, -0.27828983],
                     [0.21368122, 0.14997036, 0.33532337, 0.86188467, 0.24441264,
                      1.67804964, 1.23194647],
                     [-0.11252794,  0.17174866,  0.29878078,  0.20567272,  0.15575764,
                      0.42801755,  0.32848109],
                     [0.04621034, 0.32157638, 0.50847193, 0.38039415, 0.72899358,
                      0.94849045, 0.80232748],
                     [ 0.0311975 , -0.4063855 ,  0.15154425,  0.25325635, -0.08940586,
                      -0.04988168,  0.14763531],
                     [-0.50154815, -0.26671061, -0.1373223 , -0.16001749, -0.36159233,
                      0.20420536,  0.29451772],
                     [-0.20470667,  0.5078227 , -0.41187953,  0.23625204,  0.58080691,
                      0.53387643,  0.22987464]])
    
    # Set variables
    axis = 0
    null_stat = 0
    n_permutations = 5000
    test_type = 'two-sided'
    
    # Compute sample statistic
    actual_stat = np.mean(data,axis=0)
    
    # Determine the maximum number of unique permutations
    max_permutations = 2 ** data.shape[axis]
    if n_permutations > max_permutations:
        n_permutations = max_permutations
    
    # Initialize a set to store unique sign flip patterns and an array to store the permutation test statistics
    unique_sign_patterns = set()
    perm_stats = np.zeros((n_permutations,) + actual_stat.shape)
    
    # Generate unique permutations by randomly flipping the signs of the observations
    i = 0
    while i < n_permutations:
        sign_flips = tuple(np.random.choice([-1, 1], size=data.shape[axis], replace=True))
        if sign_flips not in unique_sign_patterns:
            unique_sign_patterns.add(sign_flips)
            expanded_sign_flips = np.expand_dims(np.array(sign_flips), axis=tuple(range(1, data.ndim)))
            permuted_data = data * expanded_sign_flips
            perm_stats[i] = test_stat(permuted_data, axis=axis)
            i += 1
    
    # Compute p-values based on the null distribution
    if test_type == 'two-sided':
        p_values = np.mean(np.abs(perm_stats - null_stat) >= np.abs(actual_stat - null_stat), axis=0)
    elif test_type == 'one-sided-positive':
        p_values = np.mean(perm_stats >= actual_stat, axis=0)
    elif test_type == 'one-sided-negative':
        p_values = np.mean(perm_stats <= actual_stat, axis=0)
    else:
        raise ValueError("Invalid test_type. Choose from 'two-sided', 'one-sided-positive', or 'one-sided-negative'.")
        
    # Also do a parametric t-test
    p_values_ttest = stats.ttest_1samp(data,0,axis=0).pvalue
    
    # Plot
    fig, axes = plt.subplots(7,1,figsize=(4,10))
    for i in reversed(range(np.shape(data)[1])):
        ax = axes[np.shape(data)[1]-i-1]
        ax.hist(data[:,i])
        ax.plot((np.mean(data[:,i]),np.mean(data[:,i])),[0,5],label='mean')
        ax.legend()
        if i == 0:
            ax.set_xlabel('beta')
        if i == 4:
            ax.set_ylabel("N subjects")
        ax.set_title(f'Example Data N = 11: depth = {i}')
    plt.tight_layout()
    
    fig, axes = plt.subplots(7,1,figsize=(4,10))
    for i in reversed(range(np.shape(data)[1])):
        ax = axes[np.shape(data)[1]-i-1]
        ax.hist(permuted_data[:,i])
        ax.plot((np.mean(permuted_data[:,i]),np.mean(permuted_data[:,i])),[0,5],label='mean')
        ax.legend()
        if i == 0:
            ax.set_xlabel('beta')
        if i == 4:
            ax.set_ylabel("N subjects")
        ax.set_title(f'Example Permuted Data N = 11: depth = {i}')
    plt.tight_layout()
    
    fig, axes = plt.subplots(7,1,figsize=(4,10))
    for i in reversed(range(np.shape(data)[1])):
        ax = axes[np.shape(data)[1]-i-1]
        ax.hist(perm_stats[:,i],density=True)
        ax.plot((np.mean(data[:,i]),np.mean(data[:,i])),[0,1.2],label='sample mean')
        ax.legend()
        if i == 0:
            ax.set_xlabel('mean beta')
        if i == 4:
            ax.set_ylabel("Probability Density")
        ax.set_title(f'Null Perm Distribution: p={p_values[i]:.3f}, p_ttest = {p_values_ttest[i]:.3f}, depth = {i}')
    plt.tight_layout()
    