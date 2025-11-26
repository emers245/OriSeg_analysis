#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:57:00 2023

@author: joe

Error Propagation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# make some measurements
np.random.seed(92)
mean1 = 2
sd1 = 0.5
var1 = sd1*np.random.randn(1000) + mean1
mean2 = 1
sd2 = 0.25
var2 = sd2*np.random.randn(1000) + mean2

# plot distribution of measurements
plt.figure()
plt.subplot(3,1,1)
plt.hist(var1,bins=20,alpha=0.5)
plt.hist(var2,bins=20,alpha=0.5)
plt.legend(["var1","var2"])

# assume each index is an observation and that the difference between the variables on each observation
diff = var1-var2
diff_mean = np.mean(diff)
diff_sd = np.std(diff)

# plot distribution of the differences
plt.subplot(3,1,2)
plt.hist(diff,bins=20,color='g',alpha=0.5)
bin_edges = np.histogram_bin_edges(diff,bins=20)
diff_bincount = np.bincount(np.digitize(diff,bin_edges))
#plt.ylim([0,np.max(diff_bincount)])
plt.vlines([diff_mean],0,np.max(diff_bincount),colors='b')
plt.fill_betweenx([0,np.max(diff_bincount)],diff_mean-diff_sd,diff_mean+diff_sd,color='b',alpha=0.5)
plt.legend(["mean=%.2f" %diff_mean,"st. dev.=%.2f" %diff_sd,"var1-var2"])

# Now do the same thing, but this time use summary statistics to do error prop
diff_mean_ep = mean1-mean2
diff_sd_ep = np.sqrt(sd1**2+sd2**2)

# plot distribution of the differences
plt.subplot(3,1,3)
plt.hist(diff,bins=20,color='g',alpha=0.5)
bin_edges = np.histogram_bin_edges(diff,bins=20)
diff_bincount = np.bincount(np.digitize(diff,bin_edges))
#plt.ylim([0,np.max(diff_bincount)])
plt.vlines([diff_mean_ep],0,np.max(diff_bincount),colors='b')
plt.fill_betweenx([0,np.max(diff_bincount)],diff_mean_ep-diff_sd_ep,diff_mean_ep+diff_sd_ep,color='b',alpha=0.5)
plt.legend(["mean=%.2f" %diff_mean_ep,"st. dev.=%.2f" %diff_sd_ep,"var1-var2"])

# there are differences in the measurements because these distributions are only approximating an idealized normal distribution
# increasing the sample size causes better convergence toward normality, and therefore, the error calculations converge

# How does the variance of the mean act as a function of the underlying variances?
sd1_vec = np.logspace(-1,1,100)
sd2_vec = np.logspace(-1,1,100)
sd_grid = np.meshgrid(sd1_vec,sd2_vec)
sd_mean = np.sqrt(sd_grid[0]**2 + sd_grid[1]**2)

# Create a 3D plot of the surface map
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(sd_grid[0], sd_grid[1], sd_mean)
ax.set_xlabel('$\sigma_1$')
ax.set_ylabel('$\sigma_2$')
ax.set_zlabel('$\sigma_{mean}$')
plt.show()

# Plot for sd1 = sd2
grad = np.gradient(np.sqrt(sd1_vec**2+sd2_vec**2),sd1_vec)
fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax1.plot(sd1_vec, np.sqrt(sd1_vec**2+sd2_vec**2),'b')
ax1.set_xlabel('$\sigma = \sigma_1 = \sigma_2$')
ax1.set_ylabel('$\sigma_{mean}$',color='b')
ax2 = plt.twinx()
ax2.plot(sd1_vec, grad,'orange')
ax2.set_ylabel('$\\frac{d\sigma_{mean}}{d\sigma}$',color='orange')
ax2.set_ylim([1,2])
ax2.ticklabel_format(axis='y',style='plain')
plt.show()