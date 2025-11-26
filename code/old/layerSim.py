#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:49:33 2023

@author: joe

Toy Depth Analysis: I wrote this code to understand how interpreting laminar
profiles above the effective resolution can influence results.
"""

import numpy as np
import matplotlib.pyplot as plt

def linear_response(v_loc,v_size,profile,d,noise_amp=1,A=1):
    # assume the BOLD response depends linearly on the underlying neural activity with Gaussian noise
    noise = noise_amp*np.random.randn()
    sigma = v_size/(2*np.sqrt(2*np.log(2))) #convert FWHM to st. dev.
    v_span = A*(1/(2*np.pi*sigma))*np.exp(-((d-v_loc)**2)/(2*sigma**2)) #Gaussian envelope
    resp = noise+np.sum(v_span*profile)
    return resp

# Run Simple Simulation
dd = 0.01
d = np.arange(-0.25, 1.25, dd) #depth; 0 = WM, 1 = pial surface
freq = 3*np.pi #freuqency of the depth-dependent signal
profile = np.sin(freq*d)+1 #a sinusoidal underlying profile
profile[d<0] = 0 #assume no signal in WM
profile[d>1] = 0 #assume no signal in CSF

# Compute BOLD Response
Nvox = 1000 #number of voxels
v_locs = np.random.uniform(size=Nvox) #randomly distributed voxel centroids
noise_amp = 10
v_size = 0.6/2.5 #roughly how much I expect a 0.6 mm isotropic V1 voxel to cover in normalized units of cortical depth
responses = np.zeros((Nvox,)) #initialize voxel responses
for v_i, v in enumerate(v_locs):
    responses[v_i] = dd*linear_response(v,v_size,profile,d,noise_amp=noise_amp)
    
# Bin data
Ndepths = 20
d_bins = np.linspace(0,1,Ndepths+1)
d_bin_centers = d_bins[:-1]+np.diff(d_bins)[0]
bin_assignments = np.digitize(v_locs,d_bins)
avg_profile = np.zeros((Ndepths,))
std_profile = np.zeros((Ndepths,))
N_profile = np.zeros((Ndepths,))
for d_i, depth in enumerate(d_bins[:-1]):
    avg_profile[d_i] = np.average(responses[bin_assignments==(d_i+1)])
    std_profile[d_i] = np.std(responses[bin_assignments==(d_i+1)])
    N_profile[d_i] = np.sum(bin_assignments==(d_i+1))

# Plot
fig, ax = plt.subplots(1,4,figsize=(10,4))
ax[0].plot(profile,d) #plot underlying neural signal
ax[0].set_title("Neural Profile")
ax[0].set_xlabel("Neural Response")
ax[0].set_ylabel("Normalized Depth")
ax[0].set_ylim([np.min(d),np.max(d)])
sigma = v_size/(2*np.sqrt(2*np.log(2))) #convert FWHM to st. dev.
v_span = (1/(2*np.pi*sigma))*np.exp(-((d-0.5)**2)/(2*sigma**2)) #voxel envelope
ax[1].plot(v_span,d) #plot underlying neural signal
ax[1].set_title("Voxel Envelope")
ax[1].set_xlabel("Voxel Pooling")
ax[1].set_ylabel("Normalized Depth")
ax[1].set_ylim([np.min(d),np.max(d)])
ax[2].plot(responses,v_locs,'.') #plot the resulting BOLD responses in the voxels as a centroid plot
ax[2].set_title("BOLD Centroid")
ax[2].set_xlabel("BOLD Response")
ax[2].set_ylabel("Normalized Depth")
ax[2].set_ylim([np.min(d),np.max(d)])
ax[3].errorbar(avg_profile,d_bin_centers,xerr=(std_profile/np.sqrt(N_profile))) #plot binned BOLD response
ax[3].set_title("BOLD Binned Profile")
ax[3].set_xlabel("BOLD Response")
ax[3].set_ylabel("Normalized Depth")
ax[3].set_ylim([np.min(d),np.max(d)])
fig.suptitle(f"Noise Amp = {noise_amp: .2f}")
plt.tight_layout()