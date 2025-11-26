#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:41:21 2023

@author: joe

gPPI Simulation: A simulaition to try to wrap my head around gPPI
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

nROI1 = 10
nROI2 = 10
np.random.seed(73)
t = np.linspace(0,100,1000)
f1 = 0.25
phi1 = 0
signal1 = np.sin(2*np.pi*f1*t+phi1) #condition 1
eta1 = 1
noise1 = eta1*np.random.randn(nROI1,len(t))
f2 = 0.25
phi2 = np.pi
signal2 = np.sin(2*np.pi*f2*t+phi2)+0.3 #condition 2
eta2 = 1
noise2 = eta2*np.random.randn(nROI2,len(t))
crossSignal = np.sin(2*np.pi*0.1*t)
interSignal = 0.1*scipy.signal.square(2*np.pi*f1*t) #this modulates the amount of condition dependent signal each ROI receives
y1 = signal1+crossSignal+noise1
y2 = signal2+crossSignal+interSignal*y1+noise2 #I've added in a condition depended connection from y1 -> y2
y1_avg = np.mean(y1,0).reshape([1,len(t)])
y2_avg = np.mean(y2,0).reshape([1,len(t)])

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(t,y1.T)
ax1.plot(t,y1_avg.T,'--')
ax1.set_title("ROI 1")
ax2.plot(t,y2.T)
ax2.plot(t,y2_avg.T,'--')
ax2.set_title("ROI 2")

# GLM 1
X1 = np.vstack([signal1.reshape([1,len(t)]),signal2.reshape([1,len(t)]),y1_avg,y1_avg*interSignal]).T
beta1 = np.dot(np.dot(np.linalg.inv(np.dot(X1.T,X1)),X1.T),y2.T)

# GLM 2
X2 = np.vstack([signal1.reshape([1,len(t)]),signal2.reshape([1,len(t)]),y2_avg,y2_avg*interSignal]).T
beta2 = np.dot(np.dot(np.linalg.inv(np.dot(X2.T,X2)),X2.T),y1.T)

#Reconstrcuttions
y2_recon = np.dot(X1,beta1)
y1_recon = np.dot(X2,beta2)

fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(t,y1_recon)
ax1.set_title("ROI 1: reconstruction")
ax1.set_ylim([-5,5])
ax2.plot(t,y2_recon)
ax2.set_title("ROI 2: reconstruction")
ax2.set_ylim([-5,5])

# The first two betas are our signal betas, the third is our baseline connectivity, and the fourth is our interaction beta
# Let's do some standard ROI analyses
beta1_avg = np.mean(beta1,1)
beta1_std = np.std(beta1,1)
beta2_avg = np.mean(beta2,1)
beta2_std = np.std(beta2,1)

fig, (ax1, ax2) = plt.subplots(2)
ax1.errorbar(np.arange(0,4),beta1_avg,beta1_std,linestyle='',marker='o')
ax1.set_xticks(np.arange(0,4))
ax1.set_xticklabels(['signal 1', 'signal 2', 'y2', 'inter y2'])
ax1.set_title("ROI 2 -> ROI 1")
ax2.errorbar(np.arange(0,4),beta2_avg,beta2_std,linestyle='',marker='o')
ax2.set_xticks(np.arange(0,4))
ax2.set_xticklabels(['signal 1', 'signal 2', 'y1', 'inter y1'])
ax2.set_title("ROI 1 -> ROI 2")
plt.tight_layout()