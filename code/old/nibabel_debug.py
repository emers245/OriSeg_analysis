#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:01:53 2023

@author: joee
"""

import numpy as np
import nibabel, os

roiFile = '/home/scat-raid3/data/oriSeg/pnr756_20220627/depth/LAYNII/V1_rh_rad10_radius_p6.nii'
dataset = '/home/scat-raid3/data/oriSeg/pnr756_20220627/'
roi = nibabel.load(roiFile).get_data()
nibabel.save(nibabel.Nifti1Image(roi.astype(np.int16),affine=np.eye(4)),os.path.join(dataset,'rois','test_roi.nii'))
cmd = '3drefit -space ORIG %s' %(os.path.join(dataset,'rois','test_roi.nii'))
os.system(cmd)
cmd = '3drefit -orient RSP %s' %(os.path.join(dataset,'rois','test_roi.nii'))
os.system(cmd)  