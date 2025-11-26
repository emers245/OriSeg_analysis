# OriSeg_analysis
Code for analyzing fMRI data from the OriSeg project

# Files and Directories
/OriSeg_analysis
    *main analysis code

    > /old
        *old analysis code from before the git repo was established
        
        >/20241105
	    *files that were moved from the main analysis directory to old on 11/05/2024

    > /supplemental
        *code used for additional analyses that are not published
    
    > /roi_data
	*contains CSV files for each subject containing beta weights for each voxel

    > /analyzed_data
	*contains CSV files saved after running oriseg_fullAnalysis.py
    
# Analysis

Main analysis code is contained in:
oriSeg_fullAnalysis.py
    This is the main analysis code. The steps carried out in this code include:
	1) Collecting datasets for both V1 and V2/3 ROIs
	2) Plotting T1w profiles
	3) Plotting full model p-values for each ROI
	4) Visualizing ROIs on surface
	5) Combining ROIs within subjects
	6) Deveining ROIs
	7) Significance thresholding ROIs
	8) Computing and plotting depth profiles
	9) Computing and plotting radial profiles
	10) Computing and plotting averages for gPPI
oriseg_funcs.py
    This holds many of the functions used in other pieces of code for computing
    profiles, running statitics, and plotting.
depth_visualizations.py
    For visualizing ROIs in 3D.
inclusionCriteria.py
    For determining which subjects are included in analyses.
