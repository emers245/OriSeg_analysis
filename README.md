# README
This repository stores analysis code for Emerson, Navarro, & Olman (in press).

## Citation

If you use the code in this repository, please cite:
Emerson, J. H., Navarro, K., & Olman, C. A. Orientation-tuned surround suppression exhibits a unique laminar signature in human primary visual cortex. Proc. Natl. Acad. Sci. U.S.A. (in press)

## Raw Data

The raw fMRI datasets are available on OpenNeuro at: https://openneuro.org/datasets/ds007862

## Set up
*Requires anaconda installation

	1) Clone repository
	2) Set up python environment: `conda env create -f environment.yml`
	3) Activate environment: `conda activate oriseg`

## Quick Start

To run the main analysis, run:

`python3 code/depthAnalysis/oriSeg_V1_analysis.py`

This generates the main subplots and QC analyses.

## Directories
### code/
Main analysis code.
	
#### depthAnalysis/

Contains the primary analysis for the paper. Analysis code is executed in oriSeg_V1_analysis.py.

**stats/** : statistics for main analysis and QC

**analyzed_data/** : contains outputs from oriSeg_V1_analysis.py such as depth and radial profiles.

#### behavior/

Contains a single jupyter notebook running analysis of behavioral data for the supplemental information.

#### orientation_classification

Contains code for running the orientation decoding analysis contained in the supplemental information. This code was run on a remote server and requires server access for use. The outputs have been stored in this directory as well as a copy of the original code.

**accuracy/** : decoding accuracies

**stats/** : statistics for orientation classification

### figs/

All plots used in generating figures, including additional plots used for quality control.

### data/

**behavior/** - trial-wise behavioral data organized by subject and scan organized in CSV format

**roi_data/** - voxel-wise preprocessed BOLD fMRI data from extracted ROIs organized by ROI location and subject
