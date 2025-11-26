# Orientation Classiciation Stats
# This code run statistics on the orientation classiciation analysis
# Some terminology:
#     bands = radius bands
#     top = top voxels sorted according to statistic (F, T, p)
#     random = randomly selected voxels
#     randomTop = randomly selected voxels above some statistical criterion (F, T, p)
# Created 09/12/2025
# Author: Joe Emerson

###########################################################################
# Install Packages (run once only)
###########################################################################
#install.packages("languageserver")
#install.packages("data.table")

###########################################################################
# Configuration Variables
###########################################################################
# Set the voxel selection method: "top" or "randomTop"
voxel_method <- "randomTop"

# Set number of voxels and bands for different analyses
localizer_6band_nvoxels <- 10
localizer_3band_nvoxels <- 30
task_6band_nvoxels <- 10  
task_3band_nvoxels <- 30

###########################################################################
# Localizer 6 Radius Bands, 10 RandomTop Voxels
###########################################################################
# Loading and reporting data
filename1 <- paste0('orientation_classification/accuracy_localizer_6bands_', 
                   localizer_6band_nvoxels, '_', voxel_method, '_voxels.csv')
print(paste("Loading file:", filename1))
df1 <- read.csv(filename1)
data1 <- df1[,c('subj', 'roi', 'depth', 'radius_band', 'cond', 'accuracy')]

library(data.table)
# pick out ROIs from just 1 hemi
rh_data <- data1[df1$roi %like% 'rh' ,]
lh_data <- data1[df1$roi %like% 'lh' ,]

# Check the data
print("=== Localizer 6 Radius Bands, 10 RandomTop Voxels ===")
print("=== DATA SUMMARY ===")
print(paste("Total rows in original data:", nrow(data)))
print(paste("Number of rows in lh_data:", nrow(lh_data)))
print(paste("Number of 'surround' condition rows:", nrow(lh_data[lh_data$cond == 'surround',])))
print(paste("Number of 'center' condition rows:", nrow(lh_data[lh_data$cond == 'center',])))

# Check N at different levels
sur_lh_data <- lh_data[lh_data$cond == 'surround',]
sur_rh_data <- rh_data[rh_data$cond == 'surround',]
print("=== SAMPLE SIZE (N) BREAKDOWN: SUR ===")
print(paste("LH Total observations (N):", nrow(sur_lh_data)))
print(paste("Number of unique subjects:", length(unique(sur_lh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(sur_lh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(sur_lh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(sur_lh_data$radius_band))))
print(paste("RH Total observations (N):", nrow(sur_rh_data)))
print(paste("Number of unique subjects:", length(unique(sur_rh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(sur_rh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(sur_rh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(sur_rh_data$radius_band))))
ctr_lh_data <- lh_data[lh_data$cond == 'center',]
ctr_rh_data <- rh_data[rh_data$cond == 'center',]
print("=== SAMPLE SIZE (N) BREAKDOWN: CTR ===")
print(paste("LH Total observations (N):", nrow(ctr_lh_data)))
print(paste("Number of unique subjects:", length(unique(ctr_lh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(ctr_lh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(ctr_lh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(ctr_lh_data$radius_band))))
print(paste("RH Total observations (N):", nrow(ctr_rh_data)))
print(paste("Number of unique subjects:", length(unique(ctr_rh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(ctr_rh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(ctr_rh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(ctr_rh_data$radius_band))))

# Show the structure of your design
print("=== DESIGN STRUCTURE ===")
print("Subjects per condition:")
print(table(sur_lh_data$subj))
print("Observations per depth level:")
print(table(sur_lh_data$depth))
print("Observations per radius band:")
print(table(sur_lh_data$radius_band))
print("Cross-tabulation of depth x radius_band:")
print(table(sur_lh_data$depth, sur_lh_data$radius_band))

print("First few rows of filtered data:")
print(head(sur_lh_data))
print(head(ctr_lh_data))

# Run ANOVA
print("=== ANOVA RESULTS: SUR ===")
print("LH")
aov_sur_lh <- aov(data=lh_data[lh_data$cond == 'surround',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_sur_lh <- summary(aov_sur_lh)
print(summary_sur_lh)

print("RH")
aov_sur_rh <- aov(data=rh_data[rh_data$cond == 'surround',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_sur_rh <- summary(aov_sur_rh)
print(summary_sur_rh)

print("=== ANOVA RESULTS: CTR ===")
print("LH")
aov_ctr_lh <- aov(data=lh_data[lh_data$cond == 'center',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_ctr_lh <- summary(aov_ctr_lh)
print(summary_ctr_lh)

print("RH")
aov_ctr_rh <- aov(data=rh_data[rh_data$cond == 'center',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_ctr_rh <- summary(aov_ctr_rh)
print(summary_ctr_rh)

# Save ANOVA summaries to separate files
output_suffix <- paste0('_', localizer_6band_nvoxels, '_', voxel_method, '_voxels')
capture.output(summary_sur_lh, file=paste0("orientation_classification/anova_localizer_6bands", output_suffix, "_surround_lh.txt"))
capture.output(summary_sur_rh, file=paste0("orientation_classification/anova_localizer_6bands", output_suffix, "_surround_rh.txt"))
capture.output(summary_ctr_lh, file=paste0("orientation_classification/anova_localizer_6bands", output_suffix, "_center_lh.txt"))
capture.output(summary_ctr_rh, file=paste0("orientation_classification/anova_localizer_6bands", output_suffix, "_center_rh.txt"))

###########################################################################
# Task 6 Radius Bands, 10 RandomTop Voxels
###########################################################################
# Loading and reporting data
filename2 <- paste0('orientation_classification/accuracy_task_6bands_', 
                   task_6band_nvoxels, '_', voxel_method, '_voxels.csv')
print(paste("Loading file:", filename2))
df2 <- read.csv(filename2)
data2 <- df2[,c('subj', 'roi', 'depth', 'radius_band', 'cond', 'accuracy')]

library(data.table)
# pick out ROIs from just 1 hemi
rh_data <- data2[df2$roi %like% 'rh' ,]
lh_data <- data2[df2$roi %like% 'lh' ,]

# Check the data
print("=== Task 6 Radius Bands, 10 RandomTop Voxels ===")
print("=== DATA SUMMARY ===")
print(paste("Total rows in original data:", nrow(data2)))
print(paste("Number of rows in lh_data:", nrow(lh_data)))
print(paste("Number of 'sur' condition rows:", nrow(lh_data[lh_data$cond == 'sur',])))
print(paste("Number of 'iso0' condition rows:", nrow(lh_data[lh_data$cond == 'iso0',])))

# Check N at different levels
sur_lh_data <- lh_data[lh_data$cond == 'sur',]
sur_rh_data <- rh_data[rh_data$cond == 'sur',]
print("=== SAMPLE SIZE (N) BREAKDOWN: SUR ===")
print(paste("LH Total observations (N):", nrow(sur_lh_data)))
print(paste("Number of unique subjects:", length(unique(sur_lh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(sur_lh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(sur_lh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(sur_lh_data$radius_band))))
print(paste("RH Total observations (N):", nrow(sur_rh_data)))
print(paste("Number of unique subjects:", length(unique(sur_rh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(sur_rh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(sur_rh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(sur_rh_data$radius_band))))
iso0_lh_data <- lh_data[lh_data$cond == 'iso0',]
iso0_rh_data <- rh_data[rh_data$cond == 'iso0',]
print("=== SAMPLE SIZE (N) BREAKDOWN: ISO0 ===")
print(paste("LH Total observations (N):", nrow(iso0_lh_data)))
print(paste("Number of unique subjects:", length(unique(iso0_lh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(iso0_lh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(iso0_lh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(iso0_lh_data$radius_band))))
print(paste("RH Total observations (N):", nrow(iso0_rh_data)))
print(paste("Number of unique subjects:", length(unique(iso0_rh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(iso0_rh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(iso0_rh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(iso0_rh_data$radius_band))))

# Show the structure of your design
print("=== DESIGN STRUCTURE ===")
print("Subjects per condition:")
print(table(sur_lh_data$subj))
print("Observations per depth level:")
print(table(sur_lh_data$depth))
print("Observations per radius band:")
print(table(sur_lh_data$radius_band))
print("Cross-tabulation of depth x radius_band:")
print(table(sur_lh_data$depth, sur_lh_data$radius_band))

print("First few rows of filtered data:")
print(head(sur_lh_data))
print(head(iso0_lh_data))

# Run ANOVA
print("=== ANOVA RESULTS: SUR ===")
print("LH")
aov_sur_lh <- aov(data=lh_data[lh_data$cond == 'sur',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_sur_lh_task <- summary(aov_sur_lh)
print(summary_sur_lh_task)

print("RH")
aov_sur_rh <- aov(data=rh_data[rh_data$cond == 'sur',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_sur_rh_task <- summary(aov_sur_rh)
print(summary_sur_rh_task)

print("=== ANOVA RESULTS: ISO0 ===")
print("LH")
aov_iso0_lh <- aov(data=lh_data[lh_data$cond == 'iso0',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_iso0_lh <- summary(aov_iso0_lh)
print(summary_iso0_lh)

print("RH")
aov_iso0_rh <- aov(data=rh_data[rh_data$cond == 'iso0',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_iso0_rh <- summary(aov_iso0_rh)
print(summary_iso0_rh)

# Save ANOVA summaries to separate files
output_suffix <- paste0('_6bands_', task_6band_nvoxels, '_', voxel_method, '_voxels')
capture.output(summary_sur_lh_task, file=paste0("orientation_classification/anova_task", output_suffix, "_surround_lh.txt"))
capture.output(summary_sur_rh_task, file=paste0("orientation_classification/anova_task", output_suffix, "_surround_rh.txt"))
capture.output(summary_iso0_lh, file=paste0("orientation_classification/anova_task_iso0", output_suffix, "_lh.txt"))
capture.output(summary_iso0_rh, file=paste0("orientation_classification/anova_task_iso0", output_suffix, "_rh.txt"))

###########################################################################
# Localizer 3 Radius Bands, 30 RandomTop Voxels
###########################################################################
# Loading and reporting data
filename3 <- paste0('orientation_classification/accuracy_localizer_3bands_', 
                   localizer_3band_nvoxels, '_', voxel_method, '_voxels.csv')
print(paste("Loading file:", filename3))
df1 <- read.csv(filename3)
data1 <- df1[,c('subj', 'roi', 'depth', 'radius_band', 'cond', 'accuracy')]

library(data.table)
# pick out ROIs from just 1 hemi
rh_data <- data1[df1$roi %like% 'rh' ,]
lh_data <- data1[df1$roi %like% 'lh' ,]

# Check the data
print("=== Localizer 3 Radius Bands, 30 RandomTop Voxels ===")
print("=== DATA SUMMARY ===")
print(paste("Total rows in original data:", nrow(data)))
print(paste("Number of rows in lh_data:", nrow(lh_data)))
print(paste("Number of 'surround' condition rows:", nrow(lh_data[lh_data$cond == 'surround',])))
print(paste("Number of 'center' condition rows:", nrow(lh_data[lh_data$cond == 'center',])))

# Check N at different levels
sur_lh_data <- lh_data[lh_data$cond == 'surround',]
sur_rh_data <- rh_data[rh_data$cond == 'surround',]
print("=== SAMPLE SIZE (N) BREAKDOWN: SUR ===")
print(paste("LH Total observations (N):", nrow(sur_lh_data)))
print(paste("Number of unique subjects:", length(unique(sur_lh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(sur_lh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(sur_lh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(sur_lh_data$radius_band))))
print(paste("RH Total observations (N):", nrow(sur_rh_data)))
print(paste("Number of unique subjects:", length(unique(sur_rh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(sur_rh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(sur_rh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(sur_rh_data$radius_band))))
ctr_lh_data <- lh_data[lh_data$cond == 'center',]
ctr_rh_data <- rh_data[rh_data$cond == 'center',]
print("=== SAMPLE SIZE (N) BREAKDOWN: CTR ===")
print(paste("LH Total observations (N):", nrow(ctr_lh_data)))
print(paste("Number of unique subjects:", length(unique(ctr_lh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(ctr_lh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(ctr_lh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(ctr_lh_data$radius_band))))
print(paste("RH Total observations (N):", nrow(ctr_rh_data)))
print(paste("Number of unique subjects:", length(unique(ctr_rh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(ctr_rh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(ctr_rh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(ctr_rh_data$radius_band))))

# Show the structure of your design
print("=== DESIGN STRUCTURE ===")
print("Subjects per condition:")
print(table(sur_lh_data$subj))
print("Observations per depth level:")
print(table(sur_lh_data$depth))
print("Observations per radius band:")
print(table(sur_lh_data$radius_band))
print("Cross-tabulation of depth x radius_band:")
print(table(sur_lh_data$depth, sur_lh_data$radius_band))

print("First few rows of filtered data:")
print(head(sur_lh_data))
print(head(ctr_lh_data))

# Run ANOVA
print("=== ANOVA RESULTS: SUR ===")
print("LH")
aov_sur_lh <- aov(data=lh_data[lh_data$cond == 'surround',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_sur_lh <- summary(aov_sur_lh)
print(summary_sur_lh)

print("RH")
aov_sur_rh <- aov(data=rh_data[rh_data$cond == 'surround',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_sur_rh <- summary(aov_sur_rh)
print(summary_sur_rh)

print("=== ANOVA RESULTS: CTR ===")
print("LH")
aov_ctr_lh <- aov(data=lh_data[lh_data$cond == 'center',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_ctr_lh <- summary(aov_ctr_lh)
print(summary_ctr_lh)

print("RH")
aov_ctr_rh <- aov(data=rh_data[rh_data$cond == 'center',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_ctr_rh <- summary(aov_ctr_rh)
print(summary_ctr_rh)

# Save ANOVA summaries to separate files
output_suffix <- paste0('_3bands_', localizer_3band_nvoxels, '_', voxel_method, '_voxels')
capture.output(summary_sur_lh, file=paste0("orientation_classification/anova_localizer", output_suffix, "_surround_lh.txt"))
capture.output(summary_sur_rh, file=paste0("orientation_classification/anova_localizer", output_suffix, "_surround_rh.txt"))
capture.output(summary_ctr_lh, file=paste0("orientation_classification/anova_localizer", output_suffix, "_center_lh.txt"))
capture.output(summary_ctr_rh, file=paste0("orientation_classification/anova_localizer", output_suffix, "_center_rh.txt"))

###########################################################################
# Task 3: Radius Bands, 30 RandomTop Voxels
###########################################################################
# Loading and reporting data
filename4 <- paste0('orientation_classification/accuracy_task_3bands_', 
                   task_3band_nvoxels, '_', voxel_method, '_voxels.csv')
print(paste("Loading file:", filename4))
df2 <- read.csv(filename4)
data2 <- df2[,c('subj', 'roi', 'depth', 'radius_band', 'cond', 'accuracy')]

library(data.table)
# pick out ROIs from just 1 hemi
rh_data <- data2[df2$roi %like% 'rh' ,]
lh_data <- data2[df2$roi %like% 'lh' ,]

# Check the data
print("=== Task 3 Radius Bands, 30 RandomTop Voxels ===")
print("=== DATA SUMMARY ===")
print(paste("Total rows in original data:", nrow(data2)))
print(paste("Number of rows in lh_data:", nrow(lh_data)))
print(paste("Number of 'sur' condition rows:", nrow(lh_data[lh_data$cond == 'sur',])))
print(paste("Number of 'iso0' condition rows:", nrow(lh_data[lh_data$cond == 'iso0',])))

# Check N at different levels
sur_lh_data <- lh_data[lh_data$cond == 'sur',]
sur_rh_data <- rh_data[rh_data$cond == 'sur',]
print("=== SAMPLE SIZE (N) BREAKDOWN: SUR ===")
print(paste("LH Total observations (N):", nrow(sur_lh_data)))
print(paste("Number of unique subjects:", length(unique(sur_lh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(sur_lh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(sur_lh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(sur_lh_data$radius_band))))
print(paste("RH Total observations (N):", nrow(sur_rh_data)))
print(paste("Number of unique subjects:", length(unique(sur_rh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(sur_rh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(sur_rh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(sur_rh_data$radius_band))))
iso0_lh_data <- lh_data[lh_data$cond == 'iso0',]
iso0_rh_data <- rh_data[rh_data$cond == 'iso0',]
print("=== SAMPLE SIZE (N) BREAKDOWN: ISO0 ===")
print(paste("LH Total observations (N):", nrow(iso0_lh_data)))
print(paste("Number of unique subjects:", length(unique(iso0_lh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(iso0_lh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(iso0_lh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(iso0_lh_data$radius_band))))
print(paste("RH Total observations (N):", nrow(iso0_rh_data)))
print(paste("Number of unique subjects:", length(unique(iso0_rh_data$subj))))
print(paste("Number of unique ROIs:", length(unique(iso0_rh_data$roi))))
print(paste("Number of unique depth levels:", length(unique(iso0_rh_data$depth))))
print(paste("Number of unique radius bands:", length(unique(iso0_rh_data$radius_band))))

# Show the structure of your design
print("=== DESIGN STRUCTURE ===")
print("Subjects per condition:")
print(table(sur_lh_data$subj))
print("Observations per depth level:")
print(table(sur_lh_data$depth))
print("Observations per radius band:")
print(table(sur_lh_data$radius_band))
print("Cross-tabulation of depth x radius_band:")
print(table(sur_lh_data$depth, sur_lh_data$radius_band))

print("First few rows of filtered data:")
print(head(sur_lh_data))
print(head(iso0_lh_data))

# Run ANOVA
print("=== ANOVA RESULTS: SUR ===")
print("LH")
aov_sur_lh <- aov(data=lh_data[lh_data$cond == 'sur',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_sur_lh_task <- summary(aov_sur_lh)
print(summary_sur_lh_task)

print("RH")
aov_sur_rh <- aov(data=rh_data[rh_data$cond == 'sur',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_sur_rh_task <- summary(aov_sur_rh)
print(summary_sur_rh_task)

print("=== ANOVA RESULTS: ISO0 ===")
print("LH")
aov_iso0_lh <- aov(data=lh_data[lh_data$cond == 'iso0',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_iso0_lh <- summary(aov_iso0_lh)
print(summary_iso0_lh)

print("RH")
aov_iso0_rh <- aov(data=rh_data[rh_data$cond == 'iso0',],
              accuracy ~ depth + radius_band + radius_band*depth)
summary_iso0_rh <- summary(aov_iso0_rh)
print(summary_iso0_rh)

# Save ANOVA summaries to separate files
output_suffix <- paste0('_3bands_', task_3band_nvoxels, '_', voxel_method, '_voxels')
capture.output(summary_sur_lh_task, file=paste0("orientation_classification/anova_task", output_suffix, "_surround_lh.txt"))
capture.output(summary_sur_rh_task, file=paste0("orientation_classification/anova_task", output_suffix, "_surround_rh.txt"))
capture.output(summary_iso0_lh, file=paste0("orientation_classification/anova_task_iso0", output_suffix, "_lh.txt"))
capture.output(summary_iso0_rh, file=paste0("orientation_classification/anova_task_iso0", output_suffix, "_rh.txt"))
