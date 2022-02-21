# V1Locomotion

This repo houses the analysis and figure code for Liska et al., 2022. 


The import and preprocessing of the raw datafiles is performed in a [separate repo](https://github.com/jcbyts/V1FreeViewingCode). 

## Getting the data
To get the data, you need access to the [HuklabTreadmill Google drive folder](https://drive.google.com/drive/folders/1te-Fna8YGaocWpO9rfNSoLtLeuzSNCPq?usp=sharing). Request access and Jack or Jake can grant it.

## Set your paths
This repo uses Matlab preferences to manage the paths. You need to set the right preference to point to the data directory (this is the top folder with subfolders `gratings` and `brain_observatory_1.1`). If you need to construct the Allen Institute data for some reason, run `Code/allen_data_to_matlab.py`. You'll need the Allen institute SDK to run this.

```
setpref('FREEVIEWING', 'HUKLAB_DATASHARE', 'Full/Path/To/Your/Folder')
```

## Get the super sessions
We use [supersessioning](https://www.biorxiv.org/content/10.1101/2020.08.09.243279v2.abstract) to combine across experimental sessions for stable units. To build a supersession file for each marmoset and all mouse sessions, run `get_supersession_files.m`. This will spit out a big file (6-12GB) for each subject (mice are combined into one file).

## Run the analyses
The analyses are broken into separate `*.m` files that begin with `fig_`. 

These are meant to be run as interactive scripts. Step through the cells to run analyses and generate plots.

### fig_example_session.m
This generates the example session plot in Figure 1 (currently 1b).

### fig_subject_rfs.m
This generates the example RFs in Figure 1 and the contour plots for Marmoset and Mouse (Figure 1 e,f)

### fig_example_units.m
This builds the raster plot from figure 1d.

### fig_session_corr.m
This generates the panels from figure 2 and the corresponding statistics.

### fig_main.m
This runs the analyses for the scatter plots in Figure 3. This will dump MANY figures and text files for all the different conditions.

### fig_regression_analysis.m
This runs the spike count regression analysis for figure 4. You can then plot the outcomes to produce the panels from figure 4.

### fig_eyepos_running.m
This figure explores the relationship between eye position, drift, saccades, and running.

This will not run on the mouse data because the "brain_observatory 1.1" dataset does not have consistent eye-tracking data.