# V1Locomotion

This repo houses the analysis and figure code for Liska et al., 2022. 

## Getting the data

If you want to reproduce the results and explore the shared-gain model, the processed data are available [here](). All analyses work off of aggregate [supersessions](https://www.biorxiv.org/content/10.1101/2020.08.09.243279v2.abstract) so there is one file for each subject (all mice are combiend into one file).

The processed individual marmoset sessions are availabled [here](). These are used only to create the example session figure and generate the supersession files. For raw data, contact Alex Huk. The import and preprocessing of the raw datafiles is performed in a [separate repo](https://github.com/jcbyts/V1FreeViewingCode).

All mouse data comes from the Allen Institute [brain observatory 1.1](https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels). This repo contains scripts to import and preprocess the data using the [Allen Institute SDK](https://allensdk.readthedocs.io/en/latest/).

## Set your paths
This repo uses Matlab preferences to manage the paths. You need to set the right preference to point to the data directory (this is the folder with files `allenD_all.mat`, `gru_all.mat`, and `brieD_all.mat`; For recreating supersessioned files, you must also have data in subfolders `gratings` and `brain_observatory_1.1`). If you need to construct the Allen Institute data for some reason, run `Code/allen_data_to_matlab.py`. You'll need the Allen institute SDK to run this.

```
setpref('FREEVIEWING', 'HUKLAB_DATASHARE', 'Full/Path/To/Your/Folder')
```

## Get the super sessions
We use [supersessioning](https://www.biorxiv.org/content/10.1101/2020.08.09.243279v2.abstract) to combine across experimental sessions for stable units. The supersessioned files are available preprocessed [here](). To re-build a supersession file for each marmoset and all mouse sessions, run `get_supersession_files.m`. This will spit out a big file for each subject (mice are combined into one file).

## Run the analyses
The analyses are broken into separate `*.m` files that begin with `fig_`. 

These are meant to be run as interactive scripts. Step through the cells to run analyses and generate plots.

### Scripts that require individual session data:
#### fig_example_session.m
This generates the example session plot in Figure 1 (currently 1b).

#### fig_subject_rfs.m
This generates the example RFs in Figure 1 and the contour plots for Marmoset and Mouse (Figure 1 e,f).

#### fig_example_units.m
This builds the raster plot from figure 1d.

### Scripts that only require the supersessions:
#### fig_session_corr.m
This generates the panels from figure 2 and the corresponding statistics.

#### fig_main.m
This runs the analyses for the scatter plots in Figure 3. This will dump MANY figures and text files for all the different conditions.

#### Code/latent_modeling.py
Must run `fig_session_corr.m` first and turn `opts.save = true;` to save out the required files.

### Additional analyses that were not included in the manuscript 
#### fig_regression_analysis.m
This runs a spike count regression analysis that was not included in the mansucript.

#### fig_eyepos_running.m
This figure explores the relationship between eye position, drift, saccades, and running.

This will not run on the mouse data because the "brain_observatory 1.1" dataset does not have consistent eye-tracking data.
