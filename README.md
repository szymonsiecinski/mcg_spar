# mcg_spar
SPAR analysis on mechanocardiograms

This project focuses on the analysis of mechanocardiograms (cardiac mechanical signals) using the Symmetric Projection Attractor Reconstruction (SPAR), mainly for the assessment of quality.

## How to run

1. Run the script `data_importer.py` and optionally set the sampling frequency N (in Hz) with `--fs N`, target directory with `--target signal_files`, and source directory with `--data_path`
2. Run the script `spars.py`
3. Run `spars_classifier.py` and set the type of signals with `--signal ecg|scg|gcg`, labels file with `--label file`, and optionally `--stats` to enable saving descriptive statistics
and `--data_path` to choose the source files.

## Background of SPAR
TODO
