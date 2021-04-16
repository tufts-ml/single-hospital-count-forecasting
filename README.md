Welcome! 

This repository hosts code and datasets related to forecasting counts of resource utilization at a single hospital site, given a past time-series history of counts at that site. This code might be a start to help answering questions like: "How many people will be admitted on each day for the next two weeks? How many beds will be in use?" This is a collaborative research project between researchers in the Department of Computer Science at Tufts University and at Tufts Medical Center.

See our manuscript:

Alexandra Hope Lee, Panagiotis Lymperopoulos, Joshua T. Cohen, John B. Wong, and Michael C. Hughes. <i> Forecasting COVID-19 counts at a single hospital: A Hierarchical Bayesian approach. </i> In ICLR 2021 Workshop on Machine Learning for Preventing and Combating Pandemics, 2021. PDF URL: <a href="https://www.michaelchughes.com/papers/LeeEtAl_ICLRWorkshopMLPreventingCombatingPandemics_2021.pdf">https://www.michaelchughes.com/papers/LeeEtAl_ICLRWorkshopMLPreventingCombatingPandemics_2021.pdf</a>

Jump to: [Project Goals](#project-goals) &nbsp; [Limitations](#limitations) &nbsp; [Quickstart Guide](#quickstart-guide) &nbsp; [Repository Contents](#repository-contents) &nbsp; [Installation Guide](#installation) &nbsp; [Datasets Guide](#datasets) &nbsp; [How to Run Experiments](#how-to-run-experiments) 

Project Goals
=============

We consider the problem of forecasting the daily number of hospitalized COVID-19 patients at a single hospital site, in order to help administrators with logistics and planning.

We develop several candidate hierarchical Bayesian models which can:

* capture  the  count  nature  of  data  via either the generalized  Poisson  likelihood (recommended) or the standard Poisson likelihood
* model  time-series  dependencies  via  two kinds of latent process: autoregressive  and  Gaussian  process
* share statistical strength across related sites

In our workshop paper we demonstrated our approach on several public datasets:

* 8 hospitals in Massachusetts, U.S.A.
* 10 hospitals in the United Kingdom

Further prospective evaluation in our manuscript compares our approach favorably to baselines currently used by stakeholders at 3 related hospitals to forecast 2-week-ahead demand by rescaling state-level forecasts.

Limitations
-----------

There are some serious limitations to our approach, which are thoroughly reviewed in our manuscript. Assuming the future is like the past is always a frail assumption, and the data we assume is available to do forecasting is quite limited. Nevertheless, we believe our probabilistic models can help communicate uncertainty and be a starting point for helping make challenging decisions from limited data.


Quickstart Guide
================

To create a set of forecasts for a single site:

1. Checkout this repository on your local machine
2. Install and activate conda environment ([see Installation Guide](#installation))
3. Create a subdirectory called `gar_samples`
4. To produce a forecast for Tufts Medical Center given our available data from summer 2020, you can do:

```
python run_simple_forecast.py -a ../datasets/mass_dot_gov/tufts_medical_center_2020-04-29_to_2020-07-06.csv
```

#### Expected output from running forecasts for Tufts Medical Center:

![Example forecast plot for Tufts Medical Center](example_forecast_tmc.pdf)

Repository Contents
===================

Files and directories in this repository:

- `notebooks` - Jupyter notebooks with detailed specifications of the models and how they translate to code.
- `datasets/` - CSV files with data used for experiments.
- `src/`
  - `arg_types.py` - Checks that filenames specified by command-line arguments have proper suffixes.
  - `gar_forecast.py` - Makes future forecasts using single-site GAR model.
  - `gar_grid_search.py` - Performs grid search for single-site GAR model.
  - `GenPoisson.py` - Defines a Generalized Poisson distribution as a PyMC3 custom Discrete distribution.
  - `GenPoissonAutoregression.py` - Defines a generalized autoregressive (GAR) model with Generalized Poisson likelihood.
  - `GenPoissonGaussianProcess.py` - Defines a generalized Gaussian Process (GGP) model with Generalized Poisson likelihood.
  - `ggp_forecast.py` - Makes future forecasts using single-site GGP model.
  - `ggp_grid_search.py` - Performs grid search for GGP model.
  - `grid_search.py` - Launches grid searches for single-site models.
  - `multi_site_gar.py` - Trains, evaluates, and makes forecasts for multi-site GAR model.
  - `plot_forecasts.py` - Plots summary statistics of forecasts against true observed counts.
  - `poisson_vs_genpoisson.py` - Compares standard and generalized Poisson likelihoods on our model.
  - `run_simple_forecast.py` - Launches forecasting for single-site models.


Installation
=============

### Requirements

* Anaconda 4.8 or higher
* Python 3.6+

Follow the two steps below to install *everything* on your local system.

These have been successfully tested on TODO LIST PLATFORMS as of 2021-04-06.

### Step 1) Install `conda` to manage your environment and packages

Links for installation of "minimal" version of conda:

https://docs.conda.io/en/latest/miniconda.html

Make sure this will edit your PATH (should be automatic on OS X and Linux).

### Step 2) Using conda, create the project specific environment (includes all python packages needed)

We use the included YAML specification file: [`site_level_forecaster.yml`](site_level_forecaster.yml)

To install, just open any terminal, then do:

```
$ conda env create -f site_level_forecaster.yml
```


How We Ran Our Experiments
==========================

Required Libraries
---------
- pymc3: https://docs.pymc.io/
- argparse, numpy, matplotlib, pandas, json, datetime, theano, scipy, os

Experiment #1: Standard vs Generalized Poisson
-------------
On each dataset, trains and scores GAR model with W=1 first using Standard Poisson likelihood, then using Generalized Poisson likelihood.

Reads all files in the directory specified on line 16 of the script. To change the directory name and target column name, modify lines 16, 18, and 28.

**Command**

`python poisson_vs_genpoisson.py`

**Dependencies**

- `GenPoisson.py`
- `GenPoissonAutoregression.py`

**Output**

- Heldout log likelihood for each model to standard output

Experiment #2: Single-Site GGP vs GAR
-------------
Divides sequence of counts into training, validation, and test windows. Runs a grid search over a predefined set of hyperparameters for each model, evaluating on validation set. Takes the best parameters and trains the training and validation set together, then makes forecasts on the test window.

**Command**

`python grid_search.py <input_csv_file>`
- `<input_csv_file>` must have a column `date` with dates in ISO format, and a target column with integer counts.
- Use the flag `-a` to only run the GAR, or the flag `-g` to only run the GGP. Otherwise defaults to running both models and producing side-by-side plots.

**Optional arguments and their defaults**

    -c, --target_col_name       'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu'
    -m, --gar_model_file        'gar_model.json'
    -o, --ggp_model_file        'ggp_model.json'
    -p, --performance_plot_file 'performance.pdf'
    -f, --forecast_plot_file    'heldout_forecasts.pdf'

**Dependencies**

- `gar_grid_search.py`
- `ggp_grid_search.py`
- `GenPoissonAutoregression.py`
- `GenPoissonGaussianProcess.py`
- `GenPoisson.py`
- `arg_types.py`
- `plot_forecasts.py`

**Output**

- Plot of heldout log likelihood for each window size (GAR)
- Plot of heldout log likelihood for each timescale prior mean (GGP)
- JSON file for each model with best model parameters found
- Plot of summary statistics of forecasts against true observed counts

Experiment #3: Multi-Site GAR
-------------
Trains and scores multi-site GAR model and makes forecasts on test window.

Reads all files in directory specified on line 20 of the script. To change the directory name and target column name, modify lines 20, 23, and 33. CSV files must have a column named `date` with dates in ISO format, and a target column with integer counts. Saves traceplot and forecast plots in directory specified on lines 80 and 125 of the script.

**Command**

`python multi_site_gar.py`

**Dependencies**

- `GenPoisson.py`
- `plot_forecasts.py`

**Output**

- Heldout log likelihood for each dataset to standard output
- Traceplot of MCMC chains for key variables
- Forecast plot for each dataset

Single-Site Future Forecasting
------------------------------
After running Experiment #2, use resulting JSON files with best model parameters to make future forecasts.

**Command**

`python run_simple_forecast.py <input_csv_file>`
- `<input_csv_file>` must have a column `date` with dates in ISO format, and a target column with integer counts.
- Use the flag `-a` to only run the GAR, or the flag `-g` to only run the GGP. Otherwise defaults to running both models and producing side-by-side plots.

**Optional arguments and their defaults**

    -c, --target_col_name           'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu'
    -m, --gar_model_file            'gar_model.json'
    -f, --ggp_model_file            'ggp_model.json'
    -o, --gar_csv_file_pattern      'gar_samples/output-*.csv'
    -u, --ggp_csv_file_pattern      'ggp_samples/output-*.csv'
    -s, --day_forecasts_start       day after last day of data
    -d, --n_days_ahead              14
    -p, --plot_file                 'forecasts.pdf'

**Dependencies**

- `gar_forecast.py`
- `ggp_forecast.py`
- `GenPoissonAutoregression.py`
- `GenPoissonGaussianProcess.py`
- `GenPoisson.py`
- `arg_types.py`
- `plot_forecasts.py`

**Output**

- Plot of summary statistics of forecasts
- CSV file for each sampled set of forecasts
