Quickstart Guide
================
To create a set of forecasts for a single site:
1. Download all files in `src` and `mass_dot_gov_datasets`
2. Install and activate conda environment
3. Create a subdirectory called `gar_samples`
4. Run `python run_simple_forecast.py -a <input_csv_file>`, where `<input_csv_file>` is any of the provided csv files.

Output from running forecasts for Tufts Medical Center:

![Example forecast plot](forecasts.pdf)

Files & Directories
===================
- `notebooks` - Jupyter notebooks with detailed specifications of the models and how they translate to code.
- `mass_dot_gov_datasets` - CSV files with data used for experiments.
- `src`
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

How We Ran Our Experiments
==========================
Required Libraries
---------
- pymc3: https://docs.pymc.io/
- argparse, numpy, matplotlib, pandas, json, datetime, theano, scipy, os

Experiment #1: Standard vs Generalized Poisson
-------------
On each dataset, trains and scores GAR model with W=1 first using Standard Poisson likelihood, then using Generalized Poisson likelihood.

Reads all files in the directory specified on line 17 of the script. To change the directory name and target column name, modify lines 17, 19, and 29.

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
