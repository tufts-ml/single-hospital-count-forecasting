'''
run_simple_forecast.py
----------------------
Produces samples of forecasted counts using GAR and/or GGP.
User can specify -a to use GAR only, or -g for GGP only.
Defaults to using both models and producing side-by-side plots of forecasts.
Takes as input CSV file of counts and JSON file that specifies model parameters.
Writes predictions to CSV files, and plots summary statistics of forecasts.
'''

import json
import argparse
import arg_types
from datetime import date
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gar_forecast import gar_forecast
from ggp_forecast import ggp_forecast

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_csv_file', type=arg_types.csv_file,
                        help='name of input CSV file, must have a column \'date\'')
    parser.add_argument('-c', '--target_col_name',
                        default='hospitalized_total_covid_patients_suspected_and_confirmed_including_icu',
                        help='column of CSV file with counts to make predictions on, \
                              default \'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu\'')

    parser.add_argument('-a', '--autoregressive', action='store_true', help='only use autoregressive model')
    parser.add_argument('-g', '--gaussian', action='store_true', help='only use gaussian process')

    parser.add_argument('-m', '--gar_model_file', type=arg_types.json_file, default='gar_model.json',
                        help='JSON file that specifies GAR hyperparameters, default \'gar_model.json\'')
    parser.add_argument('-f', '--ggp_model_file', type=arg_types.json_file, default='ggp_model.json',
                        help='JSON file that specifies GGP hyperparmeters, default \'ggp_model.json\'')

    parser.add_argument('-o', '--gar_csv_file_pattern', type=arg_types.csv_file, default='gar_samples/output-*.csv',
                        help='pathname pattern for output CSV files, default \'gar_samples/output-*.csv\'')
    parser.add_argument('-u', '--ggp_csv_file_pattern', type=arg_types.csv_file, default='ggp_samples/output-*.csv',
                        help='pathname pattern for output CSV files, default \'ggp_samples/output-*.csv\'')

    parser.add_argument('-s', '--day_forecasts_start', type=date.fromisoformat, metavar='YYYY-MM-DD',
                        help='default: day after last day of data')
    parser.add_argument('-d', '--n_days_ahead', default=14, type=int,
                        help='number of days of predictions (including start date), default 14')

    parser.add_argument('-p', '--plot_file', type=arg_types.pdf_file, default='forecasts.pdf',
                        help='name of PDF file to save plot as, default \'forecasts.pdf\'')

    args = parser.parse_args()

    if args.n_days_ahead <= 0:
        raise argparse.ArgumentTypeError('n_days_ahead must be a positive integer.')

    n_predictions = args.n_days_ahead

    ### Read data from CSV file ###
    train_df = pd.read_csv(args.input_csv_file)
    dates = train_df['date'].values
    counts = train_df[args.target_col_name].astype(float)

    ### Set start date of forecasts ###
    next_day = date.fromisoformat(dates[-1]) + timedelta(1) # day after last day of data
    start = args.day_forecasts_start
    if start == None:
        start = next_day # default to next_day
    if not date.fromisoformat(dates[0]) + timedelta(10) <= start <= next_day: # check that start date is within range
        raise argparse.ArgumentTypeError('day_forecasts_start must be at least 10 days after \
                                          earliest data point and at most 1 day after last data point')

    counts = counts[:(start - date.fromisoformat(dates[0])).days] # only train on data up until start

    np.random.seed(42)

    if args.autoregressive:
        with open(args.gar_model_file) as f:
            gar_model_dict = json.load(f)

        fig,ax = plt.subplots(figsize=(8,6))
        gar_forecast(gar_model_dict, counts,
                     n_predictions,
                     args.gar_csv_file_pattern,
                     start, ax)

    elif args.gaussian:
        with open(args.ggp_model_file) as f:
            ggp_model_dict = json.load(f)

        fig,ax = plt.subplots(figsize=(8,6))
        ggp_forecast(ggp_model_dict, counts,
                     n_predictions,
                     args.ggp_csv_file_pattern,
                     start, ax)

    else:
        with open(args.gar_model_file) as f:
            gar_model_dict = json.load(f)
        with open(args.ggp_model_file) as f:
            ggp_model_dict = json.load(f)

        fig,ax = plt.subplots(ncols=2, sharey=True, figsize=(16,6))
        gar_forecast(gar_model_dict, counts,
                     n_predictions,
                     args.gar_csv_file_pattern,
                     start, ax[0])
        ggp_forecast(ggp_model_dict, counts,
                     n_predictions,
                     args.ggp_csv_file_pattern,
                     start, ax[1])

    plt.savefig(args.plot_file)
