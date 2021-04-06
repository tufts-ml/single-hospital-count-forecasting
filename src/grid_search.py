'''
grid_search.py
--------------
Runs grid search for single-site GAR and/or GGP.
User can specify -a to run grid search for GAR only, or -g for GGP only.
Defaults to using both models and producing side-by-side plots of performance.
Plots heldout log likelihood vs window size for GAR, and heldout log likelihood
vs time-scale prior for GGP.
'''

import argparse
import arg_types
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from gar_grid_search import gar_grid_search
from ggp_grid_search import ggp_grid_search

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('input_csv_file', type=arg_types.csv_file, help='name of input CSV file')
	parser.add_argument('-c', '--target_col_name',
		                default='hospitalized_total_covid_patients_suspected_and_confirmed_including_icu',
	                    help='column of CSV file with counts to make predictions on, \
	                    	  default \'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu\'')

	parser.add_argument('-a', '--autoregressive', action='store_true', help='only use autoregressive model')
	parser.add_argument('-g', '--gaussian', action='store_true', help='only use gaussian process')

	parser.add_argument('-m', '--gar_model_file', type=arg_types.json_file, default='gar_model.json',
						help='name of JSON file to write GAR model parameters to, default \'gar_model.json\'')
	parser.add_argument('-o', '--ggp_model_file', type=arg_types.json_file, default='ggp_model.json',
	                    help='name of JSON file to write GGP model parameters to, default \'ggp_model.json\'')

	parser.add_argument('-p', '--performance_plot_file', type=arg_types.pdf_file, default='performance.pdf',
						help='name of PDF file to save plot as, default \'performance.pdf\'')
	parser.add_argument('-f', '--forecast_plot_file', type=arg_types.pdf_file, default='heldout_forecasts.pdf',
						help='name of PDF file to save plot as, default \'heldout_forecasts.pdf\'')
	
	args = parser.parse_args()

	print(f'Input file: {args.input_csv_file}\n')

	train_df = pd.read_csv(args.input_csv_file)
	counts = train_df[args.target_col_name].astype(float)

	dates = train_df['date'].values
	end = dates[-1]

	np.random.seed(42)

	if args.autoregressive:
		fig1,perf_ax = plt.subplots(figsize=(8,6))
		fig2,forecast_ax = plt.subplots(figsize=(8,6))
		gar_grid_search(counts, args.gar_model_file, perf_ax, forecast_ax, end)

	elif args.gaussian:
		fig1,perf_ax = plt.subplots(figsize=(8,6))
		fig2,forecast_ax = plt.subplots(figsize=(8,6))
		ggp_grid_search(counts, args.ggp_model_file, perf_ax, forecast_ax, end)

	else:
		fig1,perf_ax = plt.subplots(ncols=2, sharey=True, figsize=(16,6))
		fig2,forecast_ax = plt.subplots(ncols=2, sharey=True, figsize=(16,6))
		gar_grid_search(counts, args.gar_model_file, perf_ax[0], forecast_ax[0], end)
		ggp_grid_search(counts, args.ggp_model_file, perf_ax[1], forecast_ax[1], end)

	fig1.savefig(args.performance_plot_file)
	fig2.savefig(args.forecast_plot_file)

