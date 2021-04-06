'''
gar_grid_search.py
------------------
Perform grid search over a set of hyperparameters for a Generalized Autoregressive
Process (GAR) fit to the given count data.
Divide sequence of counts into training, validation, and test windows.
During grid search, evaluate on validation set.
Write best model parameters to JSON file with the given filename.
Plot best heldout log likelihood for each window size W on `perf_ax`.
Use best parameters to train on training and validation together and
evaluate on test set. Make forecasts on test window and plot on `forecast_ax`.
'''

import numpy as np
import json
from datetime import date
from datetime import timedelta

from GenPoissonAutoregression import GenPoissonAutoregression
from plot_forecasts import plot_forecasts

def gar_grid_search(counts, output_model_file, perf_ax, forecast_ax, end):
    F = 14
    y_te = counts[-F:]
    y_va = counts[-2*F:-F]
    y_tr = counts[:-2*F]

    window_sizes = [1, 2, 5, 7, 10, 14]
    prior_sigmas = [(0.1, 0.1)]
    
    score_per_window_size = list()
    params_per_window_size = list()

    for W in window_sizes:

        score_list = list()

        for bias_sigma, beta_sigma in prior_sigmas:

            model_dict = {
                'window_size': W,
                'bias': [0, bias_sigma],
                'beta_recent': [1, beta_sigma],
                'beta': [0, beta_sigma],
            }

            model = GenPoissonAutoregression(model_dict)
            model.fit(y_tr, F)
            score = model.score(y_va)
            score_list.append(score)

        best_id = np.argmax(score_list)
        best_score = score_list[best_id]

        print(f'Best params for W = {W}: sigmas = {prior_sigmas[best_id]} | score = {best_score}\n')
        score_per_window_size.append(best_score)
        params_per_window_size.append(prior_sigmas[best_id])

    best_id = np.argmax(score_per_window_size)
    print('Best hypers overall:')
    print(f'W = {window_sizes[best_id]} | sigmas = {params_per_window_size[best_id]} | score = {score_per_window_size[best_id]}\n')

    perf_ax.plot(window_sizes, score_per_window_size, 's-')
    perf_ax.set_title('GAR Performance vs Window Size')
    perf_ax.set_xlabel('window size')
    perf_ax.set_ylabel('heldout log likelihood')

    model = dict()
    model['window_size'] = window_sizes[best_id]
    model['bias'] = [0, params_per_window_size[best_id][0]]
    model['beta_recent'] = [1, params_per_window_size[best_id][1]]
    model['beta'] = [0, params_per_window_size[best_id][1]]

    with open(output_model_file, 'w') as f:
        json.dump(model, f, indent=4)

    # Plot heldout forecasts using best model
    best_model = GenPoissonAutoregression(model)
    best_model.fit(np.concatenate((y_tr, y_va)), F)
    best_model.score(y_te)
    samples = best_model.forecast()
    forecast_ax.set_title('Single-site GAR Heldout Forecasts')
    start = date.fromisoformat(end) - timedelta(F-1)
    plot_forecasts(samples, start, forecast_ax, y_va, y_te)


