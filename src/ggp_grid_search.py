'''
ggp_grid_search.py
------------------
Perform grid search over a set of hyperparameters for a Generalized Gaussian
Process (GGP) fit to the given count data.
Divide sequence of counts into training, validation, and test windows.
During grid search, evaluate on validation set.
Write best model parameters to JSON file with the given filename.
Plot best heldout log likelihood for timescale prior on `perf_ax`.
Use best parameters to train on training and validation together and
evaluate on test set. Make forecasts on test window and plot on `forecast_ax`.
'''

import numpy as np
import json
from datetime import date
from datetime import timedelta

from GenPoissonGaussianProcess import GenPoissonGaussianProcess
from plot_forecasts import plot_forecasts

def ggp_grid_search(counts, output_model_file, perf_ax, forecast_ax, end):
    F = 14
    y_te = counts[-F:]
    y_va = counts[-2*F:-F]
    y_tr = counts[:-2*F]

    ### Initialize hyperparameter spaces ###
    l_mus = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50] # time-scale
    c_mus = [4] # mean

    score_per_time_scale = list()
    params_per_time_scale = list()

    for l in l_mus:

        score_list = list()

        for c_mu in c_mus:

            ### Fit and score a model with the current parameters ###
            model_dict = {
                'c': [c_mu, 2],
                'a': 2,
                'l': [l, 2],
            }

            model = GenPoissonGaussianProcess(model_dict)
            model.fit(y_tr, F)
            score = model.score(y_va)
            score_list.append(score)

        ### Choose the best model for the current time-scale ###
        best_id = np.argmax(score_list)
        best_score = score_list[best_id]
        best_params = c_mus[best_id]

        print(f'Best prior params for l = {l}: ', end='')
        print(f'mean = {best_params} | ', end='')
        print(f'score = {best_score}\n')

        score_per_time_scale.append(best_score)
        params_per_time_scale.append(best_params)

    ### Choose the best model overall ###
    best_id = np.argmax(score_per_time_scale)
    best_score = score_per_time_scale[best_id]
    best_params = params_per_time_scale[best_id]

    print('Best prior params overall: ', end='')
    print(f'l = {l_mus[best_id]} | ', end='')
    print(f'mean = {best_params} | ', end='')
    print(f'score = {best_score}\n')

    ### Plot best score for each timescale prior assumption ###
    perf_ax.set_title('GGP Performance vs Time-Scale Prior')
    perf_ax.set_xlabel('Time-scale prior mean')
    perf_ax.set_ylabel('Heldout log lik')
    perf_ax.plot(l_mus, score_per_time_scale, 's-')

    ### Write best model parameters to json file ###
    model = dict()
    model['c'] = [best_params, 2]
    model['a'] = 2
    model['l'] = [l_mus[best_id], 2]

    with open(output_model_file, 'w') as f:
        json.dump(model, f, indent=4)

    ### Plot heldout forecasts using best model ###
    best_model = GenPoissonGaussianProcess(model)
    best_model.fit(np.concatenate((y_tr, y_va)), F)
    best_model.score(y_te)
    samples = best_model.forecast()
    forecast_ax.set_title('Single-site GGP Heldout Forecasts')
    start = date.fromisoformat(end) - timedelta(F-1)
    plot_forecasts(samples, start, forecast_ax, y_va, y_te)


