'''
multi_site_gar.py
-----------------
'''

import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import theano.tensor as tt
import os
import scipy
from datetime import date
from datetime import timedelta

from GenPoisson import GenPoisson
from plot_forecasts import plot_forecasts

''' PREP DATA '''
input_files = os.listdir('../mass_dot_gov_datasets')
input_files.sort()
H = len(input_files)
col_name = 'hospitalized_total_covid_patients_suspected_and_confirmed_including_icu'

F = 14
T = []
y_tr = []
y_va = []
y_te = []
dates = []

for filename in input_files:
    df = pd.read_csv(f'../mass_dot_gov_datasets/{filename}')
    counts = df[col_name].astype(float)
    dates.append(df['date'].values)
    y_te.append(counts[-F:])
    y_va.append(counts[-2*F:-F])
    y_tr.append(counts[:-2*F])
    T.append(len(counts) - 2*F)

''' TRAIN ON TRAIN+VALID '''
W = 1

with pm.Model() as model:
    bias = pm.Normal('beta[0]', mu=0, sigma=0.1)
    beta_recent = pm.Normal('beta[1]', mu=1, sigma=0.1)
    rho = [bias, beta_recent]
    for i in range(2, W+1):
        beta = pm.Normal(f'beta[{i}]', mu=0, sigma=0.1)
        rho.append(beta)
    tau = pm.HalfNormal('tau', sigma=0.1)

    f = []
    y_past = []
    y_past_logp = []
    lam = []

    for i in range(H):
        f.append(pm.AR(f'f[{i}]', rho, sigma=tau, constant=True, shape=T[i]+2*F))
        lam.append(pm.TruncatedNormal(f'lam[{i}]', mu=0, sigma=0.3, lower=-1, upper=1))
        y_past.append(GenPoisson(f'y_past[{i}]', theta=tt.exp(f[i][:-F]), lam=lam[i],
                                 observed=np.concatenate((y_tr[i], y_va[i]))))
        y_past_logp.append(pm.Deterministic(f'y_past_logp[{i}]', y_past[i].logpt))

    trace = pm.sample(5000, tune=2000, chains=2, cores=1, target_accept=0.99,
                      init='adapt_diag', max_treedepth=15, random_seed=42)

summary = pm.summary(trace)['mean'].to_dict()
for i in range(W+1):
    print(f'beta[{i}]', summary[f'beta[{i}]'])
print('tau', summary['tau'])
for i in range(H):
    print(f'lambda[{i}]', summary[f'lam[{i}]'])
print()

var_names = ['tau']
for i in range(H):
    var_names.append(f'lam[{i}]')
pm.traceplot(trace, var_names=var_names)
plt.savefig(f'multi_site_gar/traceplot.pdf')
plt.close()

print(f'Training Scores on Train+Valid')
print('------------------------------')
for i in range(H):
    print(input_files[i])
    print(np.log(np.mean(np.exp(trace.get_values(f'y_past_logp[{i}]', chains=0)))) / (T[i]+F))
    print(np.log(np.mean(np.exp(trace.get_values(f'y_past_logp[{i}]', chains=1)))) / (T[i]+F))
    print()

''' HELDOUT SCORING ON TEST SET'''
with model:
    y_future = []
    y_logp = []
    for i in range(H):
        y_future.append(GenPoisson(f'y_future[{i}]', theta=tt.exp(f[i][-F:]), lam=lam[i], observed=y_te[i]))
        y_logp.append(pm.Deterministic(f'y_logp[{i}]', y_future[i].logpt))
    logp_list = pm.sample_posterior_predictive(trace, vars=y_logp, keep_size=True)

print(f'Heldout Scores on Test Set')
print('--------------------------')
for i in range(H):
    print(input_files[i])
    logp_samples = logp_list[f'y_logp[{i}]'][0]
    scores = np.zeros(10)
    for j in range(10):
        scores[j] = np.log(np.mean(np.exp(logp_samples[500*j : 500*j+500]))) / F
    print(f'Chain 1: {np.mean(scores)} ± {scipy.stats.sem(scores)}')
    logp_samples = logp_list[f'y_logp[{i}]'][1]
    scores = np.zeros(10)
    for j in range(10):
        scores[j] = np.log(np.mean(np.exp(logp_samples[500*j : 500*j+500]))) / F
    print(f'Chain 2: {np.mean(scores)} ± {scipy.stats.sem(scores)}')
    print()

''' HELDOUT FORECASTS ON TEST SET'''
with model:
    for i in range(H):
        y_pred = GenPoisson(f'y_pred[{i}]', theta=tt.exp(f[i][-F:]), lam=lam[i], shape=F, testval=1)
        samples = pm.sample_posterior_predictive(trace, vars=[y_pred], keep_size=True, random_seed=42)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_title(f'Site-Specific Lambda Heldout Forecasts W={W}')
        start = date.fromisoformat(dates[i][-1]) - timedelta(F-1)
        plot_forecasts(samples[f'y_pred[{i}]'][0], start, ax, y_va[i], y_te[i], future=False)
        fig.savefig(f'multi_site_gar/{input_files[i][:-4]}.pdf')
        plt.close()




