'''
GenPoissonGaussianProcess.py
----------------------------
Defines a generalized Gaussian Process model with Generalized Poisson likelihood.
Contains fit, score, and forecast methods.
'''

import pymc3 as pm
import numpy as np
import pandas as pd
import theano.tensor as tt
import scipy
from GenPoisson import GenPoisson
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

class GenPoissonGaussianProcess:

    '''
    init
    ----
    Takes in dictionary that specifies the model parameters.
    Each prior is a Truncated Normal dist, lower bounded at 0.
    For c and l, arrays give mean and standard deviation.
    For a, mean is 0 and the value is the standard deviation.
    
    --- Example input ---
    {
        "c": [4, 2],
        "a": 2,
        "l": [7, 2],
    }
    
    c: value of Constant mean fn
    a: amplitude of SqExp cov fn
    l: time-scale of SqExp cov fn
    '''
    def __init__(self, model_dict=None):
        if model_dict is None:
            self.c = [4, 2]
            self.a = 2
            self.l = [7, 2]
        else:
            self.c = model_dict['c']
            self.a = model_dict['a']
            self.l = model_dict['l']

    '''
    fit
    ---
    Fits a PyMC3 model for a latent GP with Generalized Poisson likelihood
    to the given data.
    Samples all model parameters from the posterior.
    '''
    def fit(self, y_tr, n_future):
        T = len(y_tr)
        self.F = n_future
        t = np.arange(T+self.F)[:,None]

        with pm.Model() as self.model:
            c = pm.TruncatedNormal('mean', mu=self.c[0], sigma=self.c[1], lower=0)
            mean_func = pm.gp.mean.Constant(c=c)
            
            a = pm.HalfNormal('amplitude', sigma=self.a)
            l = pm.TruncatedNormal('time-scale', mu=self.l[0], sigma=self.l[1], lower=0)
            cov_func = a**2 * pm.gp.cov.ExpQuad(input_dim=1, ls=l)
            
            self.gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
            self.f = self.gp.prior('f', X=t)

            self.lam = pm.TruncatedNormal('lam', mu=0, sigma=0.1, lower=-1, upper=1)
            y_past = GenPoisson('y_past', theta=tt.exp(self.f[:T]), lam=self.lam, observed=y_tr, testval=1)
            y_past_logp = pm.Deterministic('y_past_logp', y_past.logpt)

            self.trace = pm.sample(5000, tune=1000, target_accept=.98, chains=2, random_seed=42, cores=1,
                                   init='adapt_diag', max_treedepth=15)

            summary = pm.summary(self.trace)['mean'].to_dict()
            print('Posterior Means:')
            for key in ['mean', 'amplitude', 'time-scale', 'lam']:
                print(key, summary[key])
            print()

            print('Training Scores:')
            logp_samples = self.trace.get_values('y_past_logp', chains=0)
            scores = np.zeros(10)
            for i in range(10):
                scores[i] = np.log(np.mean(np.exp(logp_samples[500*i : 500*i+500]))) / T
            print(f'Chain 1: {np.mean(scores)} ± {scipy.stats.sem(scores)}')
            logp_samples = self.trace.get_values('y_past_logp', chains=1)
            scores = np.zeros(10)
            for i in range(10):
                scores[i] = np.log(np.mean(np.exp(logp_samples[500*i : 500*i+500]))) / T
            print(f'Chain 2: {np.mean(scores)} ± {scipy.stats.sem(scores)}')
            print()

    '''
    score
    -----
    Returns the heldout log probability of the given dataset under the model.
    '''
    def score(self, y_va):
        assert len(y_va) == self.F

        with self.model:
            y_future = GenPoisson('y_future', theta=tt.exp(self.f[-self.F:]), lam=self.lam, observed=y_va)
            y_logp = pm.Deterministic('y_logp', y_future.logpt)
            logp_list = pm.sample_posterior_predictive(self.trace, vars=[y_logp], keep_size=True)

        print('Heldout Scores:')
        logp_samples = logp_list['y_logp'][0]
        scores = np.zeros(10)
        for i in range(10):
            scores[i] = np.log(np.mean(np.exp(logp_samples[500*i : 500*i+500]))) / self.F
        mean_score = np.mean(scores)
        print(f'Chain 1: {mean_score} ± {scipy.stats.sem(scores)}')
        logp_samples = logp_list['y_logp'][1]
        scores = np.zeros(10)
        for i in range(10):
            scores[i] = np.log(np.mean(np.exp(logp_samples[500*i : 500*i+500]))) / self.F
        print(f'Chain 2: {np.mean(scores)} ± {scipy.stats.sem(scores)}')
        print()
        return mean_score

    '''
    forecast
    --------
    Samples from the joint predictive distribution. Writes each set of forecasts to a CSV file.
    '''
    def forecast(self, output_csv_file_pattern=None):
        with self.model:
            y_pred = GenPoisson('y_pred', theta=tt.exp(self.f[-self.F:]), lam=self.lam, shape=self.F, testval=1)
            forecasts = pm.sample_posterior_predictive(self.trace, vars=[y_pred], keep_size=True, random_seed=42)
        samples = forecasts['y_pred'][0]

        if output_csv_file_pattern != None:
            for i in range(len(samples)):
                if(i % 1000 == 0):
                    print(f'Saved {i} forecasts...')
                output_dict = {'forecast': samples[i]}
                output_df = pd.DataFrame(output_dict)
                output_df.to_csv(output_csv_file_pattern.replace('*', str(i+1)))

        return samples

