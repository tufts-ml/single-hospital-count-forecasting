'''
GenPoisson.py
-------------
Defines a Generalized Poisson distribution as a PyMC3 custom Discrete distribution
'''

import pymc3 as pm
from pymc3.distributions.dist_math import bound, logpow, factln
from pymc3.distributions.distribution import draw_values, generate_samples
import theano.tensor as tt
import numpy as np

class GenPoisson(pm.Discrete):

    def __init__(self, theta, lam, *args, **kwargs):
        super(GenPoisson, self).__init__(*args, **kwargs)
        self.theta = theta
        self.lam = lam

    def logp(self, value):
        theta = self.theta
        lam = self.lam
        return genpoisson_logp(theta, lam, value)

    def random(self, point=None, size=None):
        theta, lam = draw_values([self.theta, self.lam], point=point, size=size)
        return generate_samples(genpoisson_rvs,
                                theta=theta, lam=lam,
                                size=size)

'''
genpoisson_logp
---------------
Returns the log likelihood of the GenPoisson distribution with the given
parameters evaluated at the specified value
'''
def genpoisson_logp(theta, lam, value):
    log_prob = bound(np.log(theta) + logpow(theta + lam * value, value - 1)
                     - (theta + lam * value) - factln(value),
                     theta >= 0,
                     -1 <= lam, -theta/4 <= lam, lam <= 1,
                     value >= 0)
    # Return zero when value > m, where m is the largest pos int for which theta + m * lam > 0 (when lam < 0)
    return tt.switch(theta + value * lam <= 0,
                     0, log_prob)

'''
genpoisson_rvs
--------------
Returns a random sample from the GenPoisson distribution with the given
parameters and shape
Analogous to the `scipy.stats.<dist_name>.rvs`
'''
def genpoisson_rvs(theta, lam, size=None):
    if size is not None:
        assert size == theta.shape
    else:
        size = theta.shape
    lam = lam[0]
    omega = np.exp(-lam)
    X = np.full(size, 0)
    S = np.exp(-theta)
    P = np.copy(S)
    for i in range(size[0]):
        U = np.random.uniform()
        while U > S[i]:
            X[i] += 1
            C = theta[i] - lam + lam * X[i]
            P[i] = omega * C * (1 + lam/C)**(X[i]-1) * P[i] / X[i]
            S[i] += P[i]
    return X