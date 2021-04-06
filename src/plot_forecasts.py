'''
plot_forecasts.py
-----------------
Plots mean, median, 2.5 and 97.5 percentiles for each forecast on the given
axis. Sets xlabels to dates of forecasts, starting from given start date.

Args
----
forecasts : (n_samples, n_days_of_forecasts)
start : start date of forecasts (datetime.date)
ax : axis for plot
past : array of observed counts prior to forecast window
observed : array of observed counts on same days as forecasts
           (either empty if making future predictions, or
            must have same number of days as forecasts)
'''

import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def plot_forecasts(forecasts, start, ax, past, observed):
    n_predictions = len(forecasts[0])

    if len(observed) != 0:
        assert len(observed) == n_predictions, 'observed must be either empty or same length as each set of forecasts'

    low = np.zeros(n_predictions)
    high = np.zeros(n_predictions)
    median = np.zeros(n_predictions)

    for i in range(n_predictions):
        low[i] = np.percentile(forecasts[:,i], 2.5)
        high[i] = np.percentile(forecasts[:,i], 97.5)
        median[i] = np.percentile(forecasts[:,i], 50)

    x_future = np.arange(n_predictions)
    ax.errorbar(x_future, median,
                yerr=[median-low, high-median],
                capsize=2, fmt='x', linewidth=1,
                label='2.5, 50, 97.5 percentiles')

    x_past = np.arange(-len(past), 0)

    if len(observed) == 0:
        ax.plot(x_past, past, '.', label='observed')
    else:
        ax.plot(np.concatenate((x_past, x_future)),
                np.concatenate((past, observed)),
                '.', label='observed')

    ax.set_xticks([-len(past), 0, len(x_future)-1])
    dates = [start + timedelta(-len(past)),
             start,
             start + timedelta(len(x_future)-1)]
    ax.set_xticklabels([date.strftime('%d-%b-%y') for date in dates])


    ax.legend()




