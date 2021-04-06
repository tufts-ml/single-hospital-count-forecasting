'''
ggp_forecast.py
---------------
Fits a GenPoissonGaussianProcess model with the parameters in `model_dict` to
the given count data. Produces 5000 sets of forecasts for n_predictions
days ahead. Writes forecasts to CSV files with the given filename pattern,
and plots forecasts on `ax`.
'''

from GenPoissonGaussianProcess import GenPoissonGaussianProcess
from plot_forecasts import plot_forecasts

def ggp_forecast(model_dict, counts, n_predictions,
                 output_csv_file_pattern, start, ax):
    
    T = len(counts)

    model = GenPoissonGaussianProcess(model_dict)
    model.fit(counts, n_predictions)

    samples = model.forecast(output_csv_file_pattern)
    ax.set_title('GGP Forecasts')
    plot_forecasts(samples, start, ax, counts[-30:], [])
