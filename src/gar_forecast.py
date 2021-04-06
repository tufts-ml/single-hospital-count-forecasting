'''
gar_forecast.py
---------------
Fits a GenPoissonAutoregression model with the parameters in model_dict to
the given count data. Produces 5000 sets of forecasts for n_predictions
days ahead. Writes forecasts to CSV files with the given filename pattern,
and plots forecasts on `ax`.
'''

from GenPoissonAutoregression import GenPoissonAutoregression
from plot_forecasts import plot_forecasts

def gar_forecast(model_dict, counts, n_predictions,
                 output_csv_file_pattern, start, ax):

    T = len(counts)

    model = GenPoissonAutoregression(model_dict)
    model.fit(counts, n_predictions)

    samples = model.forecast(output_csv_file_pattern)
    ax.set_title('GAR Forecasts')
    plot_forecasts(samples, start, ax, counts[-30:], [])
