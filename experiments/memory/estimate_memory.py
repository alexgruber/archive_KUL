
import os

import pandas as pd

from validation_good_practice.data_readers import reader
from myprojects.validation import estimate_tau

def estimate_memory():

    io = reader()
    result_file = r'D:\work\sm_memory\result.csv'
    lut = pd.read_csv(r"D:\data_sets\EASE2_grid\grid_lut.csv", index_col=0)

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print '%i / %i' % (cnt, len(lut))
        try:
            df = io.read(gpi, return_anomaly=False)
            tau_abs = estimate_tau(df, n_lags=180)

            df = io.read(gpi, return_anomaly=True)
            tau_anom = estimate_tau(df, n_lags=60)

            cols = df.columns.values
            result = pd.DataFrame({'lon': data.ease2_lon, 'lat': data.ease2_lat,
                                   'row': data.ease2_row, 'col': data.ease2_col,
                                   'tau_abs_' + cols[0]: tau_abs[0],
                                   'tau_abs_' + cols[1]: tau_abs[1],
                                   'tau_abs_' + cols[2]: tau_abs[2],
                                   'tau_abs_' + cols[3]: tau_abs[3],
                                   'tau_abs_' + cols[4]: tau_abs[4],
                                   'tau_anom_' + cols[0]: tau_anom[0],
                                   'tau_anom_' + cols[1]: tau_anom[1],
                                   'tau_anom_' + cols[2]: tau_anom[2],
                                   'tau_anom_' + cols[3]: tau_anom[3],
                                   'tau_anom_' + cols[4]: tau_anom[4]}, index=(gpi,))
        except:
            continue

        if (os.path.isfile(result_file) == False):
            result.to_csv(result_file, float_format='%0.4f')
        else:
            result.to_csv(result_file, float_format='%0.4f', mode='a', header=False)

if __name__=='__main__':
    estimate_memory()
