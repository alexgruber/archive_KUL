
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from validation_good_practice.data_readers import reader, calc_anomaly, collocate
from myprojects.validation import estimate_tau, estimate_lag1_autocorr

from myprojects.experiments.validation_good_practice.plot import plot_ease_img


def plot_bl():

    res = pd.read_csv(r'D:\work\validation_good_practice\confidence_invervals\avg_ac.csv',index_col=0)

    plt.figure(figsize=(14,5))

    plt.subplot(121)
    plot_ease_img(res, 'ac_avg', cbrange=[0.5,1], title='Lag-1 AC (abs)', cmap='jet')

    plt.subplot(122)
    plot_ease_img(res, 'ac_avg_anom', cbrange=[0,0.6], title='Lag-1 AC (anom)', cmap='jet')

    plt.tight_layout()
    plt.show()


def estimate_bl():

    io = reader()
    result_file = r'D:\work\validation_good_practice\confidence_invervals\avg_spc.csv'
    lut = pd.read_csv(r"D:\data_sets\EASE2_grid\grid_lut.csv", index_col=0)

    sensors = ['MERRA2', 'ASCAT', 'AMSR2']

    if os.path.isfile(result_file):
        idx = pd.read_csv(result_file,index_col=0).index[-1]
        start = np.where(lut.index == idx)[0][0]+1
        lut = lut.iloc[start::,:]

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print('%i / %i' % (cnt, len(lut)))
        try:
            df = io.read(gpi, sensors=sensors)
            tau_abs = estimate_tau(df, n_lags=180)

            df_anom = calc_anomaly(df)
            tau_anom = estimate_tau(df_anom, n_lags=60)

            df_matched = collocate(df)[0]
            df_anom_matched = collocate(df_anom)[0]

            ac_avg, avg_spc = estimate_lag1_autocorr(df_matched, tau_abs)
            ac_avg_anom, avg_spc_anom = estimate_lag1_autocorr(df_anom_matched, tau_anom)

            # bl = calc_bootstrap_blocklength(df_matched, ac_avg)
            # bl_anom = calc_bootstrap_blocklength(df_anom_matched, ac_avg_anom)

            result = pd.DataFrame({'lon': data.ease2_lon, 'lat': data.ease2_lat,
                                   'row': data.ease2_row, 'col': data.ease2_col,
                                   'avg_spc': avg_spc,
                                   'avg_spc_anom': avg_spc_anom}, index=(gpi,))
        except:
        #     print 'error'
            continue

        if (os.path.isfile(result_file) == False):
            result.to_csv(result_file, float_format='%0.4f')
        else:
            result.to_csv(result_file, float_format='%0.4f', mode='a', header=False)



if __name__=='__main__':
    plot_bl()
