
import os

import pandas as pd
import matplotlib.pyplot as plt

from myprojects.validation_good_practice.data_readers.interface import reader, calc_anomaly, collocate
from myprojects.validation import estimate_tau, estimate_lag1_autocorr, calc_bootstrap_blocklength, bootstrap, tc

from myprojects.validation_good_practice.temp.plot import plot_ease_img


def plot_bl():

    res = pd.read_csv(r'D:\work\validation_good_practice\confidence_invervals\blocklengths.csv',index_col=0)

    plt.figure(figsize=(14,5))

    plt.subplot(121)
    plot_ease_img(res, 'bl_abs', cbrange=[0,160], title='Blocklength (abs)', cmap='jet')

    plt.subplot(122)
    plot_ease_img(res, 'bl_anom', cbrange=[0,25], title='Blocklength (anom)', cmap='jet')

    plt.tight_layout()
    plt.show()


def estimate_bl():

    io = reader()
    result_file = r'D:\work\validation_good_practice\confidence_invervals\blocklengths.csv'
    lut = pd.read_csv(r"D:\data_sets\EASE2_grid\grid_lut.csv", index_col=0)

    sensors = ['ASCAT','AMSR2','MERRA2']

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print '%i / %i' % (cnt, len(lut))
        try:
            df = io.read(gpi, sensors=sensors)
            tau_abs = estimate_tau(df, n_lags=180)

            df_anom = calc_anomaly(df)
            tau_anom = estimate_tau(df_anom, n_lags=60)

            df_matched = collocate(df)
            df_anom_matched = collocate(df_anom)

            ac_avg = estimate_lag1_autocorr(df_matched, tau_abs)
            ac_avg_anom = estimate_lag1_autocorr(df_anom_matched, tau_anom)

            bl = calc_bootstrap_blocklength(df_matched, ac_avg)
            bl_anom = calc_bootstrap_blocklength(df_anom_matched, ac_avg_anom)

            result = pd.DataFrame({'lon': data.ease2_lon, 'lat': data.ease2_lat,
                                   'row': data.ease2_row, 'col': data.ease2_col,
                                   'bl_abs': bl,
                                   'bl_anom': bl_anom}, index=(gpi,))
        except:
            continue

        if (os.path.isfile(result_file) == False):
            result.to_csv(result_file, float_format='%0.4f')
        else:
            result.to_csv(result_file, float_format='%0.4f', mode='a', header=False)



if __name__=='__main__':
    plot_bl()
