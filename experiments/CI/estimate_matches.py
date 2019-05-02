
import os

import pandas as pd
import matplotlib.pyplot as plt

from validation_good_practice.data_readers import reader, calc_anomaly, collocate

from myprojects.experiments.validation_good_practice.plot import plot_ease_img


def plot_matches():

    res = pd.read_csv(r'D:\work\validation_good_practice\confidence_invervals\n_matches.csv',index_col=0)

    plt.figure(figsize=(14,5))

    plt.subplot(121)
    plot_ease_img(res, 'n_matches_abs', cbrange=[0,700], title='# matches (abs)', cmap='jet')

    plt.subplot(122)
    plot_ease_img(res, 'n_matches_anom', cbrange=[0,700], title='# matches (anom)', cmap='jet')

    plt.tight_layout()
    plt.show()


def estimate_matches():

    io = reader()
    result_file = r'D:\work\validation_good_practice\confidence_invervals\n_matches.csv'
    lut = pd.read_csv(r"D:\data_sets\EASE2_grid\grid_lut.csv", index_col=0)

    sensors = ['MERRA2','ASCAT','AMSR2']

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print '%i / %i' % (cnt, len(lut))
        try:
            df = io.read(gpi, sensors=sensors)
            df_anom = calc_anomaly(df)

            df_matched = collocate(df)
            df_anom_matched = collocate(df_anom)

            result = pd.DataFrame({'lon': data.ease2_lon, 'lat': data.ease2_lat,
                                   'row': data.ease2_row, 'col': data.ease2_col,
                                   'n_matches_abs': len(df_matched.dropna()),
                                   'n_matches_anom': len(df_anom_matched.dropna())}, index=(gpi,))
        except:
            continue

        if (os.path.isfile(result_file) == False):
            result.to_csv(result_file, float_format='%0.4f')
        else:
            result.to_csv(result_file, float_format='%0.4f', mode='a', header=False)

if __name__=='__main__':
    plot_matches()
