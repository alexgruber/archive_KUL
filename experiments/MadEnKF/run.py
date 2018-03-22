
import os
import platform

import numpy as np
import pandas as pd

from multiprocessing import Pool

from pyapi.api import API
from pyass.filter import AdEnKF

from myprojects.readers.ascat import HSAF_io
from myprojects.readers.mswep import MSWEP_io

from myprojects.timeseries import calc_anomaly


def main():

    io = MSWEP_io()
    cells = np.unique(io.grid.dgg_cell.copy().astype('int'))
    io.close()

    p = Pool(48)
    p.map(run, cells)

    # cells = 601
    # for cell in np.atleast_1d(cells):cd
    #     run(cell)

def run(cell):

    ascat = HSAF_io()
    mswep = MSWEP_io()

    if platform.system() == 'Windows':
        result_file = os.path.join('D:', 'work', 'API', 'AdEnKF', 'result_%04i.csv' % cell)
    else:
        result_file = os.path.join('/', 'scratch', 'leuven', '320', 'vsc32046', 'output', 'AdEnKF_test', 'result_%04i.csv' % cell)

    dt = ['2012-01-01','2016-12-31']

    for data, info in mswep.iter_cell(cell):

        try:
            precip = mswep.read(info.name)
            sm = ascat.read(info.dgg_gpi)

            if (precip is None) | (sm is None):
                continue

            precip = calc_anomaly(precip[dt[0]:dt[1]], method='harmonic', longterm=False)
            sm = calc_anomaly(sm[dt[0]:dt[1]], method='harmonic', longterm=False)

            df = pd.DataFrame({1: precip, 2: sm}, index=pd.date_range(dt[0],dt[1]))
            df.loc[np.isnan(df[1]), 1] = 0.

            api = API(gamma=info.gamma, Q=np.nan)

            x_ana, P_ana, R, Q, H, checkvar = AdEnKF(api, df[1].values, df[2].values, n_ens=42, n_iter=17)

            result = pd.DataFrame({'lon': info.lon, 'lat': info.lat,
                                   'col': info.col, 'row': info.row,
                                   'R': R, 'Q': Q, 'H': H,
                                   'checkvar': checkvar}, index=(info.name,))

            if (os.path.isfile(result_file) == False):
                result.to_csv(result_file, float_format='%0.4f')
            else:
                result.to_csv(result_file, float_format='%0.4f', mode='a', header=False)
        except:
            print 'GPI failed.'
            continue

    ascat.close()
    mswep.close()

if __name__=='__main__':
    main()