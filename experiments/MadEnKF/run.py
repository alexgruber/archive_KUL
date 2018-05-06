
import os
import platform

import numpy as np
import pandas as pd

from multiprocessing import Pool

from pyapi.api import API
from pyass.filter import MadEnKF, EnKF

from myprojects.readers.ascat import HSAF_io
from myprojects.readers.mswep import MSWEP_io

from myprojects.timeseries import calc_anomaly

import timeit

def main():

    io = MSWEP_io()
    cells = np.unique(io.grid.dgg_cell.copy().astype('int'))
    io.close()

    p = Pool(18)
    p.map(run, cells)

    # cells = 601
    # for cell in np.atleast_1d(cells):
    #    run(cell)

def run(cell):

    ascat = HSAF_io()
    mswep = MSWEP_io()

    # Median Q from MadEnKF API/CONUS run.
    Q_avg = 16.

    if platform.system() == 'Windows':
        result_file = os.path.join('D:', 'work', 'MadEnKF', 'API', 'CONUS', 'result_%04i.csv' % cell)
    else:
        result_file = os.path.join('/', 'scratch', 'leuven', '320', 'vsc32046', 'output', 'MadEnKF', 'API', 'result_%04i.csv' % cell)

    dt = ['2012-01-01','2016-12-31']

    for data, info in mswep.iter_cell(cell):

        # if True:
        try:
            precip = mswep.read(info.name)
            sm = ascat.read(info.dgg_gpi)

            if (precip is None) | (sm is None):
                continue

            precip = calc_anomaly(precip[dt[0]:dt[1]], method='harmonic', longterm=False)
            sm = calc_anomaly(sm[dt[0]:dt[1]], method='harmonic', longterm=False)

            df = pd.DataFrame({1: precip, 2: sm}, index=pd.date_range(dt[0],dt[1]))
            df.loc[np.isnan(df[1]), 1] = 0.

            api = API(gamma=info.gamma)

            t = timeit.default_timer()
            x, P, R_madenkf, Q_madenkf, H_madenkf, checkvar_madenkf, K_madenkf = \
                MadEnKF(api, df[1].values, df[2].values, n_ens=42, n_iter=13)
            t_madenkf = timeit.default_timer() - t

            # --- get OL ts  ---
            OL = np.full(len(precip), np.nan)
            model = API(gamma=info.gamma)
            for t, f in enumerate(df[1].values):
                x = model.step(f)
                OL[t] = x

            P_avg = Q_avg / (1 - info.gamma**2)
            forc_pert = ['normal', 'additive', Q_avg]

            df2 = pd.DataFrame({1: OL, 2: sm}, index=pd.date_range(dt[0], dt[1])).dropna()

            R_rmsd = (np.nanmean((df2[1].values - df2[2].values) ** 2) - P_avg)
            obs_pert = ['normal', 'additive', R_rmsd]
            x, P, checkvar_enkf, K_enkf = \
                EnKF(api, df[1].values, df[2].values, forc_pert, obs_pert, H=None, n_ens=42)

            t = timeit.default_timer()
            R_rmsd_scaled = (np.nanmean((df2[1].values - H_madenkf * df2[2].values) ** 2) - P_avg) / H_madenkf ** 2
            obs_pert = ['normal', 'additive', R_rmsd_scaled]
            x, P, checkvar_enkf_scaled, K_enkf_scaled = \
                EnKF(api, df[1].values, df[2].values, forc_pert, obs_pert, H=H_madenkf, n_ens=42)
            t_enkf = timeit.default_timer() - t

            result = pd.DataFrame({'lon': info.lon, 'lat': info.lat,
                                   'col': info.col, 'row': info.row,
                                   'R_madenkf': R_madenkf,
                                   'Q_madenkf': Q_madenkf,
                                   'P_madenkf': Q_madenkf / (1 - info.gamma**2),
                                   'H_madenkf': H_madenkf,
                                   'K_madenkf': K_madenkf,
                                   'checkvar_madenkf': checkvar_madenkf,
                                   'Q_avg': Q_avg,
                                   'P_avg': P_avg,
                                   'R_rmsd': R_rmsd,
                                   'K_enkf': K_enkf,
                                   'checkvar_enkf': checkvar_enkf,
                                   'R_rmsd_scaled': R_rmsd_scaled,
                                   'K_enkf_scaled': K_enkf_scaled,
                                   'checkvar_enkf_scaled': checkvar_enkf_scaled,
                                   't_madenkf': t_madenkf,
                                   't_enkf': t_enkf}, index=(info.name,))

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
