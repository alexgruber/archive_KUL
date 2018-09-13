
import os
import platform

import numpy as np
import pandas as pd

from multiprocessing import Pool

from pyapi.api import API
from pyass.filter import MadEnKF, EnKF

from myprojects.readers.smos import SMOS_io
from myprojects.readers.ascat import HSAF_io
from myprojects.readers.mswep import MSWEP_io

from myprojects.timeseries import calc_anomaly

from pytesmo.metrics import tcol_snr

import timeit

def main(part):

    io = MSWEP_io()
    cells = np.unique(io.grid.dgg_cell.copy().astype('int'))
    io.close()

    parts = 4

    subs = (np.arange(parts + 1) * len(cells) / parts).astype('int')
    subs[-1] = len(cells)
    start = subs[part - 1]
    end = subs[part]
    cells = cells[start:end]

    print cells

    for cell in np.atleast_1d(cells):
       run(cell)

def run(cell):

    smos = SMOS_io()
    ascat = HSAF_io()
    mswep = MSWEP_io()

    # Median Q from MadEnKF API/CONUS run.
    Q_avg = 16.

    if platform.system() == 'Windows':
        result_file = os.path.join('D:', 'work', 'MadEnKF', 'API', 'CONUS', 'result_%04i.csv' % cell)
    else:
        result_file = os.path.join('/', 'scratch', 'leuven', '320', 'vsc32046', 'output', 'MadEnKF', 'API', 'result_%04i.csv' % cell)

    dt = ['2012-01-01','2015-12-31']

    for data, info in mswep.iter_cell(cell):

        # print info.name
        # if True:
        try:
            precip = mswep.read(info.name)
            sm_ascat = ascat.read(info.dgg_gpi)
            sm_smos = smos.read(info.smos_gpi)*100.

            if (precip is None) | (sm_ascat is None) | (sm_smos is None):
                continue

            precip = calc_anomaly(precip[dt[0]:dt[1]], method='moving_average', longterm=False)
            sm_ascat = calc_anomaly(sm_ascat[dt[0]:dt[1]], method='moving_average', longterm=False)
            sm_smos = calc_anomaly(sm_smos[dt[0]:dt[1]], method='moving_average', longterm=False)

            api = API(gamma=info.gamma)

            # Regularize time steps
            df = pd.DataFrame({1: precip, 2: sm_ascat, 3: sm_smos}, index=pd.date_range(dt[0],dt[1]))
            df.loc[np.isnan(df[1]), 1] = 0.

            # --- get OL ts  ---
            OL = np.full(len(precip), np.nan)
            model = API(gamma=info.gamma)
            for t, f in enumerate(df[1].values):
                x = model.step(f)
                OL[t] = x

            # collocate OL and satellite data sets.
            df2 = pd.DataFrame({1: OL, 2: sm_ascat, 3: sm_smos}, index=pd.date_range(dt[0], dt[1])).dropna()

            # calculate TCA based uncertainty and scaling coefficients
            snr, err, beta = tcol_snr(df2[1].values, df2[2].values, df2[3].values)
            P_TC = err[0]**2
            Q_TC = P_TC * (1 - info.gamma ** 2)
            R_TC = err[1]**2
            H_TC = beta[1]

            # ----- Run EnKF using TCA-based uncertainties -----
            t = timeit.default_timer()
            forc_pert = ['normal', 'additive', Q_TC]
            obs_pert = ['normal', 'additive', R_TC]
            x_tc, P, checkvar_tc, K_tc = \
                EnKF(api, df[1].values, df[2].values, forc_pert, obs_pert, H=H_TC, n_ens=42)
            t_tc = timeit.default_timer() - t

            # ----- Run EnKF using RMSD-based uncertainties (corrected for model uncertainty) -----
            t = timeit.default_timer()
            P_avg = Q_avg / (1 - info.gamma**2)
            forc_pert = ['normal', 'additive', Q_avg]
            R_rmsd = (np.nanmean((df2[1].values - H_TC * df2[2].values) ** 2) - P_avg)
            if R_rmsd < 0:
                R_rmsd *= -1
            obs_pert = ['normal', 'additive', R_rmsd]
            x_rmsd, P, checkvar_rmsd, K_rmsd = \
                EnKF(api, df[1].values, df[2].values, forc_pert, obs_pert, H=H_TC, n_ens=42)
            t_rmsd = timeit.default_timer() - t

            # ----- Run MadEnKF -----
            t = timeit.default_timer()
            x_madenkf, P, R_madenkf, Q_madenkf, H_madenkf, checkvar_madenkf, K_madenkf = \
                MadEnKF(api, df[1].values, df[2].values, n_ens=42, n_iter=13)
            t_madenkf = timeit.default_timer() - t

            # TC evaluation of assimilation results
            df3 = pd.DataFrame({1: x_tc, 2: x_rmsd, 3: x_madenkf, 4: sm_ascat, 5: sm_smos}, index=pd.date_range(dt[0], dt[1])).dropna()
            P_ana_tc = tcol_snr(df3[1].values, df3[4].values, df3[5].values)[1][0]**2
            P_ana_rmsd = tcol_snr(df3[2].values, df3[4].values, df3[5].values)[1][0]**2
            P_ana_madenkf = tcol_snr(df3[3].values, df3[4].values, df3[5].values)[1][0]**2

            result = pd.DataFrame({'lon': info.lon, 'lat': info.lat,
                                   'col': info.col, 'row': info.row,
                                   'P_tc': P_TC,
                                   'Q_tc': Q_TC,
                                   'R_tc': R_TC,
                                   'H_tc': H_TC,
                                   'K_tc': K_tc,
                                   'checkvar_tc': checkvar_tc,
                                   'Q_avg': Q_avg,
                                   'P_avg': P_avg,
                                   'R_rmsd': R_rmsd,
                                   'K_enkf': K_rmsd,
                                   'checkvar_enkf': checkvar_rmsd,
                                   'R_madenkf': R_madenkf,
                                   'Q_madenkf': Q_madenkf,
                                   'P_madenkf': Q_madenkf / (1 - info.gamma**2),
                                   'H_madenkf': H_madenkf,
                                   'K_madenkf': K_madenkf,
                                   'checkvar_madenkf': checkvar_madenkf,
                                   't_tc': t_tc,
                                   't_rmsd': t_rmsd,
                                   't_madenkf': t_madenkf,
                                   'P_ana_tc': P_ana_tc,
                                   'P_ana_rmsd': P_ana_rmsd,
                                   'P_ana_madenkf': P_ana_madenkf}, index=(info.name,))

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
    # main(4)
    run(601)