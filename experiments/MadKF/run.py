
import os
import platform

import numpy as np
import pandas as pd

from multiprocessing import Pool

from pyapi.api import API
from pyass.filter import MadKF, EnKF, KF, KF_2D

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

    parts = 12

    subs = (np.arange(parts + 1) * len(cells) / parts).astype('int')
    subs[-1] = len(cells)
    start = subs[part - 1]
    end = subs[part]
    cells = cells[start:end]

    print cells

    for cell in np.atleast_1d(cells):
       run(cell=cell)

def run(cell=None, gpi=None):

    if (cell is None) and (gpi is None):
        print 'No cell/gpi specified.'
        return

    smos = SMOS_io()
    ascat = HSAF_io(ext=None)
    mswep = MSWEP_io()

    if gpi is not None:
        cell = mswep.gpi2cell(gpi)

    # Median Q/R from TC run.
    Q_avg = 12.
    R_avg = 74.

    if platform.system() == 'Windows':
        result_file = os.path.join('D:', 'work', 'MadKF', 'CONUS', 'result_%04i.csv' % cell)
    else:
        result_file = os.path.join('/', 'scratch', 'leuven', '320', 'vsc32046', 'output', 'MadKF', 'CONUS', 'result_%04i.csv' % cell)

    dt = ['2012-01-01','2015-12-31']

    for data, info in mswep.iter_cell(cell, gpis=gpi):

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

            n_inv_precip = len(np.where(np.isnan(df[1]))[0])
            n_inv_ascat = len(np.where(np.isnan(df[2]))[0])
            n_inv_smos = len(np.where(np.isnan(df[3]))[0])
            n_inv_asc_smo = len(np.where(np.isnan(df[2]) & np.isnan(df[3]))[0])

            df.loc[np.isnan(df[1]), 1] = 0.

            # --- get OL ts  ---
            OL = np.full(len(precip), np.nan)
            model = API(gamma=info.gamma)
            for t, f in enumerate(df[1].values):
                x = model.step(f)
                OL[t] = x

            # collocate OL and satellite data sets.
            df2 = pd.DataFrame({1: OL, 2: sm_ascat, 3: sm_smos}, index=pd.date_range(dt[0], dt[1])).dropna()

            # ----- Calculate uncertainties -----
            # convert (static) forcing to model uncertainty
            P_avg = Q_avg / (1 - info.gamma**2)

            # calculate TCA based uncertainty and scaling coefficients
            snr, err, beta = tcol_snr(df2[1].values, df2[2].values, df2[3].values)
            P_TC = err[0]**2
            Q_TC = P_TC * (1 - info.gamma ** 2)
            R_TC = (err[1]/beta[1])**2
            H_TC = beta[1]

            # Calculate RMSD based uncertainty
            R_rmsd = (np.nanmean((df2[1].values - H_TC * df2[2].values) ** 2) - P_avg)
            if R_rmsd < 0:
                R_rmsd *= -1
            # -----------------------------------

            # ----- Run KF using TCA-based uncertainties -----
            api_kf = API(gamma=info.gamma, Q=Q_TC)
            R_2D = np.array([(err[1]/beta[1])**2, (err[2]/beta[2])**2])
            H_2D = np.array([beta[1]**(-1), beta[2]**(-1)])
            x_2d, P, checkvar1_2d, checkvar2_2d, checkvar3_2d, K1_2d, K2_2d = \
                KF_2D(api_kf, df[1].values.copy(), df[2].values.copy(), df[3].values.copy(), R_2D, H=H_2D)

            # ----- Run KF using TCA-based uncertainties -----
            api_kf = API(gamma=info.gamma, Q=Q_TC)
            x_kf, P, R_innov_kf, checkvar_kf, K_kf = \
                KF(api_kf, df[1].values.copy(), df[2].values.copy(), R_TC, H=H_TC)

            # ----- Run EnKF using TCA-based uncertainties -----
            forc_pert = ['normal', 'additive', Q_TC]
            obs_pert = ['normal', 'additive', R_TC]
            x_tc, P, R_innov_tc, checkvar_tc, K_tc = \
                EnKF(api, df[1].values.copy(), df[2].values.copy(), forc_pert, obs_pert, H=H_TC, n_ens=50)

            # ----- Run EnKF using static uncertainties -----
            forc_pert = ['normal', 'additive', Q_avg]
            obs_pert = ['normal', 'additive', R_avg]
            x_avg, P, R_innov_avg, checkvar_avg, K_avg = \
                EnKF(api, df[1].values.copy(), df[2].values.copy(), forc_pert, obs_pert, H=H_TC, n_ens=50)

            # ----- Run EnKF using RMSD-based uncertainties (corrected for model uncertainty) -----
            t = timeit.default_timer()
            forc_pert = ['normal', 'additive', Q_avg]
            obs_pert = ['normal', 'additive', R_rmsd]
            x_rmsd, P, R_innov_rmsd, checkvar_rmsd, K_rmsd = \
                EnKF(api, df[1].values.copy(), df[2].values.copy(), forc_pert, obs_pert, H=H_TC, n_ens=50)
            t_enkf = timeit.default_timer() - t

            # ----- Run MadKF -----
            t = timeit.default_timer()
            x_madkf, P, R_madkf, Q_madkf, H_madkf, R_innov_madkf, checkvar_madkf, K_madkf = \
                MadKF(api, df[1].values.copy(), df[2].values.copy(), n_ens=100, n_iter=20)
            t_madkf = timeit.default_timer() - t

            # TC evaluation of assimilation results
            # df3 = pd.DataFrame({1: x_tc, 2: x_avg, 3: x_rmsd, 4: x_madkf, 5: sm_ascat, 6: sm_smos}, index=pd.date_range(dt[0], dt[1])).dropna()
            #
            # rmse_ana_tc = tcol_snr(df3[1].values, df3[5].values, df3[6].values)[1][0]
            # rmse_ana_avg = tcol_snr(df3[2].values, df3[5].values, df3[6].values)[1][0]
            # rmse_ana_rmsd = tcol_snr(df3[3].values, df3[5].values, df3[6].values)[1][0]
            # rmse_ana_madkf = tcol_snr(df3[4].values, df3[5].values, df3[6].values)[1][0]

            result = pd.DataFrame({'lon': info.lon, 'lat': info.lat,
                                   'col': info.col, 'row': info.row,
                                   'P_tc': P_TC,
                                   'Q_tc': Q_TC,
                                   'R_tc': R_TC,
                                   'H_tc': H_TC,
                                   'K_tc': K_tc,
                                   'R_innov_tc': R_innov_tc,
                                   'checkvar_tc': checkvar_tc,
                                   'K_kf': K_kf,
                                   'R_innov_kf': R_innov_kf,
                                   'checkvar_kf': checkvar_kf,
                                   'K1_2d': K1_2d,
                                   'K2_2d': K2_2d,
                                   'checkvar1_2d': checkvar1_2d,
                                   'checkvar2_2d': checkvar2_2d,
                                   'checkvar3_2d': checkvar3_2d,
                                   'P_avg': P_avg,
                                   'Q_avg': Q_avg,
                                   'R_avg': R_avg,
                                   'K_avg': K_avg,
                                   'R_innov_avg': R_innov_avg,
                                   'checkvar_avg': checkvar_avg,
                                   'R_rmsd': R_rmsd,
                                   'K_rmsd': K_rmsd,
                                   'R_innov_rmsd': R_innov_rmsd,
                                   'checkvar_rmsd': checkvar_rmsd,
                                   'P_madkf': Q_madkf / (1 - info.gamma**2),
                                   'Q_madkf': Q_madkf,
                                   'R_madkf': R_madkf,
                                   'H_madkf': H_madkf,
                                   'K_madkf': K_madkf,
                                   'R_innov_madkf': R_innov_madkf,
                                   'checkvar_madkf': checkvar_madkf,
                                   't_enkf': t_enkf,
                                   't_madkf': t_madkf,
                                   'n_inv_precip': n_inv_precip,
                                   'n_inv_ascat': n_inv_ascat,
                                   'n_inv_smos': n_inv_smos,
                                   'n_inv_asc_smo': n_inv_asc_smo}, index=(info.name,))

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

    # lat = 37.028681964
    # lon = -120.279189837
    # gpi = MSWEP_io().lonlat2gpi(lon,lat)

    run(cell=601)












