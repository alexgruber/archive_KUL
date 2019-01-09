
import os
import platform

import numpy as np
import pandas as pd

from copy import deepcopy
from multiprocessing import Pool

from pyapi.api import API
from pyass.filter import MadKF, EnKF, KF

from myprojects.readers.smos import SMOS_io
from myprojects.readers.ascat import HSAF_io
from myprojects.readers.mswep import MSWEP_io
from myprojects.readers.insitu import ISMN_io

from myprojects.timeseries import calc_anomaly

from myprojects.validation import tc

def lonlat2gpi(lon,lat,grid):

    londif = np.abs(grid.lon - lon)
    latdif = np.abs(grid.lat - lat)
    idx = np.where((np.abs(londif - londif.min()) < 0.0001) & (np.abs(latdif - latdif.min()) < 0.0001))[0][0]

    return grid.iloc[idx]['dgg_gpi']

def main():

    # part = 3
    # run(part)

    parts = np.arange(4) + 1
    p = Pool(4)
    p.map(run, parts)

def run(part):
    parts = 4

    smos = SMOS_io()
    ismn = ISMN_io()
    ascat = HSAF_io(ext=None)
    mswep = MSWEP_io()

    # Median Q from MadKF API/CONUS run.
    Q_avg = 12.
    R_avg = 74.

    # Select only SCAN and USCRN
    ismn.list.index = np.arange(len(ismn.list))

    # Split station list in 4 parts for parallelization
    subs = (np.arange(parts + 1) * len(ismn.list) / parts).astype('int')
    subs[-1] = len(ismn.list)
    start = subs[part - 1]
    end = subs[part]
    ismn.list = ismn.list.iloc[start:end,:]

    if platform.system() == 'Windows':
        result_file = os.path.join('D:', 'work', 'MadKF', 'CONUS', 'ismn_eval', 'result_part%i.csv' % part)
    else:
        result_file = os.path.join('/', 'scratch', 'leuven', '320', 'vsc32046', 'output', 'MadKF', 'CONUS', 'ismn_eval', 'result_part%i.csv' % part)

    dt = ['2010-01-01','2015-12-31']

    for cnt, (station, insitu) in enumerate(ismn.iter_stations(surf_depth=0.1)):
        print '%i / %i' % (cnt, len(ismn.list))

        # if True:
        try:
            gpi = lonlat2gpi(station.lon, station.lat, mswep.grid)
            mswep_idx = mswep.grid.index[mswep.grid.dgg_gpi == gpi][0]
            smos_gpi = mswep.grid.loc[mswep_idx,'smos_gpi']

            precip = mswep.read(mswep_idx)
            sm_ascat = ascat.read(gpi)
            sm_smos = smos.read(smos_gpi)*100.

            if (precip is None) | (sm_ascat is None) | (sm_smos is None) | (insitu is None):
                continue

            precip = calc_anomaly(precip[dt[0]:dt[1]], method='moving_average', longterm=False)
            sm_ascat = calc_anomaly(sm_ascat[dt[0]:dt[1]], method='moving_average', longterm=False)
            sm_smos = calc_anomaly(sm_smos[dt[0]:dt[1]], method='moving_average', longterm=False)
            insitu = calc_anomaly(insitu['sm_surface'][dt[0]:dt[1]].resample('1d').first(), method='moving_average', longterm=False)

            df = pd.DataFrame({1: precip, 2: sm_ascat, 3: sm_smos, 4: insitu}, index=pd.date_range(dt[0],dt[1]))
            df.loc[np.isnan(df[1]), 1] = 0.
            n = len(df)

            if len(df.dropna()) < 50:
                continue
            gamma = mswep.grid.loc[mswep_idx,'gamma']
            api = API(gamma=gamma)

            # --- OL run ---
            x_OL = np.full(n, np.nan)
            model = deepcopy(api)
            for t, f in enumerate(precip.values):
                x = model.step(f)
                x_OL[t] = x

            # ----- Calculate uncertainties -----
            # convert (static) forcing to model uncertainty
            P_avg = Q_avg / (1 - gamma ** 2)

            # calculate TCA based uncertainty and scaling coefficients
            tmp_df = pd.DataFrame({1: x_OL, 2: sm_ascat, 3: sm_smos}, index=pd.date_range(dt[0], dt[1])).dropna()
            snr, r_tc, err, beta = tc(tmp_df)
            P_TC = err[0] ** 2
            Q_TC = P_TC * (1 - gamma ** 2)
            R_TC = (err[1] / beta[1]) ** 2
            H_TC = beta[1]

            # Calculate RMSD based uncertainty
            R_rmsd = (np.nanmean((tmp_df[1].values - H_TC * tmp_df[2].values) ** 2) - P_avg)
            if R_rmsd < 0:
                R_rmsd *= -1
            # -----------------------------------

            # ----- Run KF using TCA-based uncertainties -----
            api_kf = API(gamma=gamma, Q=Q_TC)
            x_kf, P, R_innov_kf, checkvar_kf, K_kf = \
                KF(api_kf, df[1].values.copy(), df[2].values.copy(), R_TC, H=H_TC)

            # ----- Run EnKF using static uncertainties -----
            forc_pert = ['normal', 'additive', Q_avg]
            obs_pert = ['normal', 'additive', R_avg]
            x_avg, P, R_innov_avg, checkvar_avg, K_avg = \
                EnKF(api, df[1].values.copy(), df[2].values.copy(), forc_pert, obs_pert, H=H_TC, n_ens=50)

            # ----- Run EnKF using RMSD-based uncertainties (corrected for model uncertainty) -----
            forc_pert = ['normal', 'additive', Q_avg]
            obs_pert = ['normal', 'additive', R_rmsd]
            x_rmsd, P, R_innov_rmsd, checkvar_rmsd, K_rmsd = \
                EnKF(api, df[1].values.copy(), df[2].values.copy(), forc_pert, obs_pert, H=H_TC, n_ens=50)

            # ----- Run MadKF -----
            x_madkf, P, R_madkf, Q_madkf, H_madkf, R_innov_madkf, checkvar_madkf, K_madkf = \
                MadKF(api, df[1].values.copy(), df[2].values.copy(), n_ens=100, n_iter=20)

            df['x_ol'] = x_OL
            df['x_kf'] = x_kf
            df['x_avg'] = x_avg
            df['x_rmsd'] = x_rmsd
            df['x_madkf'] = x_madkf

            tc_ol = tc(df[[4,3,'x_ol']])
            tc_kf = tc(df[[4,3,'x_kf']])
            tc_avg = tc(df[[4,3,'x_avg']])
            tc_rmsd = tc(df[[4,3,'x_rmsd']])
            tc_madkf = tc(df[[4,3,'x_madkf']])

            corr = df.dropna().corr()
            n_all = len(df.dropna())

            result = pd.DataFrame({'lon': station.lon, 'lat': station.lat,
                                   'network': station.network, 'station': station.station,
                                   'gpi': gpi,
                                   'n_all': n_all,
                                   'corr_ol': corr[4]['x_ol'],
                                   'corr_kf': corr[4]['x_kf'],
                                   'corr_avg': corr[4]['x_avg'],
                                   'corr_rmsd': corr[4]['x_rmsd'],
                                   'corr_madkf': corr[4]['x_madkf'],
                                   'snr_ol': tc_ol[0][2],
                                   'snr_kf': tc_kf[0][2],
                                   'snr_avg': tc_avg[0][2],
                                   'snr_rmsd': tc_rmsd[0][2],
                                   'snr_madkf': tc_madkf[0][2],
                                   'r_ol': tc_ol[1][2],
                                   'r_kf': tc_kf[1][2],
                                   'r_avg': tc_avg[1][2],
                                   'r_rmsd': tc_rmsd[1][2],
                                   'r_madkf': tc_madkf[1][2],
                                   'rmse_ol': tc_ol[2][2],
                                   'rmse_kf': tc_kf[2][2],
                                   'rmse_avg': tc_avg[2][2],
                                   'rmse_rmsd': tc_rmsd[2][2],
                                   'rmse_madkf': tc_madkf[2][2],
                                   'checkvar_kf': checkvar_kf,
                                   'checkvar_avg': checkvar_avg,
                                   'checkvar_rmsd': checkvar_rmsd,
                                   'checkvar_madkf': checkvar_madkf,
                                   'R_innov_kf': R_innov_kf,
                                   'R_innov_avg': R_innov_avg,
                                   'R_innov_rmsd': R_innov_rmsd,
                                   'R_innov_madkf': R_innov_madkf}, index=(station.name,))

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
