
import os
import platform

import numpy as np
import pandas as pd

from copy import deepcopy
from multiprocessing import Pool

from pyapi.api import API
from pyass.filter import EnKF, MadEnKF

from myprojects.readers.ascat import HSAF_io
from myprojects.readers.mswep import MSWEP_io
from myprojects.readers.insitu import ISMN_io

from myprojects.timeseries import calc_anomaly

def lonlat2gpi(lon,lat,grid):

    londif = np.abs(grid.lon - lon)
    latdif = np.abs(grid.lat - lat)
    idx = np.where((np.abs(londif - londif.min()) < 0.0001) & (np.abs(latdif - latdif.min()) < 0.0001))[0][0]

    return grid.iloc[idx]['dgg_gpi']

def main():

    # part = 1
    # run(part)

    parts = np.arange(6) + 1
    p = Pool(6)
    p.map(run, parts)

def run(part):
    parts = 6

    ismn = ISMN_io()
    ascat = HSAF_io()
    mswep = MSWEP_io()

    # Select only SCAN and USCRN
    # ismn.list = ismn.list[(ismn.list.network=='SCAN')|(ismn.list.network=='USCRN')]
    ismn.list.index = np.arange(len(ismn.list))

    # Split station list in 4 parts for parallelization
    subs = (np.arange(parts + 1) * len(ismn.list) / parts).astype('int')
    subs[-1] = len(ismn.list)
    start = subs[part - 1]
    end = subs[part]
    ismn.list = ismn.list.iloc[start:end,:]

    if platform.system() == 'Windows':
        result_file = os.path.join('D:', 'work', 'MadEnKF', 'API', 'CONUS', 'ismn_eval', 'result_part%i.csv' % part)
    else:
        result_file = os.path.join('/', 'scratch', 'leuven', '320', 'vsc32046', 'output', 'MadEnKF', 'API', 'CONUS', 'ismn_eval', 'result_part%i.csv' % part)

    dt = ['2007-01-01','2016-12-31']

    for station, insitu in ismn.iter_stations():

        #if True:
        try:
            gpi = lonlat2gpi(station.lon, station.lat, mswep.grid)
            mswep_idx = mswep.grid.index[mswep.grid.dgg_gpi == gpi][0]

            precip = mswep.read(mswep_idx)
            sm = ascat.read(gpi)

            if (precip is None) | (sm is None) | (insitu is None):
                continue

            precip = calc_anomaly(precip[dt[0]:dt[1]], method='harmonic', longterm=False)
            sm = calc_anomaly(sm[dt[0]:dt[1]], method='harmonic', longterm=False)
            insitu = calc_anomaly(insitu['sm_surface'][dt[0]:dt[1]].resample('1d').first(), method='harmonic', longterm=False)

            df = pd.DataFrame({1: precip, 2: sm, 3: insitu}, index=pd.date_range(dt[0],dt[1]))
            df.loc[np.isnan(df[1]), 1] = 0.
            n = len(df)

            if len(df.dropna()) < 50:
                continue

            api = API(gamma=mswep.grid.loc[mswep_idx,'gamma'])

            # --- OL run ---
            x_OL = np.full(n, np.nan)
            model = deepcopy(api)
            for t, f in enumerate(precip.values):
                x = model.step(f)
                x_OL[t] = x

            # --- EnKF run ---
            force_pert = ['normal','additive', 15] # random value for avg error (variance)
            obs_pert = ['normal','additive', 50]  # random value for avg error (variance)
            x_ana_enkf, P_ana_enkf, checkvar_enkf = EnKF(api, df[1].values, df[2].values, force_pert, obs_pert, n_ens=42)

            # --- MadEnKF run ---
            x_ana_madenkf, P_ana_madenkf, R, Q, H, checkvar_madenkf = MadEnKF(api, df[1].values, df[2].values, n_ens=42, n_iter=13)

            df['x_ol'] = x_OL
            df['x_enkf'] = x_ana_enkf
            df['x_madenkf'] = x_ana_madenkf

            corr = df.dropna().corr()
            n_vals = len(df.dropna())

            result = pd.DataFrame({'lon': station.lon, 'lat': station.lat,
                                   'network': station.network, 'station': station.station,
                                   'gpi': gpi,
                                   'R': R, 'Q': Q, 'H': H, 'n': n_vals,
                                   'corr_insitu_ol': corr[3]['x_ol'],
                                   'corr_insitu_enkf': corr[3]['x_enkf'],
                                   'corr_insitu_madenkf': corr[3]['x_madenkf'],
                                   'checkvar_enkf': checkvar_enkf,
                                   'checkvar_madenkf': checkvar_madenkf}, index=(station.name,))

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