

import os

import numpy as np
import pandas as pd

from multiprocessing import Pool

from scipy.stats import pearsonr

from pytesmo.temporal_matching import df_match

from myprojects.readers.insitu import ISMN_io
from myprojects.readers.cci_sm import CCISM_io

from myprojects.timeseries import calc_anomaly

from myprojects.validation import tc


def main():

    part = 3
    run(part)

    # parts = np.arange(6) + 1
    # p = Pool(6)
    # p.map(run, parts)

def run(part):

    parts = 6

    result_file = r'D:\work\ESA_CCI_SM\ismn_r2\ismn_r2_part%i.csv' % part

    cci = CCISM_io()
    ismn = ISMN_io()

    # ismn.list = ismn.list.iloc[100:120]

    # Split station list in 4 parts for parallelization
    subs = (np.arange(parts + 1) * len(ismn.list) / parts).astype('int')
    subs[-1] = len(ismn.list)
    start = subs[part - 1]
    end = subs[part]
    ismn.list = ismn.list.iloc[start:end, :]

    freq = ['abs', 'anom']

    res = ismn.list.copy()
    res.drop(['ease_col','ease_row'], axis='columns', inplace=True)
    res['r_abs'] = np.nan
    res['r_anom'] = np.nan

    for i, (meta, ts_insitu) in enumerate(ismn.iter_stations()):
        print('%i/%i (Proc %i)' % (i, len(ismn.list), part))

        if ts_insitu is None:
            print('No in situ data for ' + meta.network + ' / ' + meta.station)
            continue
        ts_insitu = ts_insitu['2007-10-01':'2014-12-31']
        if len(ts_insitu) < 10:
            print('No in situ data for ' + meta.network + ' / ' + meta.station)
            continue
        df_insitu = pd.DataFrame(ts_insitu).dropna()
        df_insitu_anom = pd.DataFrame(calc_anomaly(ts_insitu)).dropna()

        df_cci = cci.read(meta.lon,meta.lat, version='v04.4', mode=['ACTIVE','PASSIVE']).dropna()
        if len(df_cci) < 10:
            print('No CCI data for ' + meta.network + ' / ' + meta.station)
            continue

        for f in freq:
            if f == 'abs':
                matched = df_match(df_cci, df_insitu, window=0.5)
            else:
                df_cci.loc[:, 'ACTIVE_v04.4'] = calc_anomaly(df_cci['ACTIVE_v04.4'])
                df_cci.loc[:, 'PASSIVE_v04.4'] = calc_anomaly(df_cci['PASSIVE_v04.4'])
                df_cci.dropna(inplace=True)
                if (len(df_cci) < 10) | (len(df_insitu_anom) < 10):
                    print('No in situ or CCI anomaly data for ' + meta.network + ' / ' + meta.station)
                    continue
                matched = df_match(df_cci, df_insitu_anom, window=0.5)

            data = df_cci.join(matched['insitu']).dropna()

            if len(data) < 100:
                continue

            vals = data[['insitu','ACTIVE_v04.4']].values
            c1, p1 = pearsonr(vals[:, 0], vals[:, 1])
            vals = data[['insitu','PASSIVE_v04.4']].values
            c2, p2 = pearsonr(vals[:, 0], vals[:, 1])
            vals = data[['ACTIVE_v04.4','PASSIVE_v04.4']].values
            c3, p3 = pearsonr(vals[:, 0], vals[:, 1])

            if (c1<0)|(c2<0)|(c3<0)|(p1>0.05)|(p2>0.05)|(p3>0.05):
                continue

            res.loc[meta.name, 'r_' + f] = np.sqrt(tc(data)[1][2])

    res.to_csv(result_file, float_format='%0.4f')

if __name__=='__main__':
    main()


