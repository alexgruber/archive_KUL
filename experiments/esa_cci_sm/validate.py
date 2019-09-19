

import os

import numpy as np
import pandas as pd

from multiprocessing import Pool

from scipy.stats import pearsonr

from pytesmo.temporal_matching import df_match

from myprojects.readers.insitu import ISMN_io
from myprojects.readers.cci_sm import CCISM_io

from myprojects.timeseries import calc_anomaly



def main():

    part = 3
    run(part)

    # parts = np.arange(6) + 1
    # p = Pool(6)
    # p.map(run, parts)

def run(part):

    parts = 6

    result_file = r'D:\work\ESA_CCI_SM\validation_%i.csv' % part

    cci = CCISM_io()
    ismn = ISMN_io()

    # ismn.list = ismn.list.iloc[100:120]

    # Split station list in 4 parts for parallelization
    subs = (np.arange(parts + 1) * len(ismn.list) / parts).astype('int')
    subs[-1] = len(ismn.list)
    start = subs[part - 1]
    end = subs[part]
    ismn.list = ismn.list.iloc[start:end, :]

    periods = { 'p1': ['2007-10-01', '2010-01-14'],
                'p2': ['2010-01-15', '2011-10-04'],
                'p3': ['2011-10-05', '2012-06-30'],
                'p4': ['2012-07-01', '2014-12-31']}

    freq = ['abs', 'anom']

    corr_tags = ['corr_'+m+'_'+v+'_'+p+'_'+f for m in cci.modes for v in cci.versions for p in periods.keys() for f in freq]
    p_tags = ['p_'+m+'_'+v+'_'+p+'_'+f for m in cci.modes for v in cci.versions for p in periods.keys() for f in freq]
    n_tags = ['n_'+m+'_'+v+'_'+p+'_'+f for m in cci.modes for v in cci.versions for p in periods.keys() for f in freq]

    res = ismn.list.copy()
    res.drop(['ease_col','ease_row'], axis='columns', inplace=True)
    for col in corr_tags + p_tags:
        res[col] = np.nan
    for col in n_tags:
        res[col] = 0

    for i, (meta, ts_insitu) in enumerate(ismn.iter_stations()):
        print('%i/%i (Proc %i)' % (i, len(ismn.list), part))

        if ts_insitu is None:
            print('No in situ data for ' + meta.network + ' / ' + meta.station)
            continue
        ts_insitu = ts_insitu[periods['p1'][0]:periods['p4'][1]]
        if len(ts_insitu) < 10:
            print('No in situ data for ' + meta.network + ' / ' + meta.station)
            continue
        df_insitu = pd.DataFrame(ts_insitu).dropna()
        df_insitu_anom = pd.DataFrame(calc_anomaly(ts_insitu)).dropna()

        for m in cci.modes:
            df_cci = cci.read(meta.lon,meta.lat, mode=m).dropna()
            if len(df_cci) < 10:
                print('No CCI ' + m + ' data for ' + meta.network + ' / ' + meta.station)
                continue

            for f in freq:
                if f == 'abs':
                    matched = df_match(df_cci, df_insitu, window=0.5)
                else:
                    for v in cci.versions:
                        df_cci.loc[:, m + '_' + v] = calc_anomaly(df_cci[m + '_' + v])
                    df_cci.dropna(inplace=True)
                    if (len(df_cci) < 10) | (len(df_insitu_anom) < 10):
                        print('No in situ or CCI ' + m + ' anomaly data for ' + meta.network + ' / ' + meta.station)
                        continue
                    matched = df_match(df_cci, df_insitu_anom, window=0.5)

                data = df_cci.join(matched['insitu']).dropna()

                for p in periods.keys():
                    vals = data[periods[p][0]:periods[p][1]].values

                    n_matches = vals.shape[0]
                    if n_matches < 10:
                        continue
                    for k, v in enumerate(cci.versions):
                        corr, p_value = pearsonr(vals[:,k], vals[:,-1])
                        res.loc[meta.name, 'corr_' + m + '_' + v + '_' + p + '_' + f] = corr
                        res.loc[meta.name, 'p_' + m + '_' + v + '_' + p + '_' + f] = p_value
                        res.loc[meta.name, 'n_' + m + '_' + v + '_' + p + '_' + f] = n_matches

    res.to_csv(result_file, float_format='%0.4f')

if __name__=='__main__':
    main()


