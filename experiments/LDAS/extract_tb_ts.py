
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pyldas.interface import LDAS_io

def get_spc_descr(io):
    obsparam = io.read_obsparam()[['orbit', 'pol', 'ang']]
    obsparam.index += 1
    obsparam.loc[obsparam['orbit'] == 1, 'orbit'] = 'A'
    obsparam.loc[obsparam['orbit'] == 2, 'orbit'] = 'D'
    obsparam.loc[obsparam['pol'] == 1, 'pol'] = 'H'
    obsparam.loc[obsparam['pol'] == 2, 'pol'] = 'V'
    obsparam['ang'] = obsparam['ang'].astype('int')
    obsparam['spc_descr'] = obsparam['orbit'] + '_' + obsparam['pol'] + '_' + obsparam['ang'].astype('str')
    return obsparam['spc_descr']


def extract_timeseries():

    col = 60
    row = 60

    outfile = '/Users/u0116961/data_sets/LDASsa_runs/test.csv'

    ofa = LDAS_io('ObsFcstAna', exp='US_M36_SMOS_DA_cal_scaled_yearly')
    cat = LDAS_io('xhourly', exp='US_M36_SMOS_DA_cal_scaled_yearly')

    descr = get_spc_descr(cat)

    res = cat.timeseries[['sm_surface','soil_temp_layer1']].isel(lat=row,lon=col).to_dataframe()
    res.drop(['lat','lon'], axis='columns', inplace=True)
    res.columns = ['soil_moisture','soil_temperature']
    res.index += pd.to_timedelta('2 hours')

    for spc in ofa.timeseries['species'].values:
        res[descr[spc]] = ofa.timeseries['obs_obs'][spc-1, row, col, :].to_dataframe()['obs_obs'].dropna()

    # res.drop(['soil_moisture','soil_temperature'], axis='columns').interpolate(method='linear').plot()
    # plt.tight_layout()
    # plt.show()

    res.to_csv(outfile, float_format='%.6f')


if __name__=='__main__':

    extract_timeseries()



