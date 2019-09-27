
import numpy as np
import pandas as pd

from pathlib import Path

def config():

    root = Path('/data_sets/GLEAM')
    force_path = root / 'v33b_forcing_ts_chunked'
    static_file = root / 'gleam_v33b_static_ens.nc'
    startup_file = root / 'init_ens.nc'

    timeres = 24
    sdate = '2017-01-01'
    edate = '2017-12-31'
    period = pd.date_range(sdate, edate, freq="D")

    nens = 8

    fAE = np.array([.9, .8, .65, .7, .8])
    alpha = np.array([.97, 1.26, 1.26, 1.1, 1.26])

    d_frac_l1 = 100  # depth of top layer
    d_frac_l2 = 900  # depth of second layer, accessed by herbaceous vegetation
    d_frac_l3 = 1500  # third layer, also accessed by tall vegetation
    ldepth = np.repeat(np.array([d_frac_l1, d_frac_l2, d_frac_l3]).reshape((1, 1, 3, 1)), 8, axis=3)

    return force_path, static_file, startup_file, timeres, period, nens, fAE, alpha, ldepth