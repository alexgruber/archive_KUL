
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from netCDF4 import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from pathlib import Path

from pyldas.interface import LDAS_io
from pyldas.templates import template_error_Tb40

from myprojects.experiments.MadKF.CLSM.ensemble_covariance import plot_ease_img2

def plot_fig(res, fname, title=''):

    f = plt.figure(figsize=(20, 8))
    p = sns.lineplot(x='idx', y='corr', hue='parameter', data=res.melt('idx', res.columns[:-1], 'parameter', 'corr'),
                     ax=plt.gca())
    plt.axhline(color='k', linestyle='--', linewidth=1)
    p.set_xticklabels(res.index, rotation=90)
    plt.xlabel('')
    plt.ylabel('Correlation')
    plt.title(title)

    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def plot_r_obs_fcst_err_mod_params():

    stratify = None
    # stratify = 'vegcls'
    # stratify = 'RTMvegcls'

    dir_out = Path('/Users/u0116961/Documents/work/MadKF/CLSM/error_pattern_assesment')

    lai = Dataset('/Users/u0116961/data_sets/MERRA2/MERRA2_images.nc4')['LAI'][:,:,:].mean(axis=0) # [lat,lon]
    lats = Dataset('/Users/u0116961/data_sets/MERRA2/MERRA2_images.nc4')['lat'][:]
    lons = Dataset('/Users/u0116961/data_sets/MERRA2/MERRA2_images.nc4')['lon'][:]

    root = Path(f'~/Documents/work/MadKF/CLSM/SMAP/iter_31').expanduser()

    fname = root / 'result_files' / 'mse_corrected.csv'
    mse = pd.read_csv(fname, index_col=0)

    params1 = LDAS_io(exp='US_M36_SMAP_TB_MadKF_DA_it31').read_params('catparam')
    params2 = LDAS_io(exp='US_M36_SMAP_TB_MadKF_DA_it31').read_params('RTMparam')
    params2.columns = [f'RTM{c}' for c in params2.columns]
    params = pd.concat((params1,params2), axis='columns')

    tc = LDAS_io().grid.tilecoord

    mse['lat'] = tc['com_lat']
    mse['lon'] = tc['com_lon']
    mse['LAI'] = np.nan

    for spc in range(1,5):
        params[f'mse_obs_spc{spc}'] = mse[f'mse_obs_spc{spc}']
        params[f'mse_fcst_spc{spc}'] = mse[f'mse_fcst_spc{spc}']

    for idx in mse.index:
        ind_lat = np.argmin(np.abs(lats-mse.loc[idx]['lat']))
        ind_lon = np.argmin(np.abs(lons-mse.loc[idx]['lon']))
        mse.loc[idx, 'LAI'] = lai[ind_lat, ind_lon]
        params.loc[idx, 'LAI'] = lai[ind_lat, ind_lon]

    cols = [f'mse_obs_spc{spc}' for spc in range(1,5)] + [f'mse_fcst_spc{spc}' for spc in range(1,5)]

    sns.set_context('talk', font_scale=0.8)

    if stratify is not None:
        clss = np.unique(params[stratify])
        clss = clss[clss != -9999]
        for cls in clss:
            tmp_params = params[params[stratify] == cls].drop(stratify, axis='columns')
            corr = tmp_params.corr()[cols].drop(cols).dropna().sort_values('mse_fcst_spc1')
            corr['idx'] = corr.index
            fout = dir_out / f'corr_{stratify}_{cls}.png'
            plot_fig(corr, fout, title=f'{stratify}_{cls} ({len(tmp_params)})')
    else:
        corr = params.corr()[cols].drop(cols).dropna().sort_values('mse_fcst_spc1')
        corr['idx'] = corr.index
        # fout = dir_out / 'corr_all.png'
        plot_fig(corr, fout, title=f'{stratify}_{cls}')

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    plot_r_obs_fcst_err_mod_params()