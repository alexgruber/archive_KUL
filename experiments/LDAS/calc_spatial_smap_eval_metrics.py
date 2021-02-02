import warnings
warnings.filterwarnings("ignore")

import h5py

import numpy as np
import pandas as pd

from pathlib import Path

# import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from validation_good_practice.ancillary.grid import EASE2

from pyldas.interface import LDAS_io

# sns.set_context('talk', font_scale=0.8)

def plot_ease_img(data,
                  # llcrnrlat=24,
                  # urcrnrlat=51,
                  # llcrnrlon=-128,
                  # urcrnrlon=-64,
                  llcrnrlat=-90,
                  urcrnrlat=90,
                  llcrnrlon=-180,
                  urcrnrlon=180,
                  cmap='jet_r',
                  fontsize=12):

    grid = EASE2()
    lons, lats = np.meshgrid(grid.ease_lons, grid.ease_lats)
    img_masked = np.ma.masked_invalid(data)
    m = Basemap(projection='mill',
                llcrnrlat=llcrnrlat,
                urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,
                urcrnrlon=urcrnrlon,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    cb = m.colorbar(im, "bottom", size="8%", pad="4%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)

    return im

def calc_metrics():

    fdir = Path('/Users/u0116961/data_sets/SMAP/SPL2SMP.006')
    files = sorted(fdir.glob('**/*.h5'))
    dates = pd.to_datetime([str(f)[-29:-14] for f in files]).round('3h') - pd.to_timedelta('2 hours') # timedelta b/c wrongly assigned in LDAS!!

    # Get LDAS domain information
    ds = LDAS_io('xhourly', 'US_M36_SMOS40_TB_MadKF_DA_it532')
    row_offs = ds.grid.tilegrids.loc['domain', 'j_offg']
    col_offs = ds.grid.tilegrids.loc['domain', 'i_offg']
    n_rows = ds.grid.tilegrids['N_lat'].domain
    n_cols = ds.grid.tilegrids['N_lon'].domain
    dim = (len(ds.grid.ease_lats), len(ds.grid.ease_lons))

    # 1. Iteration: Calculate mean and #obs
    length = np.zeros(dim)
    tmpsum_smap = np.zeros(dim)
    tmpsum_ldas = np.zeros(dim)
    for cnt, (f, date) in enumerate(zip(files, dates)):
        print(f'Iter 1: {cnt} / {len(files)}')
        with h5py.File(f, mode='r') as arr:
            qf = arr['Soil_Moisture_Retrieval_Data']['retrieval_qual_flag'][:]
            idx = np.where((qf == 0)|(qf == 8))
            row = arr['Soil_Moisture_Retrieval_Data']['EASE_row_index'][idx]
            col = arr['Soil_Moisture_Retrieval_Data']['EASE_column_index'][idx]
            tmpsum_smap[row, col] += arr['Soil_Moisture_Retrieval_Data']['soil_moisture'][idx]
            tmpsum_ldas[row_offs:row_offs+n_rows, col_offs:col_offs+n_cols] += ds.images['sm_surface'].sel(time=date).values
            length[row, col] += 1
    tmpsum_smap[tmpsum_smap==0] = np.nan
    tmpsum_ldas[tmpsum_ldas==0] = np.nan
    length[length==0] = np.nan

    mean_smap = tmpsum_smap / length
    mean_ldas = tmpsum_ldas / length

    # 2. Iteration: Calculate variances, covariance, and correlation coefficient
    tmpsum_smap = np.zeros(dim)
    tmpsum_ldas = np.zeros(dim)
    tmpsum = np.zeros(dim)
    for cnt, f in enumerate(files):
        print(f'Iter 2: {cnt} / {len(files)}')
        with h5py.File(f, mode='r') as arr:
            qf = arr['Soil_Moisture_Retrieval_Data']['retrieval_qual_flag'][:]
            idx = np.where((qf == 0)|(qf == 8))
            row = arr['Soil_Moisture_Retrieval_Data']['EASE_row_index'][idx]
            col = arr['Soil_Moisture_Retrieval_Data']['EASE_column_index'][idx]
            tmp_smap = np.zeros(dim)
            tmp_ldas = np.zeros(dim)
            tmp_smap[row, col] = arr['Soil_Moisture_Retrieval_Data']['soil_moisture'][idx] - mean_smap[row, col]
            tmp_ldas[row_offs:row_offs+n_rows, col_offs:col_offs+n_cols] = ds.images['sm_surface'].sel(time=date).values - mean_ldas[row_offs:row_offs+n_rows, col_offs:col_offs+n_cols]
            tmpsum_smap[row, col] += tmp_smap[row, col]**2
            tmpsum_ldas[row_offs:row_offs+n_rows, col_offs:col_offs+n_cols] += tmp_ldas[row_offs:row_offs+n_rows, col_offs:col_offs+n_cols]**2
            tmpsum[row, col] += (tmp_smap[row, col] * tmp_ldas[row, col])
    tmpsum_smap[tmpsum_smap==0] = np.nan
    tmpsum_ldas[tmpsum_ldas==0] = np.nan
    tmpsum[tmpsum==0] = np.nan

    var_smap = tmpsum_smap / (length-1)
    var_ldas = tmpsum_ldas / (length-1)
    cov = tmpsum / (length-1)

    # Derive validation metrics
    bias = mean_smap - mean_ldas
    corr = cov / np.sqrt(var_smap * var_ldas)

    root = Path('/Users/u0116961/Documents/work/LDAS/spatial_eval')
    np.save(root / 'mean_smap', mean_smap)
    np.save(root / 'mean_ldas', mean_ldas)
    np.save(root / 'var_smap', var_smap)
    np.save(root / 'var_ldas', var_ldas)
    np.save(root / 'cov', cov)
    np.save(root / 'bias', bias)
    np.save(root / 'corr', corr)


if __name__=='__main__':

    # calc_metrics()

    root = Path('/Users/u0116961/Documents/work/LDAS/spatial_eval')
    img = np.load(root / 'mean_smap.npy')
    plot_ease_img(img)
    plt.tight_layout()
    plt.show()
