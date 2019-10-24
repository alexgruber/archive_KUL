

import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from scipy.ndimage import gaussian_filter

from pygleam_ag.grid import read_grid, get_valid_gpis

def plot_image(data,tag,
                llcrnrlat=24,
                urcrnrlat=51,
                llcrnrlon=-128,
                urcrnrlon=-64,
                cbrange=(-20,20),
                cmap='jet',
                title='',
                fontsize=16):

    lats, lons, _ = read_grid()

    ind = np.unravel_index(data.index.values, lons.shape)

    img = np.full(lons.shape, np.nan)
    img[ind] = data[tag]
    img_masked = np.ma.masked_invalid(img)

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
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="5%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    if title == '':
        title = tag.lower()
    plt.title(title,fontsize=fontsize)


def plot_error_stats(outpath, gapfilled=True):

    if gapfilled is True:
        fname = '/work/GLEAM/errors/result_gapfilled_sig07.csv'
        sufix = '_gapfilled_sig07.png'
    else:
        fname = '/work/GLEAM/errors/result.csv'
        sufix = '.png'

    res = pd.read_csv(fname, index_col=0)

    # --------------------------------------------------------------------------------------------
    # plot sample size

    # cbrange = [0,100]
    # plt.figure(figsize=(12,7))
    # plot_image(res,'n', cbrange=cbrange)
    # plt.tight_layout()
    # plt.savefig(outpath / 'sample_size.png')
    # plt.close()


    # --------------------------------------------------------------------------------------------
    # plot correlations
    
    # cbrange = [0,1]
    #
    # plt.figure(figsize=(24,10))
    #
    # plt.subplot(2,3,1)
    # plot_image(res,'R_GLEAM_ASCAT', cbrange=cbrange)
    # plt.subplot(2,3,2)
    # plot_image(res,'R_GLEAM_AMSR2', cbrange=cbrange)
    # plt.subplot(2,3,3)
    # plot_image(res,'R_GLEAM_SMAP', cbrange=cbrange)
    # plt.subplot(2,3,4)
    # plot_image(res,'R_ASCAT_AMSR2', cbrange=cbrange)
    # plt.subplot(2,3,5)
    # plot_image(res,'R_ASCAT_SMAP', cbrange=cbrange)
    # plt.subplot(2,3,6)
    # plot_image(res,'R_AMSR2_SMAP', cbrange=cbrange)
    #
    # plt.tight_layout()
    # plt.savefig(outpath / 'correlation.png')
    # plt.close()


    # --------------------------------------------------------------------------------------------

    if gapfilled is False:
        r_min = 0.1

        ind = (res['R_GLEAM_ASCAT'] <= r_min) | \
              (res['R_GLEAM_AMSR2'] <= r_min) | \
              (res['R_ASCAT_AMSR2'] <= r_min)

        res.loc[ind,['TC1_R2_GLEAM', 'TC1_R2_ASCAT', 'TC1_R2_AMSR2']] = np.nan
        res.loc[ind,['TC1_RMSE_GLEAM', 'TC1_RMSE_ASCAT', 'TC1_RMSE_AMSR2']] = np.nan

        # ind = (res['p_GLEAM_ASCAT'] >= 0.05) | \
        #       (res['p_GLEAM_AMSR2'] >= 0.05) | \
        #       (res['p_ASCAT_AMSR2'] >= 0.05)
        #
        # res.loc[ind, ['TC1_R2_GLEAM', 'TC1_R2_ASCAT', 'TC1_R2_AMSR2']] = np.nan
        # res.loc[ind, ['TC1_RMSE_GLEAM', 'TC1_RMSE_ASCAT', 'TC1_RMSE_AMSR2']] = np.nan

        ind = (res['R_GLEAM_ASCAT'] <= r_min) | \
              (res['R_GLEAM_SMAP'] <= r_min) | \
              (res['R_ASCAT_SMAP'] <= r_min)

        res.loc[ind, ['TC2_R2_GLEAM', 'TC2_R2_ASCAT', 'TC2_R2_SMAP']] = np.nan
        res.loc[ind, ['TC2_RMSE_GLEAM', 'TC2_RMSE_ASCAT', 'TC2_RMSE_SMAP']] = np.nan

        # ind = (res['p_GLEAM_ASCAT'] >= 0.05) | \
        #       (res['p_GLEAM_SMAP'] >= 0.05) | \
        #       (res['p_ASCAT_SMAP'] >= 0.05)
        #
        # res.loc[ind, ['TC2_R2_GLEAM', 'TC2_R2_ASCAT', 'TC2_R2_SMAP']] = np.nan
        # res.loc[ind, ['TC2_RMSE_GLEAM', 'TC2_RMSE_ASCAT', 'TC2_RMSE_SMAP']] = np.nan


    # --------------------------------------------------------------------------------------------
    # plot TCA correlations

    cbrange = [0, 1]

    plt.figure(figsize=(24, 10))

    plt.subplot(2, 3, 1)
    plot_image(res, 'TC1_R2_GLEAM', cbrange=cbrange)
    plt.subplot(2, 3, 2)
    plot_image(res, 'TC1_R2_ASCAT', cbrange=cbrange)
    plt.subplot(2, 3, 3)
    plot_image(res, 'TC1_R2_AMSR2', cbrange=cbrange)
    plt.subplot(2, 3, 4)
    plot_image(res, 'TC2_R2_GLEAM', cbrange=cbrange)
    plt.subplot(2, 3, 5)
    plot_image(res, 'TC2_R2_ASCAT', cbrange=cbrange)
    plt.subplot(2, 3, 6)
    plot_image(res, 'TC2_R2_SMAP', cbrange=cbrange)

    plt.tight_layout()
    plt.savefig(outpath / ('TCA_R2' + sufix))
    plt.close()

    # --------------------------------------------------------------------------------------------
    # plot TCA RMSEs

    cbrange = [0, 10]

    plt.figure(figsize=(24, 10))

    plt.subplot(2, 3, 1)
    plot_image(res, 'TC1_RMSE_GLEAM', cbrange=cbrange)
    plt.subplot(2, 3, 2)
    plot_image(res, 'TC1_RMSE_ASCAT', cbrange=cbrange)
    plt.subplot(2, 3, 3)
    plot_image(res, 'TC1_RMSE_AMSR2', cbrange=cbrange)
    plt.subplot(2, 3, 4)
    plot_image(res, 'TC2_RMSE_GLEAM', cbrange=cbrange)
    plt.subplot(2, 3, 5)
    plot_image(res, 'TC2_RMSE_ASCAT', cbrange=cbrange)
    plt.subplot(2, 3, 6)
    plot_image(res, 'TC2_RMSE_SMAP', cbrange=cbrange)

    plt.tight_layout()
    plt.savefig(outpath / ('TCA_RMSE' + sufix))
    plt.close()

def plot_pert_corr(outpath):

    fname = '/work/GLEAM/perturbation_correction/result.csv'
    sufix = '.png'

    res = pd.read_csv(fname, index_col=0)

    res['c_a_rel'] = res['c_a'] / res['a']
    res['c_b_rel'] = res['c_b'] / res['b']

    plt.figure(figsize=(15, 9))

    plt.subplot(2, 2, 1)
    plot_image(res, 'a_s', cbrange=[0,3])
    plt.subplot(2, 2, 2)
    plot_image(res, 'b_s', cbrange=[0.7,1.3])
    plt.subplot(2, 2, 3)
    plot_image(res, 'c_a_rel', cbrange=[0, 2])
    plt.subplot(2, 2, 4)
    plot_image(res, 'c_b_rel', cbrange=[0, 0.025])

    plt.tight_layout()
    plt.savefig(outpath / ('pert_corr' + sufix))
    plt.close()
    # plt.show()

def plot_gamma(outpath):

    res = pd.read_csv('/work/GLEAM/autocorrelation/result.csv', index_col=0)

    cbrange = [0.5,1]

    plt.figure(figsize=(12,7))

    plot_image(res,'gamma', cbrange=cbrange)

    plt.tight_layout()
    # plt.show()

    plt.savefig(outpath / 'gamma.png')
    plt.close()


def plot_test():

    gpis_valid = get_valid_gpis(latmin=24., latmax=51., lonmin=-128., lonmax=-64.)
    ind_valid = np.unravel_index(gpis_valid, (720, 1440))

    img = np.load('/work/GLEAM/test.npy')

    from scipy.ndimage import gaussian_filter

    img_s = gaussian_filter(img, sigma=0.7, truncate=1)
    # img_s = img


    res = pd.DataFrame({'test': img_s[ind_valid]}, index=gpis_valid)

    plt.figure(figsize=(12, 7))

    plot_image(res, 'test', cbrange=[0,1])

    plt.tight_layout()
    plt.show()


if __name__=='__main__':

    outpath = Path('/work/GLEAM/plots')
    if not outpath.exists():
        outpath.mkdir()

    # plot_gamma(outpath)
    # plot_error_stats(outpath, gapfilled=True)
    plot_pert_corr(outpath)
    # plot_test()



