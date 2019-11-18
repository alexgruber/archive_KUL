

import numpy as np
import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.basemap import Basemap

from scipy.ndimage import gaussian_filter

from pygleam_ag.grid import read_grid, get_valid_gpis
from pygleam_ag.GLEAM_model import GLEAM


def plot_power_function_illustration():

    x = np.linspace(0, 2, 1000)

    plt.figure(figsize=(18, 12))

    cc = [0.5, 1.0, 1.5]

    for i, b in enumerate(cc):

        plt.subplot(2, 3, i + 1)

        for a in np.arange(0.6, 1.6, 0.2):
            y = a * x ** b
            c = (a - 0.2) / 1.8
            plt.plot(x, y, color=[c, c, c], linewidth=2, label='a = %.1f' % a)
        plt.plot((0, 2), (0, 2), '--k', linewidth=1)
        plt.ylim((0, 2))
        plt.title('a * x$^{%.1f}$ ' % b)
        if i == 0:
            plt.legend(loc='upper left')

    for i, a in enumerate(cc):

        plt.subplot(2, 3, 3 + i + 1)

        for b in np.arange(0.6, 1.6, 0.2):
            y = a * x ** b
            c = (b - 0.2) / 1.8
            plt.plot(x, y, color=[c, c, c], linewidth=2, label='b = %.1f' % b)
        plt.plot((0, 2), (0, 2), '--k', linewidth=1)
        plt.ylim((0, 2))
        plt.title('%.1f * x$^b$ ' % a)
        if i == 0:
            plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def plot_image(data,tag,
                llcrnrlat=24,
                urcrnrlat=51,
                llcrnrlon=-128,
                urcrnrlon=-64,
                cbrange=(-20,20),
                cmap='jet',
                title='',
                normalize=False,
                absolute=False,
                fontsize=16):

    lats, lons, _ = read_grid()

    ind = np.unravel_index(data.index.values, lons.shape)

    img = np.full(lons.shape, np.nan)
    if absolute:
        img[ind] = np.abs(data[tag])
    else:
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

    if normalize:
        im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True, norm=colors.LogNorm())
    else:
        im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)
    im.set_clim(vmin=cbrange[0], vmax=cbrange[1])
    cb = m.colorbar(im, "bottom", size="7%", pad="5%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(fontsize)
    if title == '':
        if absolute:
            title = 'abs(' + tag.lower() + ')'
        else:
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

    i = 2

    res['c_a_rel'] = res['c_a%i'%i] / res['a%i'%i]
    res['c_b_rel'] = res['c_b%i'%i] / res['b%i'%i]

    plt.figure(figsize=(15, 9))

    plt.subplot(2, 2, 1)
    plot_image(res, 'a%i_s'%i, cbrange=[0,3])
    plt.subplot(2, 2, 2)
    plot_image(res, 'b%i_s'%i, cbrange=[0.7,1.3])
    plt.subplot(2, 2, 3)
    plot_image(res, 'c_a_rel', cbrange=[0, 2])
    plt.subplot(2, 2, 4)
    plot_image(res, 'c_b_rel', cbrange=[0, 0.025])

    plt.tight_layout()
    plt.savefig(outpath / ('pert_corr%i'%i + sufix))
    plt.close()
    # plt.show()

def plot_pert_corr_v2(outpath, smoothed=True):

    mod = '_s' if smoothed is True else ''

    fname = '/work/GLEAM/perturbation_correction_v2/result.csv'
    sufix = '.png'

    res = pd.read_csv(fname, index_col=0)

    cb_a = [0.05,10]
    cb_b = [0.3,1.3]
    cb_c = [1e-6,1e-2]

    plt.figure(figsize=(22, 9))

    plt.subplot(2, 3, 1)
    plot_image(res, 'a1' + mod, cbrange=cb_a, normalize=True)
    plt.subplot(2, 3, 2)
    plot_image(res, 'b1' + mod, cbrange=cb_b)
    plt.subplot(2, 3, 3)
    plot_image(res, 'c1' + mod, cbrange=cb_c, normalize=True, absolute=True)
    plt.subplot(2, 3, 4)
    plot_image(res, 'a2' + mod, cbrange=cb_a, normalize=True)
    plt.subplot(2, 3, 5)
    plot_image(res, 'b2' + mod, cbrange=cb_b)
    plt.subplot(2, 3, 6)
    plot_image(res, 'c2' + mod, cbrange=cb_c, normalize=True, absolute=True)

    plt.tight_layout()
    plt.savefig(outpath / ('pert_corr_v2' + mod + sufix))
    plt.close()
    # plt.show()

def plot_pert_corr_test(outpath):

    path = Path('/work/GLEAM/perturbation_correction_test_v5')
    files = path.glob('*.csv')

    sufix = '.png'

    for i, fname in enumerate(np.sort(list(files))):

        res = pd.read_csv(fname, index_col=0)

        res['diff_rel'] = (res['avg_ens_var'] - res['pert']) / res['pert']

        plt.figure(figsize=(15, 9))

        cbrange = [0,0.005]

        plt.subplot(2, 2, 1)
        plot_image(res, 'pert', cbrange=cbrange)
        plt.subplot(2, 2, 2)
        plot_image(res, 'pert_corr', cbrange=cbrange)
        plt.subplot(2, 2, 3)
        plot_image(res, 'avg_ens_var', cbrange=cbrange)
        plt.subplot(2, 2, 4)
        plot_image(res, 'diff_rel', cbrange=[-0.4, 0.4])

        plt.tight_layout()
        plt.savefig(outpath / ('pert_corr_test_v%i' % (i+7) + sufix))
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


def plot_ts(gpi):

    pert = pd.read_csv('/work/GLEAM/errors/result_gapfilled_sig07.csv', index_col=0)['TC2_RMSE_GLEAM'] ** 2 / 100**2
    corr = pd.read_csv('/work/GLEAM/perturbation_correction/result.csv', index_col=0)
    pert_corr = (pert / corr['a1_s']) ** (1 / corr['b1_s'])

    errvar = pert_corr.loc[gpi]

    params = {'nens': 50}
    gleam = GLEAM(params)

    params = {'nens': 1}
    gleam_det = GLEAM(params)

    gleam.mod_pert = {'w1': ['normal', 'additive', errvar]}

    var = 'w1'

    res = gleam.proc_ol(gpi)[var]
    res_det = gleam_det.proc_ol(gpi)[var]

    # print(pert.loc[gpi], pert_corr.loc[gpi], res.var(axis=1).mean())

    plt.figure(figsize=(12, 7))

    pd.DataFrame(res).plot(ax=plt.gca(),linewidth=0.5, linestyle= '--', legend=False)
    pd.Series(res.mean(axis=1)).plot(ax=plt.gca(),linewidth=3)
    pd.Series(res_det[:,0]).plot(ax=plt.gca(),linewidth=3)

    r = np.corrcoef(res.mean(axis=1),res_det[:,0])[0,1]
    b = res.mean(axis=1).mean() - res_det[:,0].mean()

    print(r, b)

    plt.title(var)

    plt.ylim(0.05,0.45)

    plt.tight_layout()
    plt.show()


if __name__=='__main__':

    outpath = Path('/work/GLEAM/plots')
    if not outpath.exists():
        outpath.mkdir()

    # plot_gamma(outpath)
    # plot_error_stats(outpath, gapfilled=True)
    # plot_pert_corr_v2(outpath)
    plot_pert_corr_test(outpath)

    # TODO  AT THIS GRID CELL, THE ERROR VARIANCE STARTS BLOWING UP, PROBABLY BECAUSE OF THE TREATMENT OF PROLONGED
    # TODO: DRY PERIODS
    # lat = 35.062078
    # lon = -117.258583

    # lat = 41.299531
    # lon = -117.400013

    # from pygleam_ag.grid import find_nearest_gpi
    #
    # gpi = find_nearest_gpi(lat, lon)
    #
    # plot_ts(gpi)



    # plot_power_function_illustration()