
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from netCDF4 import Dataset

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
sns.set_context('talk', font_scale=0.4)
import colorcet as cc

from pytesmo.time_series.anomaly import calc_climatology
from myprojects.timeseries import calc_anom

from pyldas.interface import GEOSldas_io, LDASsa_io
from pyldas.templates import template_error_Tb40

from validation_good_practice.ancillary.grid import Paths
from validation_good_practice.ancillary.grid import EASE2

from validation_good_practice.plots import plot_ease_img
from myprojects.experiments.MadKF.CLSM.ensemble_covariance import plot_ease_img2
from myprojects.experiments.MadKF.CLSM.plots import plot_image
from myprojects.experiments.MadKF.CLSM.validate_madkf_smap import plot_centered_cbar


def plot_latlon_img(img, lons, lats,
                  llcrnrlat=24,
                  urcrnrlat=51,
                  llcrnrlon=-128,
                  urcrnrlon=-64,
                  cbrange=None,
                  cmap='jet_r',
                  title='',
                  fontsize=20,
                  plot_cb=False,
                  print_median=False,
                  log_scale=False):

    grid = EASE2()

    lons,lats = np.meshgrid(lons, lats)

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

    if log_scale:
        im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, norm=LogNorm(), latlon=True)
    else:
        im = m.pcolormesh(lons, lats, img_masked, cmap=cmap, latlon=True)

    if cbrange is not None:
        im.set_clim(vmin=cbrange[0], vmax=cbrange[1])

    if plot_cb is True:

        ticks = np.arange(cbrange[0],cbrange[1]+0.001, (cbrange[1]-cbrange[0])/4)
        # ticks = np.arange(9)
        # cb = m.colorbar(im, "bottom", size="8%", pad="4%", ticks=ticks)
        cb = m.colorbar(im, "bottom", size="7%", pad="5%")

        for t in cb.ax.get_xticklabels():
            t.set_fontsize(fontsize-2)

    plt.title(title,fontsize=fontsize)

    if print_median is True:
        x, y = m(-79, 25)
        plt.text(x, y, 'm. = %.3f' % np.ma.median(img_masked), fontsize=fontsize-4)

    return im


def plot_predicted_skillgain(dir_out):

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    R_anom_lst = res[f'ubrmse_grid_anom_lst_m_SMAP_tc_ASCAT_SMAP_CLSM']**2
    P_anom_lst = res[f'ubrmse_grid_anom_lst_m_CLSM_tc_ASCAT_SMAP_CLSM']**2

    R_anom_lt = res[f'ubrmse_grid_anom_lt_m_SMAP_tc_ASCAT_SMAP_CLSM']**2
    P_anom_lt = res[f'ubrmse_grid_anom_lt_m_CLSM_tc_ASCAT_SMAP_CLSM']**2

    R_anom_st = res[f'ubrmse_grid_anom_st_m_SMAP_tc_ASCAT_SMAP_CLSM']**2
    P_anom_st = res[f'ubrmse_grid_anom_st_m_CLSM_tc_ASCAT_SMAP_CLSM']**2

    R_lt_st_arr = (R_anom_lst - (R_anom_lt + R_anom_st)) / 2
    P_lt_st_arr = (P_anom_lst - (P_anom_lt + P_anom_st)) / 2

    P_lt_st_arr_rho = P_lt_st_arr / np.sqrt(P_anom_lt * P_anom_st)

    # Baseline estimates
    # R2 = res[f'r2_grid_abs_m_CLSM_tc_ASCAT_SMAP_CLSM']
    R2 = res[f'r2_grid_anom_lst_m_CLSM_tc_ASCAT_SMAP_CLSM']
    SNR = R2 / (1 - R2)
    SIG = SNR * P_anom_lst


    modes = ['anom_lt', 'anom_st', 'anom_lst', 'anom_lst']
    titles = ['LF signal', 'HF signal', 'Anomalies (lumped)', 'Anomalies (joint)']
    result = pd.DataFrame(index=res.index, columns=modes)
    result['row'] = res.row
    result['col'] = res.col

    for i, mode in enumerate(modes):

        P = res[f'ubrmse_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM'] ** 2
        R = res[f'ubrmse_grid_{mode}_m_SMAP_tc_ASCAT_SMAP_CLSM'] ** 2

        K = P / (R + P)

        if i < 3:
            K_4K = P / (4**2 + P)

            P_upd = K * R + (1-K) * P
            NSR_upd = P_upd / SIG
            R2upd = 1 / (1 + NSR_upd)

            result[f'{i}_4K'] = np.sqrt(R2upd) - np.sqrt(R2)

        if i < 2:
            # Single signal assimilation
            # R2 = res[f'r2_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM']
            # NSR = (1 - R2) / R2
            # R2upd = 1 / (1 + (1 - K) * NSR)

            P_upd = (1 - K) * P
            NSR_upd = P_upd / SIG
            R2upd = 1 / (1 + NSR_upd)

            result[i] = np.sqrt(R2upd) - np.sqrt(R2)
            result[f'P_upd_{i}'] = P_upd

        elif i == 2:
            # Joint signal assimilation
            result[i] = np.nan

            for cnt, idx in enumerate(res.index):
                print(f'{cnt} / {len(res)}')

                R11 = R_anom_lt.loc[idx]
                R22 = R_anom_st.loc[idx]
                R12 = R_lt_st_arr.loc[idx]

                P11 = P_anom_lt.loc[idx]
                P22 = P_anom_st.loc[idx]
                P12 = P_lt_st_arr.loc[idx]

                S = np.matrix([[R11, R12, 0, 0],
                               [R12, R22, 0, 0],
                               [0, 0, P11, P12],
                               [0, 0, P12, P22]])

                A = np.matrix([K.loc[idx], K.loc[idx], 1-K.loc[idx], 1-K.loc[idx]])
                P_upd = (A * S * A.T)[0,0]
                NSR_upd = P_upd / SIG.loc[idx]
                R2upd = 1 / (1 + NSR_upd)

                result.loc[idx, i] = np.sqrt(R2upd) - np.sqrt(R2.loc[idx])

        else:
            P_upd = result[f'P_upd_0'] + result[f'P_upd_1'] + 2 * P_lt_st_arr_rho * np.sqrt(result[f'P_upd_0'] * result[f'P_upd_1'])
            NSR_upd = P_upd / SIG
            R2upd = 1 / (1 + NSR_upd)
            result[i] = np.sqrt(R2upd) - np.sqrt(R2)

    f = plt.figure(figsize=(23, 7))

    for i, title in enumerate(titles):
        plt.subplot(2, 4, i+1)
        im = plot_ease_img(result, i, fontsize=12, cbrange=[-0.2,0.2], cmap=cc.cm.bjy, log_scale=False, title=title, plot_cb=False)

        if i < 3:
            plt.subplot(2, 4, i + 5)
            im = plot_ease_img(result, f'{i}_4K', fontsize=12, cbrange=[-0.2, 0.2], cmap=cc.cm.bjy, log_scale=False, title=title + ' (4K)', plot_cb=False)

    plot_centered_cbar(f, im, 3, fontsize=10)

    plt.tight_layout()
    plt.show()

    # fout = dir_out / 'skillgain_pot.png'
    # f.savefig(fout, dpi=300, bbox_inches='tight')
    # plt.close()


def plot_ascat_eval_absolute(res_path, dir_out):

    runs = ['Pcorr_OL', 'Pcorr_4K', ['Pcorr_anom_lst', 'Pcorr_anom_lt_ScDY', 'Pcorr_anom_st_ScYH'], 'Pcorr_LTST']
    # titles = ['Open-loop', '4K Benchmark', 'Individual assimilation', 'Joint assimilation']
    titles = ['$OL$', '$CTRL$', ['$DA_{anom}$','$DA_{LF}$','$DA_{HF}$'], '$DA_{joint}$']

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)
    res_tc = pd.read_csv(r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv', index_col=0)

    modes = ['anom_lst', 'anom_lt', 'anom_st']

    cb_r = [0.6, 1]

    fontsize = 17

    f = plt.figure(figsize=(25,17))

    for cnt, (run, tit) in enumerate(zip(runs,titles)):

        for i, m in enumerate(modes):

            r = run if cnt != 2 else run[i]

            ind = cnt * 3 + i + 1
            label = chr(96+ind) + ')'

            ax = plt.subplot(4, 3, ind)

            col = f'ana_r_corr_{r}_{m}'

            res[col][res[col] < 0] = 0
            res[col][res[col] > 1] = 1

            if isinstance(tit, list):
                ylabel = tit[i]
            else:
                ylabel = tit

            # if cnt == 0:
            if i == 0:
                title = r'$\rho_{anom}$' + f' ({ylabel})'
            elif i == 1:
                title = r'$\rho_{LF}$' + f' ({ylabel})'
            else:
                title = r'$\rho_{HF}$' + f' ({ylabel})'
            # else:
            #     title = ''

            r_asc_smap = res_tc[f'r_grid_{m}_p_ASCAT_SMAP']
            r_asc_clsm = res_tc[f'r_grid_{m}_p_ASCAT_CLSM']
            r_smap_clsm = res_tc[f'r_grid_{m}_p_SMAP_CLSM']
            thres = 0.2
            ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

            im = plot_ease_img(res.reindex(ind_valid), col , title=title, cmap='viridis', cbrange=cb_r, fontsize=fontsize,
                               print_median=True,
                               plot_cb=False,
                               plot_label=label)

            # ax.set_ylabel(ylabel, fontsize=fontsize)

        # plot_centered_cbar(f, im, 4, fontsize=fontsize-2, pad=0.02)
    plot_centered_cbar(f, im, 3, fontsize=fontsize - 2, bottom=0.05, wdth=0.02)

    f.savefig(dir_out / f'ascat_eval_abs.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ascat_eval_relative(res_path, dir_out):

    refs = ['Pcorr_OL', 'Pcorr_4K']
    ref_labels = ['$OL$', '$CTRL$']

    runs = [['Pcorr_anom_lst', 'Pcorr_anom_lt_ScDY', 'Pcorr_anom_st_ScYH'], 'Pcorr_LTST']
    run_labels = [['$DA_{anom}$','$DA_{LF}$','$DA_{HF}$'], '$DA_{joint}$']

    # titles = ['R$_{TC}$ - R$_{OL}$ (Individual)', 'R$_{TC}$ - R$_{OL}$ (Joint)', 'R$_{TC}$ - R$_{4K}$ (Individual)', 'R$_{TC}$ - R$_{4K}$ (Joint)']
    # titles = ['$OL$', '$CTRL$', ['$DA_{anom}$','$DA_{LF}$','$DA_{HF}$'], '$DA_{joint}$']

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)

    res_tc = pd.read_csv(r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv', index_col=0)

    modes = ['anom_lst', 'anom_lt','anom_st']
    labels = [r'$\Delta \rho_{anom}$', r'$\Delta \rho_{LF}$', r'$\Delta \rho_{HF}$']

    cb_r = [-0.25, 0.25]

    fontsize = 17

    f = plt.figure(figsize=(25,17))

    for j, (ref, ref_label) in enumerate(zip(refs, ref_labels)):

        for k, (run, run_label) in enumerate(zip(runs,run_labels)):

            for i, (m, label) in enumerate(zip(modes,labels)):

                r = run if k == 1 else run[i]
                rl = run_label if k == 1 else run_label[i]

                ref_col = f'ana_r_corr_{ref}_{m}'
                res[ref_col][res[ref_col] < 0] = 0
                res[ref_col][res[ref_col] > 1] = 1

                ind = j * 6 + k * 3 + i + 1
                figlabel = chr(96 + ind) + ')'
                ax = plt.subplot(4, 3, ind)

                col = f'ana_r_corr_{r}_{m}'

                res[col][res[col] < 0] = 0
                res[col][res[col] > 1] = 1

                res['diff'] = res[col] - res[ref_col]

                title = f'{label} ({rl} - {ref_label})'

                r_asc_smap = res_tc[f'r_grid_{m}_p_ASCAT_SMAP']
                r_asc_clsm = res_tc[f'r_grid_{m}_p_ASCAT_CLSM']
                r_smap_clsm = res_tc[f'r_grid_{m}_p_SMAP_CLSM']
                thres = 0.2
                ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

                im = plot_ease_img(res.reindex(ind_valid), 'diff' , title=title, cmap=cc.cm.coolwarm_r, cbrange=cb_r, fontsize=fontsize,
                                   print_median=True,
                                   plot_cb=False,
                                   plot_label=figlabel)

        # plot_centered_cbar(f, im, 4, fontsize=fontsize-2, pad=0.02)
        plot_centered_cbar(f, im, 3, fontsize=fontsize - 2, bottom=0.05, wdth=0.02)

    f.savefig(dir_out / f'ascat_eval_rel.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_statistics(res_path, dir_out):

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)
    res_ismn = pd.read_csv(res_path / 'insitu_TCA.csv', index_col=0)
    networks  = ['SCAN', 'USCRN']
    res_ismn = res_ismn.loc[res_ismn.network.isin(networks),:]
    res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    tg = GEOSldas_io().grid.tilegrids
    ind_ismn = []
    for col, row in zip(res_ismn.ease_col.values + tg.loc['domain','i_offg'], res_ismn.ease_row.values + tg.loc['domain','j_offg']):
        try:
            ind_ismn += [res[(res.row == row) & (res.col == col)].index.values[0]]
        except:
            continue

    runs = [f'Pcorr_{err}' for err in ['anom_lst', 'anom_lt_ScDY', 'anom_st_ScYH']]
    modes = ['anom_lst', 'anom_lt', 'anom_st']
    titles = [f'Anomaly skill', f'LF skill', f'HF skill']

    f = plt.figure(figsize=(24,15))

    output = 'ascat' # 'ascat' or 'ascat_ismn' or 'ismn'
    met = 'r_corr' # 'R_model_insitu'# or 'R2_model' / 'r' or 'r_corr'
    var = 'sm_surface'

    if 'ubRMSD' in met:
        lim = [-0.015,0.015]
        xloc = -0.0135
        bins = 15
    else:
        if 'ascat' not in output:
            lim = [-0.2,0.3]
            xloc = -0.18
            bins = 15
        else:
            lim = [-0.25,0.25]
            xloc = -0.22
            bins = 20

    f.suptitle(f'{output} / {met} / {var}')
    for i, (run, mode, title) in enumerate(zip(runs, modes, titles)):

            if 'ascat' in output:
                col_ol = f'ana_{met}_Pcorr_OL_{mode}'
                col_4k = f'ana_{met}_Pcorr_4K_{mode}'
                col_lst = f'ana_{met}_Pcorr_anom_lst_{mode}'
                col_da2 = f'ana_{met}_Pcorr_LTST_{mode}'
                col_da1 = f'ana_{met}_{run}_{mode}'
                if 'ismn' in output:
                    res = res.reindex(ind_ismn)
                else:
                    r_asc_smap = res_tc[f'r_grid_{mode}_p_ASCAT_SMAP']
                    r_asc_clsm = res_tc[f'r_grid_{mode}_p_ASCAT_CLSM']
                    r_smap_clsm = res_tc[f'r_grid_{mode}_p_SMAP_CLSM']
                    thres = 0.2
                    ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index
                    res = res.reindex(ind_valid)
            else:
                res = res_ismn
                col_ol = f'{met}_Pcorr_OL_{mode}_{var}'
                col_4k = f'{met}_Pcorr_4K_{mode}_{var}'
                col_lst = f'{met}_Pcorr_anom_lst_{mode}_{var}'
                col_da2 = f'{met}_Pcorr_LTST_{mode}_{var}'
                col_da1 = f'{met}_{run}_{mode}_{var}'
                if 'R2' in met:
                    res[col_ol] **= 0.5
                    res[col_4k] **= 0.5
                    res[col_lst] **= 0.5
                    res[col_da2] **= 0.5
                    res[col_da1] **= 0.5

            if not ((output=='ismn') and ('R2' not in met)):
                print('filtered')
                res[col_ol][res[col_ol] <= 0] = np.nan
                res[col_4k][res[col_4k] <= 0] = np.nan
                res[col_da1][res[col_da1] <= 0] = np.nan
                res[col_da2][res[col_da2] <= 0] = np.nan
                res[col_lst][res[col_lst] <= 0] = np.nan
                res[col_ol][res[col_ol] >= 1] = np.nan
                res[col_4k][res[col_4k] >= 1] = np.nan
                res[col_da1][res[col_da1] >= 1] = np.nan
                res[col_da2][res[col_da2] >= 1] = np.nan
                res[col_lst][res[col_lst] >= 1] = np.nan

            # res['single1'] = res[col_da1] - res[col_ol]
            # res['single2'] = res[col_da1] - res[col_4k]
            # res['joint1'] = res[col_da2] - res[col_ol]
            # res['joint2'] = res[col_da2] - res[col_4k]

            res['single'] = res[col_da1] - res[col_ol]
            res['joint'] = res[col_da2] - res[col_ol]
            res['anom'] = res[col_lst] - res[col_ol]
            res['4k'] = res[col_4k] - res[col_ol]


            ax = plt.subplot(3, 3, i+1)
            # plt.scatter(x=res[col_ol].reindex(ind_valid),y=res[col_da1].reindex(ind_valid))
            p1 = res['4k'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.8)
            p2 = res['anom'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.6)
            plt.title(title)
            plt.yticks(color='w', fontsize=5)
            plt.xticks(color='w', fontsize=1)
            if i == 0:
                plt.ylabel('Anomaly assimilation')
            if i==2:
                # plt.legend(labels=['R$_{TC}$ - R$_{4K}$', 'R$_{TC}$ - R$_{OL}$'], loc='upper right', fontsize=14)
                plt.legend(labels=['R$_{4K}$ - R$_{OL}$', 'R$_{TC}$ - R$_{OL}$'], loc='upper right', fontsize=14)
            plt.axvline(color='black', linestyle='--', linewidth=1)
            plt.xlim(lim)
            ylim = ax.get_ylim()
            yloc1 = ylim[1] - (ylim[1] - ylim[0])/10
            yloc2 = ylim[1] - 1.6*(ylim[1] - ylim[0])/10
            plt.text(xloc, yloc1, 'mean = %.2f' % res['4k'].mean(), color='#1f77b4')
            plt.text(xloc, yloc2, 'mean = %.2f' % res['anom'].mean(), color='#ff7f0e')

            ax = plt.subplot(3, 3, i+4)
            res['4k'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.8)
            res['single'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.6)
            plt.yticks(color='w', fontsize=5)
            plt.xticks(color='w', fontsize=1)
            if i == 0:
                plt.ylabel('Individual assimilation')
            plt.xlim(lim)
            plt.axvline(color='black', linestyle='--', linewidth=1)
            ylim = ax.get_ylim()
            yloc1 = ylim[1] - (ylim[1] - ylim[0])/10
            yloc2 = ylim[1] - 1.6*(ylim[1] - ylim[0])/10
            plt.text(xloc, yloc1, 'mean = %.2f' % res['4k'].mean(), color='#1f77b4')
            plt.text(xloc, yloc2, 'mean = %.2f' % res['single'].mean(), color='#ff7f0e')

            ax = plt.subplot(3, 3, i+7)
            res['4k'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.8)
            res['joint'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.6)
            plt.yticks(color='w', fontsize=5)
            if i == 0:
                plt.ylabel('joint assimilation')
            plt.xlim(lim)
            plt.axvline(color='black', linestyle='--', linewidth=1)
            ylim = ax.get_ylim()
            yloc1 = ylim[1] - (ylim[1] - ylim[0])/10
            yloc2 = ylim[1] - 1.6*(ylim[1] - ylim[0])/10
            plt.text(xloc, yloc1, 'mean = %.2f' % res['4k'].mean(), color='#1f77b4')
            plt.text(xloc, yloc2, 'mean = %.2f' % res['joint'].mean(), color='#ff7f0e')

            if i == 0:
                plt.title(title)

    f.savefig(dir_out / f'stats_{output}_{met}_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ismn_statistics(res_path, dir_out):

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)
    res_ismn = pd.read_csv(res_path / 'insitu_TCA.csv', index_col=0)
    networks  = ['SCAN', 'USCRN']
    res_ismn = res_ismn.loc[res_ismn.network.isin(networks),:]
    res_tc = pd.read_csv(r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv',
        index_col=0)

    tg = GEOSldas_io().grid.tilegrids
    ind_ismn = []
    for col, row in zip(res_ismn.ease_col.values + tg.loc['domain','i_offg'], res_ismn.ease_row.values + tg.loc['domain','j_offg']):
        try:
            ind_ismn += [res[(res.row == row) & (res.col == col)].index.values[0]]
        except:
            continue

    refs = ['Pcorr_OL', 'Pcorr_4K']
    runs = [['Pcorr_anom_lst', 'Pcorr_anom_lt_ScDY', 'Pcorr_anom_st_ScYH'], 'Pcorr_LTST']
    run_labels = [['$DA_{anom}$', '$DA_{LF}$', '$DA_{HF}$'], '$DA_{joint}$']

    modes = ['anom_lst', 'anom_lt', 'anom_st']
    labels = [r'$\rho_{anom}$', r'$\rho_{LF}$', r'$\rho_{HF}$']

    f = plt.figure(figsize=(23, 20))
    fontsize = 20

    outputs = ['ismn', 'ismn']
    mets = ['R_model_insitu', 'R_model_insitu']
    variables = ['sm_surface', 'sm_rootzone']
    var_labels = ['SSM', 'RZSM']

    xres = res.copy()
    xres_ismn = res_ismn.copy()

    lim = [-0.25,0.25]
    xloc = 0.09
    xlocl = -0.23
    bins = 18

    for i, (output, met, var, var_label) in enumerate(zip(outputs, mets, variables, var_labels)):

        for j, (run, run_label) in enumerate(zip(runs,run_labels)):

            for k, (mode, label) in enumerate(zip(modes, labels)):

                r = run if j == 1 else run[k]
                rl = run_label if j == 1 else run_label[k]

                res = xres.copy()
                res_ismn  = xres_ismn.copy()

                res = res_ismn
                col_ol = f'{met}_Pcorr_OL_{mode}_{var}'
                col_4k = f'{met}_Pcorr_4K_{mode}_{var}'
                col_da = f'{met}_{r}_{mode}_{var}'
                if 'R2' in met:
                    res[col_ol] **= 0.5
                    res[col_4k] **= 0.5
                    res[col_da] **= 0.5

                if not ((output=='ismn') and ('R2' not in met)):
                    print('filtered')
                    res[col_ol][res[col_ol] <= 0] = np.nan
                    res[col_4k][res[col_4k] <= 0] = np.nan
                    res[col_da][res[col_da] <= 0] = np.nan
                    res[col_ol][res[col_ol] >= 1] = np.nan
                    res[col_4k][res[col_4k] >= 1] = np.nan
                    res[col_da][res[col_da] >= 1] = np.nan

                res['da'] = res[col_da] - res[col_ol]
                res['4k'] = res[col_4k] - res[col_ol]

                ind = i * 6 + j * 3 + k + 1
                fig_label = chr(96 + ind) + ')'
                ax = plt.subplot(4, 3, ind)

                p1 = res['4k'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.8)
                p2 = res['da'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.6)
                plt.yticks(color='w', fontsize=5)
                if ind <= 9:
                    plt.xticks(color='w', fontsize=fontsize)
                else:
                    plt.xticks(fontsize=fontsize-2)
                plt.title(f'$\Delta${label}; ISMN; {var_label}', fontsize=fontsize)
                leg_labels = [f'$CTRL$ - $OL$',
                              f'{rl} - $OL$']
                plt.legend(labels=leg_labels, loc='lower left', fontsize=fontsize-4)

                plt.axvline(color='black', linestyle='--', linewidth=1)
                plt.xlim(lim)
                ylim = ax.get_ylim()
                yloc1 = ylim[1] - (ylim[1] - ylim[0])/10
                yloc2 = ylim[1] - 1.8*(ylim[1] - ylim[0])/10
                yloc3 = ylim[1] - 1.2 * (ylim[1] - ylim[0]) / 10
                plt.text(xlocl, yloc3, fig_label, fontsize=fontsize)
                plt.text(xloc, yloc1, 'mean = %.2f' % res['4k'].mean(), color='#1f77b4', fontsize=fontsize-3)
                plt.text(xloc, yloc2, 'mean = %.2f' % res['da'].mean(), color='#ff7f0e', fontsize=fontsize-3)

    f.savefig(dir_out / f'stats_ismn.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ascat_statistics(res_path, dir_out):
    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)
    res_ismn = pd.read_csv(res_path / 'insitu_TCA.csv', index_col=0)
    networks = ['SCAN', 'USCRN']
    res_ismn = res_ismn.loc[res_ismn.network.isin(networks), :]
    res_tc = pd.read_csv(
        r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv',
        index_col=0)

    tg = GEOSldas_io().grid.tilegrids
    ind_ismn = []
    for col, row in zip(res_ismn.ease_col.values + tg.loc['domain', 'i_offg'],
                        res_ismn.ease_row.values + tg.loc['domain', 'j_offg']):
        try:
            ind_ismn += [res[(res.row == row) & (res.col == col)].index.values[0]]
        except:
            continue

    refs = ['Pcorr_OL', 'Pcorr_4K']
    runs = [['Pcorr_anom_lst', 'Pcorr_anom_lt_ScDY', 'Pcorr_anom_st_ScYH'], 'Pcorr_LTST']
    run_labels = [['$DA_{anom}$', '$DA_{LF}$', '$DA_{HF}$'], '$DA_{joint}$']

    modes = ['anom_lst', 'anom_lt', 'anom_st']
    labels = [r'$\rho_{anom}$', r'$\rho_{LF}$', r'$\rho_{HF}$']

    f = plt.figure(figsize=(23, 12))
    fontsize = 20

    outputs = ['ascat_ismn',]
    mets = ['r_corr',]
    variables = ['sm_surface',]

    var_labels = ['SSM',]

    xres = res.copy()
    xres_ismn = res_ismn.copy()

    lim = [-0.25, 0.25]
    xloc = 0.09
    xlocl = -0.23
    bins = 18

    for i, (output, met, var, var_label) in enumerate(zip(outputs, mets, variables, var_labels)):

        for j, (run, run_label) in enumerate(zip(runs, run_labels)):

            for k, (mode, label) in enumerate(zip(modes, labels)):

                r = run if j == 1 else run[k]
                rl = run_label if j == 1 else run_label[k]

                res = xres.copy()
                res_ismn = xres_ismn.copy()

                col_ol = f'ana_{met}_Pcorr_OL_{mode}'
                col_4k = f'ana_{met}_Pcorr_4K_{mode}'
                col_da = f'ana_{met}_{r}_{mode}'
                if 'ismn' in output:
                    res = res.reindex(ind_ismn)
                else:
                    r_asc_smap = res_tc[f'r_grid_{mode}_p_ASCAT_SMAP']
                    r_asc_clsm = res_tc[f'r_grid_{mode}_p_ASCAT_CLSM']
                    r_smap_clsm = res_tc[f'r_grid_{mode}_p_SMAP_CLSM']
                    thres = 0.2
                    ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index
                    res = res.reindex(ind_valid)

                if not ((output == 'ismn') and ('R2' not in met)):
                    print('filtered')
                    res[col_ol][res[col_ol] <= 0] = np.nan
                    res[col_4k][res[col_4k] <= 0] = np.nan
                    res[col_da][res[col_da] <= 0] = np.nan
                    res[col_ol][res[col_ol] >= 1] = np.nan
                    res[col_4k][res[col_4k] >= 1] = np.nan
                    res[col_da][res[col_da] >= 1] = np.nan

                res['da'] = res[col_da] - res[col_ol]
                res['4k'] = res[col_4k] - res[col_ol]

                ind = i * 6 + j * 3 + k + 1
                fig_label = chr(96 + ind) + ')'
                ax = plt.subplot(2, 3, ind)

                p1 = res['4k'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.8)
                p2 = res['da'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.6)
                plt.yticks(color='w', fontsize=5)
                if ind <= 3:
                    plt.xticks(color='w', fontsize=fontsize)
                else:
                    plt.xticks(fontsize=fontsize - 2)
                plt.title(f'$\Delta${label}; ASCAT; {var_label}', fontsize=fontsize)
                leg_labels = [f'$CTRL$ - $OL$',
                              f'{rl} - $OL$']
                plt.legend(labels=leg_labels, loc='lower left', fontsize=fontsize - 4)

                plt.axvline(color='black', linestyle='--', linewidth=1)
                plt.xlim(lim)
                ylim = ax.get_ylim()
                yloc1 = ylim[1] - (ylim[1] - ylim[0]) / 10
                yloc2 = ylim[1] - 1.7 * (ylim[1] - ylim[0]) / 10
                yloc3 = ylim[1] - 1.0 * (ylim[1] - ylim[0]) / 10
                plt.text(xlocl, yloc3, fig_label, fontsize=fontsize)
                plt.text(xloc, yloc1, 'mean = %.2f' % res['4k'].mean(), color='#1f77b4', fontsize=fontsize - 3)
                plt.text(xloc, yloc2, 'mean = %.2f' % res['da'].mean(), color='#ff7f0e', fontsize=fontsize - 3)

    f.savefig(dir_out / f'stats_ascat.png', dpi=300, bbox_inches='tight')
    plt.close()

    # res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)
    # res_ismn = pd.read_csv(res_path / 'insitu_TCA.csv', index_col=0)
    # networks  = ['SCAN', 'USCRN']
    # res_ismn = res_ismn.loc[res_ismn.network.isin(networks),:]
    # res_tc = pd.read_csv(r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv',
    #     index_col=0)
    #
    # tg = GEOSldas_io().grid.tilegrids
    # ind_ismn = []
    # for col, row in zip(res_ismn.ease_col.values + tg.loc['domain','i_offg'], res_ismn.ease_row.values + tg.loc['domain','j_offg']):
    #     try:
    #         ind_ismn += [res[(res.row == row) & (res.col == col)].index.values[0]]
    #     except:
    #         continue
    #
    # refs = ['Pcorr_OL', 'Pcorr_4K']
    # runs = [['Pcorr_anom_lst', 'Pcorr_anom_lt_ScDY', 'Pcorr_anom_st_ScYH'], 'Pcorr_LTST']
    #
    # titles = ['Individual assim. (ASCAT, SSM)',
    #           'Joint assim. (ASCAT, SSM)',
    #           'Individual assim. (ISMN, SSM)',
    #           'Joint assim. (ISMN, SSM)',
    #           'Individual assim. (ISMN, RZSM)',
    #           'Joint assim. (ISMN, RZSM)',
    #           ]
    #
    # modes = ['anom_lst', 'anom_lt', 'anom_st']
    # labels = ['$R_{anom}$ ', '$R_{LF}$ ', '$R_{HF}$ ']
    #
    # f = plt.figure(figsize=(25, 19))
    # fontsize = 17
    #
    # outputs = ['ascat_ismn', 'ismn', 'ismn']
    # mets = ['r_corr', 'R_model_insitu', 'R_model_insitu']
    # variables = ['sm_surface', 'sm_surface', 'sm_rootzone']
    #
    # xres = res.copy()
    # xres_ismn = res_ismn.copy()
    #
    # for i, (output, met, var) in enumerate(zip(outputs, mets, variables)):
    #     if 'ascat' not in output:
    #         lim = [-0.2,0.3]
    #         xloc = -0.18
    #         bins = 15
    #     else:
    #         lim = [-0.25,0.25]
    #         xloc = -0.22
    #         bins = 20
    #
    #     for j, run in enumerate(runs):
    #         r = run if j == 1 else run[i]
    #
    #         for k, (mode, label) in enumerate(zip(modes, labels)):
    #
    #             res = xres.copy()
    #             res_ismn  = xres_ismn.copy()
    #
    #             title = titles[j * 2 + k] if i == 0 else ''
    #
    #             if 'ascat' in output:
    #                 col_ol = f'ana_{met}_Pcorr_OL_{mode}'
    #                 col_4k = f'ana_{met}_Pcorr_4K_{mode}'
    #                 col_da = f'ana_{met}_{r}_{mode}'
    #                 if 'ismn' in output:
    #                     res = res.reindex(ind_ismn)
    #                 else:
    #                     r_asc_smap = res_tc[f'r_grid_{mode}_p_ASCAT_SMAP']
    #                     r_asc_clsm = res_tc[f'r_grid_{mode}_p_ASCAT_CLSM']
    #                     r_smap_clsm = res_tc[f'r_grid_{mode}_p_SMAP_CLSM']
    #                     thres = 0.2
    #                     ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index
    #                     res = res.reindex(ind_valid)
    #             else:
    #                 res = res_ismn
    #                 col_ol = f'{met}_Pcorr_OL_{mode}_{var}'
    #                 col_4k = f'{met}_Pcorr_4K_{mode}_{var}'
    #                 col_da = f'{met}_{r}_{mode}_{var}'
    #                 if 'R2' in met:
    #                     res[col_ol] **= 0.5
    #                     res[col_4k] **= 0.5
    #                     res[col_da] **= 0.5
    #
    #             if not ((output=='ismn') and ('R2' not in met)):
    #                 print('filtered')
    #                 res[col_ol][res[col_ol] <= 0] = np.nan
    #                 res[col_4k][res[col_4k] <= 0] = np.nan
    #                 res[col_da][res[col_da] <= 0] = np.nan
    #                 res[col_ol][res[col_ol] >= 1] = np.nan
    #                 res[col_4k][res[col_4k] >= 1] = np.nan
    #                 res[col_da][res[col_da] >= 1] = np.nan
    #
    #             res['da'] = res[col_da] - res[col_ol]
    #             res['4k'] = res[col_4k] - res[col_ol]
    #
    #             ax = plt.subplot(6, 3, i * 6 + j * 3 + k + 1)
    #
    #             p1 = res['4k'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.8)
    #             p2 = res['da'].hist(bins=bins, grid=False, ax=ax, range=lim, alpha=0.6)
    #             plt.yticks(color='w', fontsize=5)
    #             if i < 2:
    #                 plt.xticks(color='w', fontsize=1)
    #             if i == 0:
    #                 plt.title(title, fontsize=fontsize-3)
    #             if (j == 0) & (k == 0):
    #                 plt.ylabel(label, fontsize=fontsize)
    #             if (i == 2) & (j == 2) & (k == 1):
    #                 plt.legend(labels=['R$_{4K, ref}$ - R$_{OL, ref}$', 'R$_{TC, ref}$ - R$_{OL, ref}$'], loc='lower right', fontsize=fontsize-4)
    #             plt.axvline(color='black', linestyle='--', linewidth=1)
    #             plt.xlim(lim)
    #             ylim = ax.get_ylim()
    #             yloc1 = ylim[1] - (ylim[1] - ylim[0])/10
    #             yloc2 = ylim[1] - 1.6*(ylim[1] - ylim[0])/10
    #             plt.text(xloc, yloc1, 'mean = %.2f' % res['4k'].mean(), color='#1f77b4', fontsize=fontsize-3)
    #             plt.text(xloc, yloc2, 'mean = %.2f' % res['da'].mean(), color='#ff7f0e', fontsize=fontsize-3)
    #
    # f.savefig(dir_out / f'stats_ascat.png', dpi=300, bbox_inches='tight')
    # plt.close()

def plot_freq_components(dir_out):

    ds = GEOSldas_io('ObsFcstAna', exp='NLv4_M36_US_DA_SMAP_Pcorr_LTST').timeseries

    ts = ds['obs_fcst'][:,0, 50,120].to_pandas().dropna()

    clim = calc_climatology(ts)

    anom_lt = calc_anom(ts, mode='longterm')
    anom_st = calc_anom(ts, mode='shortterm')
    clim = calc_anom(ts, return_clim=True)

    anom_lst = anom_lt+anom_st
    anom_lst.name = 'Anomalies (HF + LF)'

    f, axes = plt.subplots(figsize=(18,6),
                             nrows=3,
                             ncols=1,
                             sharex=True)


    fontsize = 11
    df = pd.concat((ts, clim), axis=1)
    df.columns = ['$T_b$ signal', 'Climatology']
    df.plot(ax=axes[0], color=['darkorange', 'blue'], linewidth=1.7, fontsize=fontsize).legend(loc='upper right', fontsize=fontsize)

    df = pd.DataFrame(anom_lst)
    df.plot(ax=axes[1], color='crimson', ylim=[-18,18], linewidth=1.7, fontsize=fontsize).legend(loc='upper right', fontsize=fontsize)
    axes[1].axhline(color='black', linestyle='--', linewidth=0.8)

    df = pd.concat((anom_st, anom_lt), axis=1)
    df.columns = ['HF signal', 'LF signal']
    df.plot(ax=axes[2], color=['goldenrod', 'teal'], ylim=[-15,15], linewidth=1.7, fontsize=fontsize).legend(loc='upper right', fontsize=fontsize)
    axes[2].axhline(color='black', linestyle='--', linewidth=0.8)

    plt.xlabel('')
    plt.minorticks_off()
    plt.xlim('2015-04', '2021,04')
    plt.xticks(fontsize=fontsize)

    f.savefig(dir_out / f'frequency_components.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_perturbations(dir_out):

    root = Path(r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\observation_perturbations\Pcorr')
    res_tc = pd.read_csv(r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv',
                      index_col=0)
    pc = 'Pcorr'
    io = GEOSldas_io('ObsFcstAna')
    io2 = LDASsa_io('ObsFcstAna')

    lut = pd.read_csv(Paths().lut, index_col=0)
    ind = np.vectorize(io.grid.colrow2tilenum)(lut.ease2_col, lut.ease2_row, local=False)

    dtype, hdr, length = template_error_Tb40()

    f = plt.figure(figsize=(23, 15))

    fontsize=14
    cbrange = [0,8]
    cmap=cc.cm.bjy
    # cmap='viridis'

    modes = ['anom_lst', 'anom_lt', 'anom_st']
    titles = ['Anomalies', 'LF signal', 'HF signal']

    for i, (mode, title) in enumerate(zip(modes,titles)):

        fA = root / f'{mode}' / 'SMOS_fit_Tb_A.bin'
        fD = root / f'{mode}' / 'SMOS_fit_Tb_D.bin'

        imgA = io2.read_fortran_binary(fA, dtype, hdr=hdr, length=length)
        imgD = io2.read_fortran_binary(fD, dtype, hdr=hdr, length=length)

        imgA.index += 1
        imgD.index += 1

        r_asc_smap = res_tc[f'r_grid_{mode}_p_ASCAT_SMAP']
        r_asc_clsm = res_tc[f'r_grid_{mode}_p_ASCAT_CLSM']
        r_smap_clsm = res_tc[f'r_grid_{mode}_p_SMAP_CLSM']
        thres = 0.2
        ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index
        ind_valid = np.vectorize(io.grid.colrow2tilenum)(res_tc.loc[ind_valid,'col'].values, res_tc.loc[ind_valid,'row'].values, local=False)

        plt.subplot(4,3,i+1)
        im = plot_ease_img2(imgA.reindex(ind).reindex(ind_valid),'err_Tbv', cbrange=cbrange, cmap=cmap, io=io, plot_cmap=False)
        if i==0:
            plt.ylabel('$\hat{R}$ (V-pol, Asc.)', fontsize=fontsize)
        plt.title(title, fontsize=fontsize)

        plt.subplot(4,3,i+4)
        plot_ease_img2(imgD.reindex(ind).reindex(ind_valid),'err_Tbv', cbrange=cbrange, cmap=cmap, io=io, plot_cmap=False)
        if i==0:
            plt.ylabel('$\hat{R}$ (V-pol, Dsc.)', fontsize=fontsize)

        plt.subplot(4,3,i+7)
        plot_ease_img2(imgA.reindex(ind).reindex(ind_valid),'err_Tbh', cbrange=cbrange, cmap=cmap, io=io, plot_cmap=False)
        if i==0:
            plt.ylabel('$\hat{R}$ (H-pol, Asc.)', fontsize=fontsize)

        plt.subplot(4,3,i+10)
        plot_ease_img2(imgD.reindex(ind).reindex(ind_valid),'err_Tbh', cbrange=cbrange, cmap=cmap, io=io, plot_cmap=False)
        if i==0:
            plt.ylabel('$\hat{R}$ (H-pol, Dsc.)', fontsize=fontsize)

    plot_centered_cbar(f, im, 3, fontsize=fontsize-2, bottom=0.06, fig_ind=7, wdth=0.02)

    plt.savefig(dir_out / f'perturbations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_tca_uncertainties(dir_out):

    sensors = ['ASCAT', 'SMAP', 'CLSM']

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    figsize = (15, 5)
    fontsize = 12
    cb = [0, 0.04]

    modes = ['anom_lst', 'anom_lt', 'anom_st']
    titles = ['Anomalies', 'LF signal', 'HF signal']
    labels = ['$\widehat{std}(\epsilon_{\Theta,smap})$', '$\widehat{std}(\epsilon_{\Theta,clsm})$']

    f = plt.figure(figsize=figsize)

    n = 0
    pos = []
    for s, l in zip(sensors[1::],labels):
        for mode, title in zip(modes, titles):
            n += 1

            plt.subplot(2, 3, n)

            tag = 'ubrmse_grid_' + mode + '_m_' + s + '_tc_ASCAT_SMAP_CLSM'

            r_asc_smap = res[f'r_grid_{mode}_p_ASCAT_SMAP']
            r_asc_clsm = res[f'r_grid_{mode}_p_ASCAT_CLSM']
            r_smap_clsm = res[f'r_grid_{mode}_p_SMAP_CLSM']
            thres = 0.2
            ind_valid = res[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

            im_r = plot_ease_img(res.reindex(ind_valid), tag, fontsize=fontsize, cbrange=cb, cmap='viridis', print_mean=True)
            # if (n == 6) | (n == 7):
            #     pos += [im_r.axes.get_position()]

            if s == 'SMAP':
                plt.title(title, fontsize=fontsize)
            if mode == 'anom_lst':
                plt.ylabel(l, fontsize=fontsize)

    plot_centered_cbar(f, im_r, 3, fontsize=fontsize - 2, bottom=0.07)

    fout = dir_out / 'tca_uncertainties.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

def plot_orthogonality_check(dir_out):

    sensors = ['ASCAT', 'SMAP', 'CLSM']

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    figsize = (11, 3)
    fontsize = 10
    # cb = [-0.02, 0.02]
    cb = [-0.02, 0.02]

    f = plt.figure(figsize=figsize)

    valid = pd.Series(True, index=res.index)
    for mode in ['anom_lt', 'anom_st', 'anom_lst']:
        r_asc_smap = res[f'r_grid_{mode}_p_ASCAT_SMAP']
        r_asc_clsm = res[f'r_grid_{mode}_p_ASCAT_CLSM']
        r_smap_clsm = res[f'r_grid_{mode}_p_SMAP_CLSM']
        thres = 0.2
        valid &= ((r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres))

    ind_valid = res[valid].index

    for i, s in enumerate(sensors[1::]):

        plt.subplot(1, 2, i+1)

        tag_lt = 'ubrmse_grid_anom_lt_m_' + s + '_tc_ASCAT_SMAP_CLSM'
        tag_st = 'ubrmse_grid_anom_st_m_' + s + '_tc_ASCAT_SMAP_CLSM'
        tag_lst = 'ubrmse_grid_anom_lst_m_' + s + '_tc_ASCAT_SMAP_CLSM'

        res['diff'] =  (res[tag_lst] - np.sqrt(res[tag_lt]**2 + res[tag_st]**2))

        im_r = plot_ease_img(res.reindex(ind_valid), 'diff', fontsize=fontsize+2, cbrange=cb, cmap=cc.cm.bjy, print_meanstd=True)

        plt.title(s, fontsize=fontsize)

    plot_centered_cbar(f, im_r, 2, fontsize=fontsize - 2, bottom=0.00, hspace=0.030, pad=0.02, wdth=0.04)

    fout = dir_out / 'orthogonality_verification.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

def plot_uncertainty_ratios(dir_out):

    sensors = ['ASCAT', 'SMAP', 'CLSM']

    res = pd.read_csv(r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv', index_col=0)
    res_tc = pd.read_csv(
        r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv', index_col=0)

    tg = GEOSldas_io().grid.tilegrids
    res_cols = res.col.values - tg.loc['domain', 'i_offg']
    res_rows = res.row.values - tg.loc['domain', 'j_offg']

    figsize = (16, 4)
    fontsize = 10
    cb = [-10, 10]
    cmap = cc.cm.bjy

    modes = ['anom_lt', 'anom_lt', 'anom_lt', 'anom_st']
    titles = ['$CTRL$', '$DA_{anom}$', '$DA_{LF}$', '$DA_{HF}$']

    ios = [GEOSldas_io('ObsFcstAna', exp=f'NLv4_M36_US_DA_Pcorr_scl_SMAP_{mode}').timeseries for mode in modes]
    io_ol = GEOSldas_io('ObsFcstAna', exp=f'NLv4_M36_US_OL_Pcorr').timeseries

    grid = GEOSldas_io().grid

    f = plt.figure(figsize=figsize)

    for n, (mode, title, io_da) in enumerate(zip(modes, titles, ios)):

        if n > 0:
            plt.subplot(2, 4, n+1)
            tagP = 'ubrmse_grid_' + mode + '_m_CLSM_tc_ASCAT_SMAP_CLSM'
            tagR = 'ubrmse_grid_' + mode + '_m_SMAP_tc_ASCAT_SMAP_CLSM'
            res['tmp'] = 10*np.log10(res[tagP]**2/ res[tagR]**2)

            r_asc_smap = res_tc[f'r_grid_{mode}_p_ASCAT_SMAP']
            r_asc_clsm = res_tc[f'r_grid_{mode}_p_ASCAT_CLSM']
            r_smap_clsm = res_tc[f'r_grid_{mode}_p_SMAP_CLSM']
            thres = 0.2
            ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

            img = plot_ease_img(res.reindex(ind_valid), 'tmp', fontsize=fontsize, cbrange=cb, cmap=cmap, plot_cb=False)
            plt.title(title, fontsize=fontsize)
            if n == 1:
                plt.ylabel('TCA unc. ratio', fontsize=fontsize)

    for n, (mode, title, io_da) in enumerate(zip(modes, titles, ios)):

        if mode != '4K':
            r_asc_smap = res_tc[f'r_grid_{mode}_p_ASCAT_SMAP']
            r_asc_clsm = res_tc[f'r_grid_{mode}_p_ASCAT_CLSM']
            r_smap_clsm = res_tc[f'r_grid_{mode}_p_SMAP_CLSM']
            thres = 0.2
            ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index
        else:
            ind_valid = res.index

        avg = np.full(io_da['obs_obsvar'].shape[1::], np.nan)
        for spc in range(4):
            tmp1 = io_da['obs_obsvar'][:, spc, :, :].values
            tmp2 = io_ol['obs_fcstvar'][:, spc, :, :].values
            avg[spc,:,:] = np.nanmean(tmp2 / tmp1, axis=0)
        avg = np.nanmean(avg, axis=0)

        res['avg'] = 10*np.log10(avg[res_rows, res_cols])

        plt.subplot(2, 4, n+5)
        img = plot_ease_img(res.reindex(ind_valid), 'avg', fontsize=fontsize, cbrange=cb, cmap=cmap, plot_cb=False)
        if n == 0:
            plt.title(title, fontsize=fontsize)
            plt.ylabel('Ens. var. ratio', fontsize=fontsize)

    plot_centered_cbar(f, img, 4, fontsize=fontsize, bottom=0.07)

    fout = dir_out / 'uncertainty_ratio.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ensvar_ratio(dir_out):

    dtype, hdr, length = template_error_Tb40()

    f = plt.figure(figsize=(25, 12))

    fontsize = 14
    cb = [-10, 10]
    cmap = cc.cm.bjy

    modes = ['4K', 'abs', 'anom_lst', 'anom_lt', 'anom_st']
    titles = ['4K benchmark', 'Total signal', 'Anomalies', 'LF signal', 'HF signal']
    labels = ['(V-pol, Asc.)', '(V-pol, Dsc.)', '(H-pol, Asc.)', '(H-pol, Dsc.)']

    ios = [GEOSldas_io('ObsFcstAna', exp=f'NLv4_M36_US_DA_SMAP_Pcorr_{mode}').timeseries for mode in modes]
    io_ol = GEOSldas_io('ObsFcstAna', exp=f'NLv4_M36_US_OL_Pcorr').timeseries

    grid = GEOSldas_io().grid

    for i, (io_da, tit) in enumerate(zip(ios,titles)):

        for spc, label in zip(range(4), labels):

            tmp1 = io_da['obs_obsvar'][:, spc, :, :].values
            tmp2 = io_ol['obs_fcstvar'][:, spc, :, :].values
            avg = np.nanmean(tmp1 / tmp2, axis=0)
            # ratio = io_da['obs_obsvar'] / io_ol['obs_fcstvar']
            # ratio = io['obs_obsvar']
            # avg = ratio.mean(dim='time', skipna=True)

        plt.subplot(4, 5, spc*5 + i + 1)
        img = plot_latlon_img(10*np.log10(avg), io_da.lon.values, io_da.lat.values, fontsize=fontsize, cbrange=cb, cmap=cmap, plot_cb=False)
        if spc == 0:
            plt.title(tit, fontsize=fontsize)
        if i == 0:
            plt.ylabel(label, fontsize=fontsize)

    plot_centered_cbar(f, img, 5, fontsize=fontsize-2)

    fout = dir_out / 'ensvar_ratio.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

def plot_filter_diagnostics(res_path, dir_out):

    fname = res_path / 'filter_diagnostics.nc'

    res = pd.read_csv(r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv',
                     index_col=0)
    res_tc = pd.read_csv(
        r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\sm_validation\Pcorr\result.csv', index_col=0)
    tg = GEOSldas_io().grid.tilegrids
    res_cols = res.col.values - tg.loc['domain', 'i_offg']
    res_rows = res.row.values - tg.loc['domain', 'j_offg']
    r_asc_smap = res_tc[f'r_grid_anom_lst_p_ASCAT_SMAP']
    r_asc_clsm = res_tc[f'r_grid_anom_lst_p_ASCAT_CLSM']
    r_smap_clsm = res_tc[f'r_grid_anom_lst_p_SMAP_CLSM']
    thres = 0.2
    ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

    fontsize = 14

    root = Path(r'D:\_KUL_backup_2022\data_sets\GEOSldas_runs')
    runs = [run.name for run in root.glob('*_DA_SMAP_*')]
    runs += ['NLv4_M36_US_OL_Pcorr', 'NLv4_M36_US_OL_noPcorr']

    tags = ['OL_Pcorr', 'Pcorr_4K', f'Pcorr_anom_lst', 'Pcorr_LTST']
    iters = [np.where([tag in run for run in runs])[0][0] for tag in tags]

    titles = ['Open-loop', '$CTRL$', '$DA_{anom}']
    labels = ['H pol. / Asc.', 'V pol. / Asc.']

    with Dataset(fname) as ds:

        lons = ds.variables['lon'][:]
        lats = ds.variables['lat'][:]
        lons, lats = np.meshgrid(lons, lats)

        var = 'innov_autocorr'
        cbrange = [0, 0.7]
        step = 0.2
        cmap = 'viridis'

        f = plt.figure(figsize=(19, 6))

        for j, (spc, label) in enumerate(zip([0, 2],labels)):
            for i, (it_tit, it) in enumerate(zip(titles,iters)):

                title = it_tit if j == 0 else ''

                plt.subplot(2, 3, j*3 + i +1)
                data = ds.variables[var][:,:,it,spc]

                res['tmp'] = data[res_rows, res_cols]
                im = plot_ease_img(res.reindex(ind_valid), 'tmp', fontsize=fontsize, cbrange=cbrange, cmap=cmap, title=title, plot_cb=False, print_meanstd=True)
                if i==0:
                    plt.ylabel(label, fontsize=fontsize)

        plot_centered_cbar(f, im, 3, fontsize=fontsize-2, bottom=0.07)
        fout = dir_out / f'{var}.png'
        f.savefig(fout, dpi=300, bbox_inches='tight')
        plt.close()

        # plt.tight_layout()
        # plt.show()

if __name__=='__main__':

    res_path = Path(r'D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\validation_all')
    # dir_out = Path('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/plots')
    dir_out = Path(r'H:\work\SMAP_DA_paper\plots')
    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    # plot_predicted_skillgain(dir_out)
    plot_ascat_eval_absolute(res_path, dir_out)
    plot_ascat_eval_relative(res_path, dir_out)
    # plot_statistics(res_path, dir_out)
    # plot_ismn_statistics(res_path, dir_out)
    # plot_ascat_statistics(res_path, dir_out)
    # plot_freq_components(dir_out)
    # plot_perturbations(dir_out)
    # plot_tca_uncertainties(dir_out)
    # plot_orthogonality_check(dir_out)
    # plot_uncertainty_ratios(dir_out)
    # plot_filter_diagnostics(res_path, dir_out)