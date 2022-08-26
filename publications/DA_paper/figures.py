
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path
from netCDF4 import Dataset

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
sns.set_context('talk', font_scale=0.8)
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


def plot_potential_skillgain_simple(dir_out):

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']
    titles = ['Total signal', 'Anomalies', 'LF signal', 'HF signal']

    f = plt.figure(figsize=(16,8))

    for i, (mode,tit) in enumerate(zip(modes,titles)):

        R = res[f'ubrmse_grid_{mode}_m_SMAP_tc_ASCAT_SMAP_CLSM']**2
        P = res[f'ubrmse_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM']**2

        R2 = res[f'r2_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM']

        K = P / (R + P)
        NSR = (1 - R2) / R2

        R2upd = 1 / (1 + (1 - K) * NSR)

        res['gain_pot'] = R2upd - R2

        plt.subplot(2, 2, i+1)
        im = plot_ease_img(res, 'gain_pot', fontsize=12, cbrange=[-0.2,0.2], cmap=cc.cm.bjy, log_scale=False, title=tit, plot_cb=False)
    plot_centered_cbar(f, im, 2, fontsize=10)

    fout = dir_out / 'skillgain_simple'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

def plot_actual_skillgain(res_path, dir_out):

    runs = [f'DA_Pcorr_{err}' for err in ['abs','anom_lst','anom_lt','anom_st']]
    titles = ['R$_{Total}$ - R$_{OL}$', 'R$_{Anomaly}$ - R$_{OL}$', 'R$_{LF}$ - R$_{OL}$', 'R$_{HF}$ - R$_{OL}$']

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)
    res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)
    m = 'abs'

    fontsize = 12

    f = plt.figure(figsize=(16, 8))

    for cnt, (r, tit) in enumerate(zip(runs,titles)):

        plt.subplot(2, 2, cnt+1)

        col = f'ana_r_corr_{r}_{m}'
        ref_col = f'ana_r_corr_OL_Pcorr_{m}'

        r_asc_smap = res_tc[f'r_grid_{m}_p_ASCAT_SMAP']
        r_asc_clsm = res_tc[f'r_grid_{m}_p_ASCAT_CLSM']
        r_smap_clsm = res_tc[f'r_grid_{m}_p_SMAP_CLSM']
        thres = 0.2
        ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

        res[col][res[col] < 0] = 0
        res[ref_col][res[ref_col] < 0] = 0
        res[col][res[col] > 1] = 1
        res[ref_col][res[ref_col] > 1] = 1

        res[f'{col}_diff'] = res[col] - res[ref_col]
        im = plot_ease_img(res.reindex(ind_valid), f'{col}_diff' , title=tit, cmap=cc.cm.bjy, cbrange=[-0.2,0.2], fontsize=fontsize, print_median=False)
    plot_centered_cbar(f, im, 2, fontsize=10)

    f.savefig(dir_out / f'skillgain_act.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_potential_skillgain_decomposed(dir_out):

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    R_abs = res[f'ubrmse_grid_abs_m_SMAP_tc_ASCAT_SMAP_CLSM']**2
    P_abs = res[f'ubrmse_grid_abs_m_CLSM_tc_ASCAT_SMAP_CLSM']**2

    R_anom_lt = res[f'ubrmse_grid_anom_lt_m_SMAP_tc_ASCAT_SMAP_CLSM']**2
    P_anom_lt = res[f'ubrmse_grid_anom_lt_m_CLSM_tc_ASCAT_SMAP_CLSM']**2

    R_anom_lst = res[f'ubrmse_grid_anom_lst_m_SMAP_tc_ASCAT_SMAP_CLSM']**2
    P_anom_lst = res[f'ubrmse_grid_anom_lst_m_CLSM_tc_ASCAT_SMAP_CLSM']**2

    R_anom_st = res[f'ubrmse_grid_anom_st_m_SMAP_tc_ASCAT_SMAP_CLSM']**2
    P_anom_st = res[f'ubrmse_grid_anom_st_m_CLSM_tc_ASCAT_SMAP_CLSM']**2

    R_clim = (R_abs - R_anom_lt - R_anom_st).abs()
    P_clim = (P_abs - P_anom_lt - P_anom_st).abs()

    R23_arr = (R_anom_lst - (R_anom_lt + R_anom_st)) / 2
    P23_arr = (P_anom_lst - (P_anom_lt + P_anom_st)) / 2

    # Baseline estimates
    # R2 = res[f'r2_grid_abs_m_CLSM_tc_ASCAT_SMAP_CLSM']
    R2 = res[f'r2_grid_anom_lst_m_CLSM_tc_ASCAT_SMAP_CLSM']
    SNR = R2 / (1 - R2)
    SIG = SNR * P_abs

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']
    titles = ['Total signal', 'Anomalies', 'LF signal', 'HF signal']
    result = pd.DataFrame(index=res.index, columns=modes)
    result['row'] = res.row
    result['col'] = res.col

    for cnt, i in enumerate(res.index):
        print(f'{cnt} / {len(res)}')

        R11 = R_clim.loc[i]
        R22 = R_anom_lt.loc[i]
        R33 = R_anom_st.loc[i]
        P11 = P_clim.loc[i]
        P22 = P_anom_lt.loc[i]
        P33 = P_anom_st.loc[i]

        r_c_l = 0.5
        r_c_s = 0.2
        r_l_s = 0.2

        R12 = r_c_l * np.sqrt(R11 * R22)
        R13 = r_c_s * np.sqrt(R11 * R33)
        # R23 = r_l_s * np.sqrt(R22 * R33)
        P12 = r_c_l * np.sqrt(P11 * P22)
        P13 = r_c_s * np.sqrt(P11 * P33)
        # P23 = r_l_s * np.sqrt(P22 * P33)

        R23 = R23_arr[i]
        P23 = P23_arr[i]

        # S = np.matrix([[R11, R12, R13, 0,   0,   0  ],
        #                [R12, R22, R23, 0,   0,   0  ],
        #                [R13, R23, R33, 0,   0,   0  ],
        #                [0,   0,   0,   P11, P12, P13],
        #                [0,   0,   0,   P12, P22, P23],
        #                [0,   0,   0,   P13, P23, P33]])
        S = np.matrix([[R22, R23 ,  0,   0  ],
                       [R23, R33,   0,   0  ],
                       [0,   0,   P22, P23],
                       [0,   0,   P23, P33]])

        for mode in modes:

            P = res.loc[i, f'ubrmse_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM'] ** 2
            R = res.loc[i, f'ubrmse_grid_{mode}_m_SMAP_tc_ASCAT_SMAP_CLSM'] ** 2
            K = P / (R + P)

            # A = np.matrix([K, K, K, 1-K, 1-K, 1-K])
            A = np.matrix([K, K, 1-K, 1-K])


            P_upd = (A * S * A.T)[0,0]

            NSR_upd = P_upd / SIG.loc[i]
            R2upd = 1 / (1 + NSR_upd)

            result.loc[i, mode] = R2upd - R2.loc[i]

    f = plt.figure(figsize=(16,8))

    for i, (mode, title) in enumerate(zip(modes, titles)):
        plt.subplot(2, 2, i+1)
        im = plot_ease_img(result, mode, fontsize=12, cbrange=[-0.2,0.2], cmap=cc.cm.bjy, log_scale=False, title=title, plot_cb=False)
    plot_centered_cbar(f, im, 2, fontsize=10)

    fout = dir_out / 'skillgain_pot.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def plot_station_locations(dir_out):

    nl = pd.read_csv(r"D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\validation_all\insitu_TCA.csv", index_col=0)
    nl.index = nl.network
    nl.drop('network', axis='columns', inplace=True)

    networks  = ['SCAN', 'USCRN']
    nl = nl.loc[nl.index.isin(networks),:]

    print('SCAN: %i' % nl[nl.index=='SCAN'].shape[0])
    print('USCRN: %i' % nl[nl.index=='USCRN'].shape[0])

    lats_scan = nl[nl.index=='SCAN']['latitude'].values
    lons_scan = nl[nl.index=='SCAN']['longitude'].values

    lats_crn = nl[nl.index=='USCRN']['latitude'].values
    lons_crn = nl[nl.index=='USCRN']['longitude'].values

    llcrnrlat=24
    urcrnrlat=51
    llcrnrlon=-128
    urcrnrlon=-64

    figsize = (20, 10)

    f = plt.figure(num=None, figsize=figsize, dpi=90, facecolor='w', edgecolor='k')

    m = Basemap(projection='mill',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
            llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='c',)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    xs, ys = m(lons_scan,lats_scan)
    xc, yc = m(lons_crn,lats_crn)

    plt.plot(xs,ys,'or',markersize = 13, label='SCAN')
    plt.plot(xc,yc,'Pb',markersize = 13, label='USCRN')

    plt.legend(fontsize=26,loc=3)

    fname = dir_out / 'station_locations.png'
    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

#    plt.tight_layout()

def plot_ismn_vs_ascat(res_path, dir_out):

    res = pd.read_csv(res_path / 'insitu_TCA.csv', index_col=0)
    res.index = res.network
    res.drop('network', axis='columns', inplace=True)

    networks  = ['SCAN', 'USCRN']
    res = res.loc[res.index.isin(networks),:]

    lats = res['latitude'].values
    lons = res['longitude'].values

    res_asc = pd.read_csv(Path(res_path / 'ascat_eval.csv'), index_col=0)
    res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    mode = 'anom_st'

    run_ol = 'Pcorr_OL'
    # run_ol = 'Pcorr_4K'

    # run_da = f'DA_Pcorr_{mode}'
    run_da = f'Pcorr_LTST'

    fontsize = 16

    f = plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)

    col_da = f'ana_r_corr_{run_da}_{mode}'
    col_ol = f'ana_r_corr_{run_ol}_{mode}'

    r_asc_smap = res_tc[f'r_grid_{mode}_p_ASCAT_SMAP']
    r_asc_clsm = res_tc[f'r_grid_{mode}_p_ASCAT_CLSM']
    r_smap_clsm = res_tc[f'r_grid_{mode}_p_SMAP_CLSM']
    thres = 0.2
    ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index
    res_asc[col_ol][res_asc[col_ol] < 0] = 0
    res_asc[col_ol][res_asc[col_ol] > 1] = 1
    res_asc[col_da][res_asc[col_da] < 0] = 0
    res_asc[col_da][res_asc[col_da] > 1] = 1

    res_asc['diff'] = res_asc[col_da] - res_asc[col_ol]

    im = plot_ease_img(res_asc.reindex(ind_valid), 'diff', title='R$_{LTST}$ - R$_{OL}$', cmap=cc.cm.bjy, cbrange=[-0.15, 0.15],
                       fontsize=fontsize, print_median=True, plot_cb=True)

    ax = plt.subplot(1,2,2)
    llcrnrlat = 24
    urcrnrlat = 51
    llcrnrlon = -128
    urcrnrlon = -64
    m = Basemap(projection='mill', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, resolution='c', )
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    xs, ys = m(lons, lats)
    c = res[f'R2_model_{run_da}_{mode}_sm_surface'].values - res[f'R2_model_{run_ol}_{mode}_sm_surface'].values
    sc = plt.scatter(xs, ys, c=c, s=30, label='R2', vmin=-0.15, vmax=0.15, cmap=cc.cm.bjy)
    plt.title('R$_{LTST,ISMN}$ - R$_{OL,ISMN}$', fontsize=fontsize)
    cb = m.colorbar(sc, "bottom", size="7%", pad="5%")
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize-2)
    x, y = m(-79, 25)
    plt.text(x, y, 'm. = %.3f' % np.nanmedian(c), fontsize=fontsize-4)

    fout = dir_out / f'ismn_vs_ascat_{mode}.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()



def plot_ismn_statistics(res_path, dir_out):

    res = pd.read_csv(res_path / 'insitu_TCA.csv', index_col=0)
    res.index = res.network
    res.drop('network', axis='columns', inplace=True)

    variables = ['sm_surface', 'sm_rootzone']
    var_labels = ['surface', 'root-zone']

    pc = 'Pcorr'

    runs = [f'OL_{pc}'] + [f'DA_{pc}_{err}' for err in ['4K', 'abs', 'anom_lst', 'anom_lt', 'anom_st']]
    run_labels = ['Open-loop', '4K benchmark', 'Total signal', 'Anomalies', 'LF signal', 'HF signal']

    n_runs = len(runs)
    offsets = np.linspace(-0.5 + 1/(n_runs+1), 0.5 - 1/(n_runs+1),n_runs)
    cols = ['darkred', 'coral',
            'darkgreen', 'forestgreen', 'limegreen', 'lightgreen',
            'navy', 'mediumblue', 'slateblue', 'lightblue',
            'rebeccapurple', 'blueviolet', 'mediumorchid', 'plum'][0:n_runs]

    width = (offsets[1]- offsets[0]) / 2.5
    ss = offsets[1] - offsets[0]
    fontsize = 16

    networks  = ['SCAN', 'USCRN']
    res = res.loc[res.index.isin(networks),:]


    # titles = ['ubRMSD', 'ubRMSE', 'Pearson R$^2$ ', 'TCA R$^2$']
    titles = ['ubRMSD', 'Pearson-R', 'ubRMSE (TCA) ', 'R (TCA)']

    ylims = [[0.00, 0.1],
             [0.0, 1],
             [0.0, 0.05],
             [0.2, 1.0]]

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']

    for mode in modes:

        f = plt.figure(figsize=(20,8))

        valss = [[[res[f'ubRMSD_model_insitu_{run}_{mode}_{var}'].values for run in runs] for var in variables],
                 [[res[f'R_model_insitu_{run}_{mode}_{var}'].values for run in runs] for var in variables],
                 [[res[f'ubRMSE_model_{run}_{mode}_{var}'].values for run in runs] for var in variables],
                 [[np.sqrt(res[f'R2_model_{run}_{mode}_{var}'].values) for run in runs] for var in variables]]

        for n, (vals, tit, ylim) in enumerate(zip(valss, titles, ylims)):

            ax = plt.subplot(2,2,n+1)

            plt.grid(color='k', linestyle='--', linewidth=0.25)

            data = list()
            ticks = list()
            pos = list()
            colors = list()

            for i, (val, var_label) in enumerate(zip(vals,var_labels)):

                ticks.append(var_label)
                for col,offs, v in zip(cols,offsets,val):
                    tmp_data = v
                    tmp_data = tmp_data[~np.isnan(tmp_data)]
                    data.append(tmp_data)
                    pos.append(i+1 + offs)
                    colors.append(col)

            box = ax.boxplot(data, whis=[5,95], showfliers=False, positions=pos, widths=width, patch_artist=True)
            for patch, color in zip(box['boxes'], colors):
                patch.set(color='black', linewidth=2)
                patch.set_facecolor(color)
            for patch in box['medians']:
                patch.set(color='black', linewidth=2)
            for patch in box['whiskers']:
                patch.set(color='black', linewidth=1)
            plt.xticks(np.arange(len(var_labels))+1, ticks,fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xlim(0.5,len(ticks)+0.5)
            plt.ylim(ylim)
            for k in np.arange(len(var_labels)):
                plt.axvline(k+0.5, linewidth=1.5, color='k')
                plt.axvline(k+1 - 2*ss, linewidth=1, color='k', linestyle='--')
                plt.axvline(k+1 - 1*ss, linewidth=1, color='k', linestyle='--')
                # plt.axvline(k+1 + 0*ss, linewidth=1, color='k', linestyle='--')
            if n == 1:
                plt.figlegend((box['boxes'][0:n_runs]),run_labels,'upper right',fontsize=fontsize-4)
            ax.set_title(tit ,fontsize=fontsize)

        f.savefig(dir_out / f'ismn_stats_{mode}.png', dpi=300, bbox_inches='tight')
        plt.close()



def plot_ascat_eval_uncorrected(res_path, dir_out):

    runs = ['OL_noPcorr'] + [f'DA_noPcorr_{err}' for err in ['4K','abs','anom_lst','anom_lt','anom_st']]
    titles = ['R$_{OL}$', 'R$_{4K}$ - R$_{OL}$', 'R$_{Total}$ - R$_{4K}$', 'R$_{Anomaly}$ - R$_{4K}$', 'R$_{LF}$ - R$_{4K}$', 'R$_{HF}$ - R$_{4K}$']

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)
    res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/noPcorr/result.csv', index_col=0)
    modes = ['anom_st']

    cb_r = [0.2, 1]
    cb_diff_ol =  [-0.2, 0.2]
    cb_diff_da = [-0.06, 0.06]

    fontsize = 14

    for i, m in enumerate(modes):

        f = plt.figure(figsize=(20,8))

        for cnt, (r, tit) in enumerate(zip(runs,titles)):

            plt.subplot(2, 3, cnt+1)

            col = f'ana_r_corr_{r}_{m}'
            ref_col_ol = f'ana_r_corr_OL_noPcorr_{m}'
            ref_col_da = f'ana_r_corr_DA_noPcorr_4K_{m}'

            r_asc_smap = res_tc[f'r_grid_{m}_p_ASCAT_SMAP']
            r_asc_clsm = res_tc[f'r_grid_{m}_p_ASCAT_CLSM']
            r_smap_clsm = res_tc[f'r_grid_{m}_p_SMAP_CLSM']
            thres = 0.2
            ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

            res[col][res[col] < 0] = 0
            res[ref_col_ol][res[ref_col_ol] < 0] = 0
            res[ref_col_da][res[ref_col_da] < 0] = 0
            res[col][res[col] > 1] = 1
            res[ref_col_ol][res[ref_col_ol] > 1] = 1
            res[ref_col_da][res[ref_col_da] > 1] = 1

            if cnt == 0:
                im = plot_ease_img(res.reindex(ind_valid), ref_col_ol , title=tit, cmap='viridis', cbrange=cb_r, fontsize=fontsize, print_median=True, plot_cb=True)
            elif cnt == 1:
                res[f'{col}_diff'] = res[col] - res[ref_col_ol]
                im = plot_ease_img(res.reindex(ind_valid), f'{col}_diff' , title=tit, cmap=cc.cm.bjy, cbrange=cb_diff_ol, fontsize=fontsize, print_median=True, plot_cb=True)
            else:
                res[f'{col}_diff'] = res[col] - res[ref_col_da]
                im = plot_ease_img(res.reindex(ind_valid), f'{col}_diff' , title=tit, cmap=cc.cm.bjy, cbrange=cb_diff_da, fontsize=fontsize, print_median=True, plot_cb=True)

        f.savefig(dir_out / f'ascat_eval_{m}_uncorr.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_ascat_eval(res_path, dir_out):

    runs = ['OL_Pcorr'] + [f'DA_Pcorr_{err}' for err in ['4K','abs','anom_lst','anom_lt','anom_st']]
    titles = ['R$_{OL}$', 'R$_{4K}$ - R$_{OL}$', 'R$_{Total}$ - R$_{4K}$', 'R$_{Anomaly}$ - R$_{4K}$', 'R$_{LF}$ - R$_{4K}$', 'R$_{HF}$ - R$_{4K}$']

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)
    res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)
    modes = ['abs', 'anom_lst', 'anom_lt','anom_st']

    cb_r = [0.2, 1]
    cb_diff_ol =  [-0.1, 0.1]
    cb_diff_da = [-0.1, 0.1]

    fontsize = 14

    for i, m in enumerate(modes):

        f = plt.figure(figsize=(20,8))

        for cnt, (r, tit) in enumerate(zip(runs,titles)):

            plt.subplot(2, 3, cnt+1)

            col = f'ana_r_corr_{r}_{m}'
            ref_col_ol = f'ana_r_corr_OL_Pcorr_{m}'
            ref_col_da = f'ana_r_corr_DA_Pcorr_4K_{m}'

            r_asc_smap = res_tc[f'r_grid_{m}_p_ASCAT_SMAP']
            r_asc_clsm = res_tc[f'r_grid_{m}_p_ASCAT_CLSM']
            r_smap_clsm = res_tc[f'r_grid_{m}_p_SMAP_CLSM']
            thres = 0.2
            ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

            res[col][res[col] < 0] = 0
            res[ref_col_ol][res[ref_col_ol] < 0] = 0
            res[ref_col_da][res[ref_col_da] < 0] = 0
            res[col][res[col] > 1] = 1
            res[ref_col_ol][res[ref_col_ol] > 1] = 1
            res[ref_col_da][res[ref_col_da] > 1] = 1

            if cnt == 0:
                im = plot_ease_img(res.reindex(ind_valid), ref_col_ol , title=tit, cmap='viridis', cbrange=cb_r, fontsize=fontsize, print_median=True, plot_cb=True)
            elif cnt == 1:
                res[f'{col}_diff'] = res[col] - res[ref_col_ol]
                im = plot_ease_img(res.reindex(ind_valid), f'{col}_diff' , title=tit, cmap=cc.cm.bjy, cbrange=cb_diff_ol, fontsize=fontsize, print_median=True, plot_cb=True)
            else:
                res[f'{col}_diff'] = res[col] - res[ref_col_da]
                im = plot_ease_img(res.reindex(ind_valid), f'{col}_diff' , title=tit, cmap=cc.cm.bjy, cbrange=cb_diff_da, fontsize=fontsize, print_median=True, plot_cb=True)

        f.savefig(dir_out / f'ascat_eval_{m}.png', dpi=300, bbox_inches='tight')
        plt.close()


def plot_filter_diagnostics(dir_out):

    fname = '/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/validation/filter_diagnostics.nc'

    fontsize = 14

    runs = ['OL_Pcorr', 'DA_Pcorr_4K'] + \
            [f'DA_Pcorr_{err}' for err in ['abs', 'anom_lst', 'anom_lt', 'anom_st']]
    titles = ['Open-loop', '4K constant', 'Total signal', 'Anomalies', 'LF signal', 'HF signal']

    ref = None

    iters = [0,2,4,6,5,7]

    with Dataset(fname) as ds:

        lons = ds.variables['lon'][:]
        lats = ds.variables['lat'][:]
        lons, lats = np.meshgrid(lons, lats)

        var = 'innov_autocorr_abs'
        cbrange = [0, 0.7]
        step = 0.2
        cmap = 'viridis'

        for spc in np.arange(4):

            f = plt.figure(figsize=(23, 8))

            for i, (it_tit, it) in enumerate(zip(titles,iters)):
                plt.subplot(2, 3, i+1)
                if ref is not None:
                    refit = iters[np.array(runs)==ref][0]
                    data = ds.variables[var][:,:,it,spc] - ds.variables[var][:,:,refit,spc]
                else:
                    data = ds.variables[var][:,:,it,spc]
                    plot_image(data, lats, lons,
                           cmap=cmap,
                           cbrange=cbrange,
                           fontsize = fontsize,
                           title=it_tit)

            f.subplots_adjust(hspace=0, wspace=0.05, bottom=0.05)

            pos1 = f.axes[-2].get_position()
            im1 = f.axes[0].collections[-1]
            ticks = np.arange(cbrange[0], cbrange[1]+1, step)

            cbar_ax = f.add_axes([pos1.x0, 0.03, pos1.x1-pos1.x0, 0.03])
            cbar = f.colorbar(im1, orientation='horizontal', cax=cbar_ax, ticks=ticks)
            for t in cbar.ax.get_xticklabels():
                t.set_fontsize(fontsize)

            fout = dir_out / f'{var}_spc{spc+1}.png'
            f.savefig(fout, dpi=300, bbox_inches='tight')
            plt.close()

            # plt.tight_layout()
            # plt.show()

def plot_perturbations(dir_out):

    root = Path('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/observation_perturbations/Pcorr')
    res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv',
                      index_col=0)

    pc = 'Pcorr'

    io = GEOSldas_io('ObsFcstAna')
    io2 = LDASsa_io('ObsFcstAna')

    lut = pd.read_csv(Paths().lut, index_col=0)
    ind = np.vectorize(io.grid.colrow2tilenum)(lut.ease2_col, lut.ease2_row, local=False)

    dtype, hdr, length = template_error_Tb40()

    f = plt.figure(figsize=(24, 12))

    fontsize=15
    cbrange = [0,8]
    cmap=cc.cm.bjy

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']
    titles = ['Total signal', 'Anomalies', 'LF signal', 'HF signal']

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

        plt.subplot(4,4,i+1)
        im = plot_ease_img2(imgA.reindex(ind).reindex(ind_valid),'err_Tbv', cbrange=cbrange, cmap=cmap, io=io, plot_cmap=False)
        plt.title(title, fontsize=fontsize)
        if mode == 'abs':
            plt.ylabel('$\hat{P}$ (V-pol, Asc.)', fontsize=fontsize)

        plt.subplot(4,4,i+5)
        plot_ease_img2(imgD.reindex(ind).reindex(ind_valid),'err_Tbv', cbrange=cbrange, cmap=cmap, io=io, plot_cmap=False)
        if mode == 'abs':
            plt.ylabel('$\hat{P}$ (V-pol, Dsc.)', fontsize=fontsize)

        plt.subplot(4,4,i+9)
        plot_ease_img2(imgA.reindex(ind).reindex(ind_valid),'err_Tbh', cbrange=cbrange, cmap=cmap, io=io, plot_cmap=False)
        if mode == 'abs':
            plt.ylabel('$\hat{P}$ (H-pol, Asc.)', fontsize=fontsize)

        plt.subplot(4,4,i+13)
        plot_ease_img2(imgD.reindex(ind).reindex(ind_valid),'err_Tbh', cbrange=cbrange, cmap=cmap, io=io, plot_cmap=False)
        if mode == 'abs':
            plt.ylabel('$\hat{P}$ (H-pol, Dsc.)', fontsize=fontsize)


    f.subplots_adjust(wspace=0.05, hspace=0, bottom=0.1)

    ticks = np.arange(cbrange[0], cbrange[1]+0.001, 2)

    pos = [plt.gcf().axes[i].get_position() for i in (7,11)]
    x1 = (pos[0].x0 + pos[0].x1) / 2
    x2 = (pos[1].x0 + pos[1].x1) / 2

    cbar_ax = f.add_axes([x1, 0.06, x2-x1, 0.03])
    cb = f.colorbar(im, orientation='horizontal', cax=cbar_ax, ticks=ticks)
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize-2)

    plt.savefig(dir_out / f'perturbations.png', dpi=300, bbox_inches='tight')
    plt.close()



def plot_sm_weight(dir_out):

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    figsize = (15, 2)
    fontsize = 12
    cb = [0, 1]

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']
    titles = ['Total signal', 'Anomalies', 'LF signal', 'HF signal']
    l = 'K$_{\Theta}$'

    f = plt.figure(figsize=figsize)

    n = 0
    pos = []
    for mode, title in zip(modes, titles):
        n += 1

        plt.subplot(1, 4, n)

        tag1 = 'ubrmse_grid_' + mode + '_m_CLSM_tc_ASCAT_SMAP_CLSM'
        tag2 = 'ubrmse_grid_' + mode + '_m_SMAP_tc_ASCAT_SMAP_CLSM'

        res['wght'] = res[tag1]**2 / (res[tag1]**2 + res[tag2]**2)

        r_asc_smap = res[f'r_grid_{mode}_p_ASCAT_SMAP']
        r_asc_clsm = res[f'r_grid_{mode}_p_ASCAT_CLSM']
        r_smap_clsm = res[f'r_grid_{mode}_p_SMAP_CLSM']
        thres = 0.2
        ind_valid = res[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

        im_r = plot_ease_img(res.reindex(ind_valid), 'wght', fontsize=fontsize, cbrange=cb, cmap='viridis')
        if (n == 2) | (n == 3):
            pos += [im_r.axes.get_position()]

        plt.title(title, fontsize=fontsize)
        if mode == 'abs':
            plt.ylabel(l, fontsize=fontsize)

    f.subplots_adjust(wspace=0.05, hspace=0, bottom=0.1)

    ticks = np.arange(cb[0], cb[1]+0.001, 0.2)

    x1 = (pos[0].x0 + pos[0].x1) / 2
    x2 = (pos[1].x0 + pos[1].x1) / 2

    cbar_ax = f.add_axes([x1, 0.03, x2-x1, 0.07])
    cb = f.colorbar(im_r, orientation='horizontal', cax=cbar_ax, ticks=ticks)
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize-2)

    fout = dir_out / 'k_sm.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tca_uncertainties(dir_out):

    sensors = ['ASCAT', 'SMAP', 'CLSM']

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    figsize = (15, 4)
    fontsize = 12
    cb = [0, 0.06]

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']
    titles = ['Total signal', 'Anomalies', 'LF signal', 'HF signal']
    labels = ['$\widehat{std}(\epsilon_{\Theta,smap})$', '$\widehat{std}(\epsilon_{\Theta,clsm})$']

    f = plt.figure(figsize=figsize)

    n = 0
    pos = []
    for s, l in zip(sensors[1::],labels):
        for mode, title in zip(modes, titles):
            n += 1

            plt.subplot(2, 4, n)

            tag = 'ubrmse_grid_' + mode + '_m_' + s + '_tc_ASCAT_SMAP_CLSM'

            r_asc_smap = res[f'r_grid_{mode}_p_ASCAT_SMAP']
            r_asc_clsm = res[f'r_grid_{mode}_p_ASCAT_CLSM']
            r_smap_clsm = res[f'r_grid_{mode}_p_SMAP_CLSM']
            thres = 0.2
            ind_valid = res[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

            im_r = plot_ease_img(res.reindex(ind_valid), tag, fontsize=fontsize, cbrange=cb, cmap='viridis')
            if (n == 6) | (n == 7):
                pos += [im_r.axes.get_position()]

            if s == 'SMAP':
                plt.title(title, fontsize=fontsize)
            if mode == 'abs':
                plt.ylabel(l, fontsize=fontsize)

    f.subplots_adjust(wspace=0.05, hspace=0, bottom=0.1)

    ticks = np.arange(cb[0], cb[1]+0.001, 0.02)

    x1 = (pos[0].x0 + pos[0].x1) / 2
    x2 = (pos[1].x0 + pos[1].x1) / 2

    cbar_ax = f.add_axes([x1, 0.07, x2-x1, 0.03])
    cb = f.colorbar(im_r, orientation='horizontal', cax=cbar_ax, ticks=ticks)
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize-2)

    fout = dir_out / 'tca_uncertainties.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def plot_freq_components():

    ds = GEOSldas_io('tavg3_1d_lnr_Nt', exp='NLv4_M36_US_DA_SMAP_Pcorr_abs').timeseries

    ts = ds['sm_rootzone'][:,30,100].to_pandas()

    clim = calc_climatology(ts)

    anom_lt = calc_anom(ts, mode='longterm')
    anom_st = calc_anom(ts, mode='shortterm')
    clim = calc_anom(ts, return_clim=True)

    df = pd.concat((ts, clim, anom_st, anom_lt), axis=1)
    df.columns = ['total signal', 'climatology', 'high-frequency signal', 'low-frequency signal']

    df.plot().legend(loc='upper left')
    plt.ylim(-0.04,0.24)

    plt.tight_layout()
    plt.show()



### --------------------------------------------------------------------------------------------------------------------
### --- Test graphics ---
### --------------------------------------------------------------------------------------------------------------------

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


def plot_tca_uncertainty_CI(dir_out):

    sensors = ['ASCAT', 'SMAP', 'CLSM']

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    figsize = (15, 2)
    fontsize = 12
    cb = [0, 0.04]

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']
    titles = ['Total signal', 'Anomalies', 'LF signal', 'HF signal']
    labels = ['80% CI (ASCAT)', '80% CI (SMAP)', '80% CI (CLSM)']

    f = plt.figure(figsize=figsize)

    n = 0
    pos = []
    for mode, title in zip(modes, titles):

        n += 1
        plt.subplot(1, 4, n)

        res['tmp'] = 0
        for s, l in zip(sensors[1::],labels):
            tag_u = 'ubrmse_grid_' + mode + '_u_' + s + '_tc_ASCAT_SMAP_CLSM'
            tag_l = 'ubrmse_grid_' + mode + '_l_' + s + '_tc_ASCAT_SMAP_CLSM'
            res['tmp'] += (res[tag_u] - res[tag_l])**2
        res['tmp'] **= (1/2)

        im_r = plot_ease_img(res, 'tmp', fontsize=fontsize, cbrange=cb, cmap='viridis')
        if (n == 2) | (n == 3):
            pos += [im_r.axes.get_position()]

        # if s == 'ASCAT':
        plt.title(title, fontsize=fontsize)
        if mode == 'abs':
            plt.ylabel('80% CI (combined)', fontsize=fontsize)

    f.subplots_adjust(wspace=0.05, hspace=0, bottom=0.1)

    ticks = np.arange(cb[0], cb[1]+0.001, 0.02)

    x1 = (pos[0].x0 + pos[0].x1) / 2
    x2 = (pos[1].x0 + pos[1].x1) / 2

    cbar_ax = f.add_axes([x1, 0.07, x2-x1, 0.03])
    cb = f.colorbar(im_r, orientation='horizontal', cax=cbar_ax, ticks=ticks)
    for t in cb.ax.get_xticklabels():
        t.set_fontsize(fontsize-2)

    fout = dir_out / 'tca_uncertainty_CI_combined.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

def plot_tca_uncertainty_ratio(dir_out):

    sensors = ['ASCAT', 'SMAP', 'CLSM']

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    figsize = (15, 2)
    fontsize = 12
    cb = [-10, 10]

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']
    titles = ['Total signal', 'Anomalies', 'LF signal', 'HF signal']

    f = plt.figure(figsize=figsize)

    n = 0
    pos = []
    for mode, title in zip(modes, titles):
        n += 1

        plt.subplot(1, 4, n)

        tagR = 'ubrmse_grid_' + mode + '_m_SMAP_tc_ASCAT_SMAP_CLSM'
        tagP = 'ubrmse_grid_' + mode + '_m_CLSM_tc_ASCAT_SMAP_CLSM'

        res['tmp'] = 10*np.log10(res[tagR]**2 / res[tagP]**2)

        im_r = plot_ease_img(res, 'tmp', fontsize=fontsize, cbrange=cb, cmap=cc.cm.bjy)
        if (n == 6) | (n == 7):
            pos += [im_r.axes.get_position()]

        plt.title(title, fontsize=fontsize)

    plot_centered_cbar(f, im_r, 4, fontsize=fontsize-2)

    fout = dir_out / 'tca_uncertainty_ratio.png'
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


def plot_ascat_eval_absolute(res_path, dir_out):

    runs = ['Pcorr_OL'] + [f'Pcorr_{err}' for err in ['4K','anom_lst', 'LTST', 'anom_lt_ScDY','anom_st_ScYH']]
    # runs = ['noPcorr_OL'] + [f'noPcorr_{err}' for err in ['4K','anom_lst', 'LTST', 'anom_lt','anom_st_ScYH']]
    titles = ['R$_{OL}$', 'R$_{4K}$', 'R$_{LF + HF}$', 'R$_{LF, HF}$', 'R$_{LF}$', 'R$_{HF}$']

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)

    res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)
    # res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/noPcorr/result.csv', index_col=0)

    modes = ['anom_lst', 'anom_lt','anom_st']

    cb_r = [0.2, 1]
    # cb_diff1 = [-0.2, 0.2]
    # cb_diff2 = [-0.04, 0.04]

    fontsize = 14

    f = plt.figure(figsize=(24,7))

    for i, m in enumerate(modes):

        for cnt, (r, tit) in enumerate(zip(runs,titles)):

            ax = plt.subplot(3, 6, i * 6 + cnt + 1)

            col = f'ana_r_corr_{r}_{m}'

            res[col][res[col] < 0] = 0
            res[col][res[col] > 1] = 1

            if i == 0:
                title = tit
            else:
                title = ''

            if cnt == 0:
                if i == 0:
                    ylabel = 'Anomalies'
                elif i == 1:
                    ylabel = 'LF signal'
                else:
                    ylabel = 'HF signal'
            else:
                ylabel = ''

            r_asc_smap = res_tc[f'r_grid_{m}_p_ASCAT_SMAP']
            r_asc_clsm = res_tc[f'r_grid_{m}_p_ASCAT_CLSM']
            r_smap_clsm = res_tc[f'r_grid_{m}_p_SMAP_CLSM']
            thres = 0.2
            ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

            # if j < 2:
            im = plot_ease_img(res.reindex(ind_valid), col , title=title, cmap='viridis', cbrange=cb_r, fontsize=fontsize, print_median=True, plot_cb=False)
            ax.set_ylabel(ylabel)
            # else:
            #     cbr = cb_diff1 if m != 'anom_st' else cb_diff2
            #     im = plot_ease_img(res.reindex(ind_valid), col , title=title, cmap=cc.cm.bjy, cbrange=cbr, fontsize=fontsize, print_median=True, plot_cb=False)

        plot_centered_cbar(f, im, 6, fontsize=fontsize-2)

    f.savefig(dir_out / f'ascat_eval_abs_Pcorr.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ascat_eval_relative(res_path, dir_out):

    ref = 'Pcorr_4K'
    if '4K' in ref:
        lab = 'R$_{TC}$ - R$_{4K}$'
    else:
        lab = 'R$_{TC}$ - R$_{OL}$'

    runs = [f'Pcorr_{err}' for err in ['anom_lst', 'anom_lt_ScDY','anom_st_ScYH', 'LTST']]
    # runs = ['noPcorr_OL'] + [f'noPcorr_{err}' for err in ['4K','anom_lst', 'LTST', 'anom_lt','anom_st_ScYH']]
    titles = ['Anomaly assimilation', 'LF assimilation', 'HF assimilation', 'LF + HF assimilation']
    # labels = [f'{lab} (Anom.)', f'{lab} (LF)', f'{lab} (HF)']
    labels = [f'Anomaly skill', f'LF skill', f'HF skill']

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)

    res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)
    # res_tc = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/noPcorr/result.csv', index_col=0)

    modes = ['anom_lst', 'anom_lt','anom_st']

    cb_r = [-0.2, 0.2]

    fontsize = 14

    f = plt.figure(figsize=(20,8))

    for i, (m, label) in enumerate(zip(modes,labels)):

        ref_col = f'ana_r_corr_{ref}_{m}'

        res[ref_col][res[ref_col] < 0] = 0
        res[ref_col][res[ref_col] > 1] = 1

        for cnt, (r, tit) in enumerate(zip(runs,titles)):

            ax = plt.subplot(3, 4, i * 4 + cnt + 1)

            col = f'ana_r_corr_{r}_{m}'

            res[col][res[col] < 0] = 0
            res[col][res[col] > 1] = 1

            res['diff'] = res[col] - res[ref_col]

            if i == 0:
                title = tit
            else:
                title = ''

            if cnt == 0:
                ylabel = label
            else:
                ylabel = ''

            r_asc_smap = res_tc[f'r_grid_{m}_p_ASCAT_SMAP']
            r_asc_clsm = res_tc[f'r_grid_{m}_p_ASCAT_CLSM']
            r_smap_clsm = res_tc[f'r_grid_{m}_p_SMAP_CLSM']
            thres = 0.2
            ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

            # if j < 2:
            im = plot_ease_img(res.reindex(ind_valid), 'diff' , title=title, cmap=cc.cm.bjy, cbrange=cb_r, fontsize=fontsize, print_median=True, plot_cb=False)
            ax.set_ylabel(ylabel)
            # else:
            #     cbr = cb_diff1 if m != 'anom_st' else cb_diff2
            #     im = plot_ease_img(res.reindex(ind_valid), col , title=title, cmap=cc.cm.bjy, cbrange=cbr, fontsize=fontsize, print_median=True, plot_cb=False)

        plot_centered_cbar(f, im, 4, fontsize=fontsize-2)

    f.savefig(dir_out / f'ascat_eval_rel_{ref}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ismn_statistics_new(res_path, dir_out):

    res = pd.read_csv(res_path / 'insitu_TCA.csv', index_col=0)
    res.index = res.network
    res.drop('network', axis='columns', inplace=True)

    variables = ['sm_surface', 'sm_rootzone']
    var_labels = ['surface', 'root-zone']

    pc = 'Pcorr'

    runs = [f'OL_{pc}'] + [f'DA_{pc}_{err}' for err in ['4K', 'anom_lst', 'anom_lt', 'anom_st']]
    run_labels = ['Open-loop', '4K benchmark', 'Anomalies', 'LF signal', 'HF signal']

    networks  = ['SCAN', 'USCRN']
    res = res.loc[res.index.isin(networks),:]


    modes = ['anom_lst', 'anom_lt', 'anom_st']

    var = 'sm_surface'

    f = plt.figure(figsize=(15,12))

    lim = [-0.2,0.2]
    lim2 = [-0.05,0.05]

    for i, mode in enumerate(modes):

        run_ol = f'OL_{pc}_{mode}'
        run_4k = f'DA_{pc}_4K_{mode}'
        run_tc = f'DA_{pc}_{mode}_{mode}'

        ax = plt.subplot(3,3,i+1)
        res['col'] = res[f'R_model_insitu_{run_4k}_{var}'] - res[f'R_model_insitu_{run_ol}_{var}']
        res['col'].hist(bins=15, grid=False, ax=ax, range=lim)
        plt.title(f'Corr: 4K - OL ({mode})')
        plt.xlim(lim)
        plt.axvline(color='black', linestyle='--', linewidth=1)


        ax = plt.subplot(3,3,i+4)
        res['col'] = res[f'R_model_insitu_{run_tc}_{var}'] - res[f'R_model_insitu_{run_ol}_{var}']
        res['col'].hist(bins=15, grid=False, ax=ax, range=lim)
        plt.title(f'Corr: TC - OL ({mode})')
        plt.xlim(lim)
        plt.axvline(color='black', linestyle='--', linewidth=1)

        ax = plt.subplot(3,3,i+7)
        res['col'] = res[f'R_model_insitu_{run_tc}_{var}'] - res[f'R_model_insitu_{run_4k}_{var}']
        res['col'].hist(bins=15, grid=False, ax=ax, range=lim2)
        plt.title(f'Corr: TC - 4K ({mode})')
        plt.xlim(lim2)
        plt.axvline(color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()

def plot_ascat_statistics_new(res_path, dir_out):

    res = pd.read_csv(res_path / 'ascat_eval.csv', index_col=0)
    res_tc = pd.read_csv(
        '/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    pc = 'Pcorr'

    run_labels = ['Open-loop', '4K benchmark', 'Anomalies', 'LF signal', 'HF signal']

    modes = ['anom_lst', 'anom_lt', 'anom_st']

    f = plt.figure(figsize=(15,12))

    lim = [-0.2,0.2]
    lim2 = [-0.05,0.05]

    for i, mode in enumerate(modes):

        col_ol = f'ana_r_corr_OL_{pc}_{mode}'
        col_4k = f'ana_r_corr_DA_{pc}_4K_{mode}'
        col_tc = f'ana_r_corr_DA_{pc}_{mode}_{mode}'

        r_asc_smap = res_tc[f'r_grid_{mode}_p_ASCAT_SMAP']
        r_asc_clsm = res_tc[f'r_grid_{mode}_p_ASCAT_CLSM']
        r_smap_clsm = res_tc[f'r_grid_{mode}_p_SMAP_CLSM']
        thres = 0.2
        ind_valid = res_tc[(r_asc_smap > thres) & (r_asc_smap > thres) & (r_asc_smap > thres)].index

        res[col_ol][res[col_ol] < 0] = 0
        res[col_4k][res[col_4k] < 0] = 0
        res[col_tc][res[col_tc] < 0] = 0
        res[col_ol][res[col_ol] > 1] = 1
        res[col_4k][res[col_4k] > 1] = 1
        res[col_tc][res[col_tc] > 1] = 1

        ax = plt.subplot(3,3,i+1)
        res['col'] = res[col_4k] - res[col_ol]
        res['col'].reindex(ind_valid).hist(bins=15, grid=False, ax=ax, range=lim)
        plt.title(f'Corr: 4K - OL ({mode})')
        plt.xlim(lim)
        plt.axvline(color='black', linestyle='--', linewidth=1)


        ax = plt.subplot(3,3,i+4)
        res['col'] = res[col_tc] - res[col_ol]
        res['col'].reindex(ind_valid).hist(bins=15, grid=False, ax=ax, range=lim)
        plt.title(f'Corr: TC - OL ({mode})')
        plt.xlim(lim)
        plt.axvline(color='black', linestyle='--', linewidth=1)

        ax = plt.subplot(3,3,i+7)
        res['col'] = res[col_tc] - res[col_4k]
        res['col'].reindex(ind_valid).hist(bins=15, grid=False, ax=ax, range=lim2)
        plt.title(f'Corr: TC - 4K ({mode})')
        plt.xlim(lim2)
        plt.axvline(color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    # res_path_hflf = Path(f'/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/validation')
    # res_path_hf = Path(f'/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/validation_ScYH')
    # res_path_lf = Path(f'/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/validation_ScDY')

    res_path = Path(r"D:\_KUL_backup_2022\Documents\work_KUL\MadKF\CLSM\SM_err_ratio\GEOSldas\validation_all")
    dir_out = Path(r'H:\work\SMAP_DA_paper\plots')
    if not dir_out.exists():
        Path.mkdir(dir_out, parents=True)

    # if '_scyh' in str(dir_out):
    #     res_path = res_path_hf
    # elif '_scdy' in str(dir_out):
    #     res_path = res_path_lf
    # else:
    #     res_path = res_path_hflf

    # plot_freq_components()
    # plot_potential_skillgain_simple(dir_out)
    # plot_actual_skillgain(res_path, dir_out)
    # plot_potential_skillgain_decomposed(dir_out)
    plot_station_locations(dir_out)
    # plot_ismn_vs_ascat(res_path, dir_out)
    # plot_ismn_statistics(res_path, dir_out)
    # plot_ascat_eval_uncorrected(res_path, dir_out)
    # plot_ascat_eval(res_path, dir_out)
    # plot_filter_diagnostics(dir_out)
    # plot_perturbations(dir_out)
    # plot_sm_weight(dir_out)
    # plot_tca_uncertainties(dir_out)
    # plot_tca_uncertainty_CI(dir_out)
    # plot_tca_uncertainty_ratio(dir_out)
    # plot_ensvar_ratio(dir_out)
    # plot_ascat_eval_absolute(res_path, dir_out)
    # plot_ascat_eval_relative(res_path, dir_out)
    # plot_ismn_statistics_new(res_path, dir_out)
    # plot_ascat_statistics_new(res_path, dir_out)
