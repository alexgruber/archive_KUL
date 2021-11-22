import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context('talk', font_scale=0.8)
import colorcet as cc

from pyldas.interface import GEOSldas_io, LDASsa_io
from pyldas.templates import template_error_Tb40

from validation_good_practice.plots import plot_ease_img
from validation_good_practice.ancillary.grid import Paths
from myprojects.experiments.MadKF.CLSM.ensemble_covariance import fill_gaps, plot_ease_img2

def plot_TCA_reliability():

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']

    f = plt.figure(figsize=(15,8))

    cmap = mpl.cm.get_cmap("magma", 8)

    for i, mode in enumerate(modes):

        p_a_s = res[f'p_grid_{mode}_p_ASCAT_SMAP']
        p_a_c = res[f'p_grid_{mode}_p_ASCAT_CLSM']
        p_s_c = res[f'p_grid_{mode}_p_SMAP_CLSM']
        r_a_s = res[f'r_grid_{mode}_p_ASCAT_SMAP']
        r_a_c = res[f'r_grid_{mode}_p_ASCAT_CLSM']
        r_s_c = res[f'r_grid_{mode}_p_SMAP_CLSM']

        res[f'p_mask_{mode}'] = ((p_a_s <= 0.05)&(r_a_s > 0.2)) * 1 + \
                                ((p_a_c <= 0.05)&(r_a_c > 0.2)) * 2 + \
                                ((p_s_c <= 0.05)&(r_s_c > 0.2)) * 4

        plt.subplot(2, 2, i+1)
        plot_ease_img(res, f'p_mask_{mode}', fontsize=12, cbrange=[0,8], cmap=cmap, log_scale=False, title=mode, plot_cb=True)



    plt.tight_layout()
    plt.show()

    # fout = f'/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/plots/skillgain_pot_Pcorr_simple.png'
    # f.savefig(fout, dpi=300, bbox_inches='tight')
    # plt.close()


def plot_potential_skillgain():

    res = pd.read_csv('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/sm_validation/Pcorr/result.csv', index_col=0)

    modes = ['abs', 'anom_lst', 'anom_lt', 'anom_st']

    f = plt.figure(figsize=(15,8))

    for i, mode in enumerate(modes):

        R = res[f'ubrmse_grid_{mode}_m_SMAP_tc_ASCAT_SMAP_CLSM']**2
        P = res[f'ubrmse_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM']**2

        R2 = res[f'r2_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM']

        K = P / (R + P)
        NSR = (1 - R2) / R2

        R2upd = 1 / (1 + (1 - K) * NSR)

        res['gain_pot'] = R2upd - R2

        plt.subplot(2, 2, i+1)
        plot_ease_img(res, 'gain_pot', fontsize=12, cbrange=[-0.2,0.2], cmap=cc.cm.bjy, log_scale=False, title=mode, plot_cb=True)


    fout = f'/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/plots/skillgain_pot_Pcorr_simple.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

def plot_potential_skillgain_decomposed():

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


    # Baseline estimates
    R2 = res[f'r2_grid_abs_m_CLSM_tc_ASCAT_SMAP_CLSM']
    SNR = R2 / (1 - R2)
    SIG = SNR * P_abs

    modes = ['abs', 'anom_lt', 'anom_lst', 'anom_st']
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

        r_c_l = 0.2
        r_c_s = 0.8
        r_l_s = 0.2

        R12 = r_c_l * np.sqrt(R11 * R22)
        R13 = r_c_s * np.sqrt(R11 * R33)
        R23 = r_l_s * np.sqrt(R22 * R33)
        P12 = r_c_l * np.sqrt(P11 * P22)
        P13 = r_c_s * np.sqrt(P11 * P33)
        P23 = r_l_s * np.sqrt(P22 * P33)

        S = np.matrix([[R11, R12, R13, 0,   0,   0  ],
                       [R12, R22, R23, 0,   0,   0  ],
                       [R13, R23, R33, 0,   0,   0  ],
                       [0,   0,   0,   P11, P12, P13],
                       [0,   0,   0,   P12, P22, P23],
                       [0,   0,   0,   P13, P23, P33]])

        for mode in modes:

            P = res.loc[i, f'ubrmse_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM'] ** 2
            R = res.loc[i, f'ubrmse_grid_{mode}_m_SMAP_tc_ASCAT_SMAP_CLSM'] ** 2
            K = P / (R + P)

            A = np.matrix([K, K, K, 1-K, 1-K, 1-K])
            P_upd = (A * S * A.T)[0,0]

            NSR_upd = P_upd / SIG.loc[i]
            R2upd = 1 / (1 + NSR_upd)

            result.loc[i, mode] = R2upd - R2.loc[i]

    f = plt.figure(figsize=(15,8))
    for i, mode in enumerate(modes):
        plt.subplot(2, 2, i+1)
        plot_ease_img(result, mode, fontsize=12, cbrange=[-0.2,0.2], cmap=cc.cm.bjy, log_scale=False, title=mode, plot_cb=True)

    fout = f'/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/plots/skillgain_pot_Pcorr_corr_asym.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def plot_kalman_gains():

    res = pd.read_csv('/Users/u0116961/Documents/work/validation_good_practice/CI80/ASCAT_SMAP_CLSM/result.csv', index_col=0)

    # k_repr_abs = [(1,0), (2,0), (3,0), (1,0.01), (1,0.02)]
    # k_repr_lt = [(1,0), (2,0), (3,0), (1,0.01), (1,0.02)]
    # k_repr_lst = [(1,0), (2,0), (3,0), (1,0.01), (1,0.02)]
    # k_repr_st = [(1,0), (1.5,0), (2.5,0), (1,0.005), (1,0.01)]
    k_repr_abs = [(0.8,0), (1,0), (1.5,0)]
    k_repr_lt = [(0.8,0), (1,0), (1.5,0)]
    k_repr_lst = [(0.8,0), (1,0), (1.5,0)]
    k_repr_st = [(0.8,0), (1,0), (1.5,0)]

    modes = ['abs', 'anom_lt', 'anom_lst', 'anom_st']

    f = plt.figure(figsize=(23,10))

    for j, (mode, k_repr) in enumerate(zip(modes,[k_repr_abs,k_repr_lt,k_repr_lst,k_repr_st])):
        for i, k in enumerate(k_repr):

            k_m, k_a = k
            tag_r = f'ubrmse_grid_{mode}_m_SMAP_tc_ASCAT_SMAP_CLSM'
            tag_p = f'ubrmse_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM'

            res['tmp_tag'] = 1 / (1 + (res[tag_r] * k_m + k_a)**2 / res[tag_p] ** 2)

            res.loc[res['tmp_tag']<0.5,'tmp_tag'] **= 1.75

            plt.subplot(3, 4, i*4 + 1 + j)
            plot_ease_img(res, 'tmp_tag', fontsize=12, cbrange=[0,1], cmap='seismic_r', log_scale=False, title=f'{mode}: k_m = {k_m} / k_a = {k_a}', plot_cb=True)


    fout = f'/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/plots/obs_weight.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()


def plot_K_vs_r_p_ratio():

    R = np.linspace(0.1,10,500)

    K = 1 / (1 + R)

    sns.lineplot(R, K)
    plt.xscale('log')

    plt.xlabel('R / P')
    plt.ylabel('Kalman gain')

    plt.show()

def create_observation_perturbations():

    froot = Path('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas')
    fbase = 'SMOS_fit_Tb_'

    pc = 'Pcorr'

    io = GEOSldas_io()

    ensvar = pd.read_csv(froot / 'ens_vars' / pc / 'ens_var.csv', index_col=0)
    obs_err = ensvar[['col', 'row']]
    obs_err.loc[:, 'tile_id'] = io.grid.tilecoord.loc[obs_err.index, 'tile_id'].values

    tc_res = pd.read_csv(froot / 'sm_validation' / pc / 'result.csv', index_col=0)
    tc_res.index = np.vectorize(io.grid.colrow2tilenum)(tc_res.col.values.astype('int'), tc_res.row.values.astype('int'), local=False)
    tc_res = tc_res.reindex(obs_err.index)

    ks = {'abs': [(1,0)],
          'anom_lt': [(1,0)],
          'anom_lst': [(1,0)],
          'anom_st':[(1,0)]}

    # ks = {'abs': [(1,0)]}

    for mode, k in ks.items():

        tag_r = f'ubrmse_grid_{mode}_m_SMAP_tc_ASCAT_SMAP_CLSM'
        tag_p = f'ubrmse_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM'

        dir_out = froot / 'observation_perturbations' / pc / mode
        if not dir_out.exists():
            Path.mkdir(dir_out, parents=True)

        for i, (k_m, k_a) in enumerate(k):

            scl = (tc_res[tag_r] * k_m + k_a)**2 / tc_res[tag_p]**2

            # scl = tc_res[tag_r]**2 / tc_res[tag_p]**2
            # tmp = (1 + scl) ** (-1)
            # tmp[tmp < 0.5] **= 1.75
            # scl = tmp ** (-1.75) - 1

            for spc in np.arange(1, 5):
                obs_err.loc[:, 'obs_var_spc%i' % spc] = (ensvar['fcst_var_spc%i' % spc] * scl)**0.5
                obs_err.loc[(obs_err['obs_var_spc%i' % spc] < 0.1), 'obs_var_spc%i' % spc] = 0.1
                # obs_err.loc[(obs_err['obs_var_spc%i' % spc] > 1600), 'obs_var_spc%i' % spc] = 1600
                obs_err.loc[:, 'obs_var_spc%i' % spc] = fill_gaps(obs_err, 'obs_var_spc%i' % spc, smooth=False, grid=io.grid)['obs_var_spc%i' % spc]

            dtype = template_error_Tb40()[0]

            angles = np.array([40., ])
            orbits = ['A', 'D']

            template = pd.DataFrame(columns=dtype.names).astype('float32')
            template['lon'] = io.grid.tilecoord['com_lon'].values.astype('float32')
            template['lat'] = io.grid.tilecoord['com_lat'].values.astype('float32')
            template.index += 1

            modes = np.array([0, 0])
            sdate = np.array([2015, 4, 1, 0, 0])
            edate = np.array([2021, 4, 1, 0, 0])
            lengths = np.array([len(template), len(angles)])  # tiles, incidence angles, whatever

            # ----- write output files -----
            for orb in orbits:
                # !!! inconsistent with the definition in the obs_paramfile (species) !!!
                modes[0] = 1 if orb == 'A' else 0

                res = template.copy()

                spc = 0 if orb == 'A' else 1
                res.loc[:, 'err_Tbh'] = obs_err.loc[res.index, 'obs_var_spc%i' % (spc + 1)].values

                spc = 2 if orb == 'A' else 3
                res.loc[:, 'err_Tbv'] = obs_err.loc[res.index, 'obs_var_spc%i' % (spc + 1)].values

                fname = dir_out / (fbase + orb + '.bin')

                with open(fname, 'wb') as fid:
                    io.write_fortran_block(fid, modes)
                    io.write_fortran_block(fid, sdate)
                    io.write_fortran_block(fid, edate)
                    io.write_fortran_block(fid, lengths)
                    io.write_fortran_block(fid, angles)

                    for f in res.columns.values:
                        io.write_fortran_block(fid, res[f].values)

def plot_perturbations():

    root = Path('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/GEOSldas/')

    pc = 'noPcorr'

    io = GEOSldas_io('ObsFcstAna')

    lut = pd.read_csv(Paths().lut, index_col=0)
    ind = np.vectorize(io.grid.colrow2tilenum)(lut.ease2_col, lut.ease2_row, local=False)

    for mode in ['abs', 'anom_lst', 'anom_lt', 'anom_st']:
    # for mode in ['abs']:

        fA = root / 'observation_perturbations' / f'{pc}' / f'{mode}' / 'SMOS_fit_Tb_A.bin'
        fD = root / 'observation_perturbations' / f'{pc}' / f'{mode}' / 'SMOS_fit_Tb_D.bin'

        dir_out = root / 'plots' / 'obs_pert' / f'{pc}'
        if not dir_out.exists():
            Path.mkdir(dir_out, parents=True)

        dtype, hdr, length = template_error_Tb40()

        imgA = io.read_fortran_binary(fA, dtype, hdr=hdr, length=length)
        imgD = io.read_fortran_binary(fD, dtype, hdr=hdr, length=length)

        imgA.index += 1
        imgD.index += 1

        cbrange = [0,15]

        plt.figure(figsize=(19, 11))

        plt.subplot(221)
        plot_ease_img2(imgA.reindex(ind),'err_Tbh', cbrange=cbrange, title='H-pol (Asc.)', io=io)
        plt.subplot(222)
        plot_ease_img2(imgA.reindex(ind),'err_Tbv', cbrange=cbrange, title='V-pol (Asc.)', io=io)
        plt.subplot(223)
        plot_ease_img2(imgD.reindex(ind),'err_Tbh', cbrange=cbrange, title='H-pol (Dsc.)', io=io)
        plt.subplot(224)
        plot_ease_img2(imgD.reindex(ind),'err_Tbv', cbrange=cbrange, title='V-pol (Dsc.)', io=io)

        plt.savefig(dir_out / f'{mode}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__=='__main__':

    # plot_kalman_gains()
    create_observation_perturbations()
    # plot_perturbations()

    # plot_K_vs_r_p_ratio()

    # plot_TCA_reliability()

    # plot_potential_skillgain()
    # plot_potential_skillgain_decomposed()
