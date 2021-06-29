import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context('talk', font_scale=0.8)

from pyldas.interface import LDAS_io
from pyldas.templates import template_error_Tb40

from validation_good_practice.plots import plot_ease_img
from validation_good_practice.ancillary.grid import Paths
from myprojects.experiments.MadKF.CLSM.ensemble_covariance import fill_gaps, plot_ease_img2


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

    froot = Path('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio/')
    fbase = 'SMOS_fit_Tb_'

    pc = 'noPcorr'

    io = LDAS_io()

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

    # ks = {'anom_lt': [(1,0)],
    #       'anom_st':[(0.8,0), (1,0), (1.5,0)]}

    for mode, k in ks.items():

        tag_r = f'ubrmse_grid_{mode}_m_SMAP_tc_ASCAT_SMAP_CLSM'
        tag_p = f'ubrmse_grid_{mode}_m_CLSM_tc_ASCAT_SMAP_CLSM'

        for i, (k_m, k_a) in enumerate(k):

            dir_out = froot / 'observation_perturbations' / pc / mode
            if not dir_out.exists():
                Path.mkdir(dir_out, parents=True)

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
            edate = np.array([2020, 4, 1, 0, 0])
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

    root = Path('/Users/u0116961/Documents/work/MadKF/CLSM/SM_err_ratio')

    lut = pd.read_csv(Paths().lut, index_col=0)
    ind = np.vectorize(LDAS_io().grid.colrow2tilenum)(lut.ease2_col, lut.ease2_row, local=False)

    for mode in ['anom_lt', 'anom_st']:
        for i in np.arange(1,2):

            fA = root / 'observation_perturbations' / f'{mode}_{i}' / 'SMOS_fit_Tb_A.bin'
            fD = root / 'observation_perturbations' / f'{mode}_{i}' / 'SMOS_fit_Tb_D.bin'

            dir_out = root / 'plots' / 'obs_pert'
            if not dir_out.exists():
                Path.mkdir(dir_out)

            dtype, hdr, length = template_error_Tb40()

            io = LDAS_io('ObsFcstAna')

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

            plt.savefig(dir_out / f'{mode}_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()

if __name__=='__main__':

    # plot_kalman_gains()
    create_observation_perturbations()
    # plot_perturbations()
