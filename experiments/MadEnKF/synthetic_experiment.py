import os
import platform

import numpy as np
import pandas as pd

from copy import deepcopy

from multiprocessing import Pool

from pyapi.api import API
from pyass.filter import MadKF, EnKF

from myprojects.readers.gen_syn_data import generate_soil_moisture

def main():

    n_runs = 24*20

    SNR_R = np.random.uniform(0.25, 4, n_runs)
    SNR_P = np.random.uniform(0.25, 4, n_runs)
    gamma = np.random.uniform(0.55, 0.95, n_runs)
    H_true = np.random.uniform(0.5, 2, n_runs)
    thread = np.arange(n_runs)

    args = zip(SNR_R, SNR_P, gamma, H_true, thread)

    Pool(4).map(run, args)

    # args = [0.8, 1.2, 0.85, 2, 0]
    # run(args)

def run(args):

    if platform.system() == 'Windows':
        root = r'D:\work\MadKF\synthetic_experiment'
    else:
        root = '/data/leuven/320/vsc32046/projects/MadKF/synthetic_experiment'

    SNR_R = args[0]
    SNR_P = args[1]
    gamma = args[2]
    H_true = args[3]
    thread = args[4]

    fname = os.path.join(root,'result_%i.csv' % thread)

    n = 1500

    api = API(gamma=gamma)

    n_ens_arr = [10, 30, 50, 70]
    n_iter_arr = [5, 7, 9, 11, 13, 15]

    idx = thread * len(n_ens_arr) * len(n_iter_arr) - 1
    for n_ens in n_ens_arr:
        for n_iter in n_iter_arr:

            idx += 1

            sm_true, precip_true = generate_soil_moisture(n, gamma=gamma, scale=7)

            R = sm_true.var()/SNR_R
            Q = sm_true.var()/SNR_P * (1-gamma**2)

            obs_err = np.random.normal(0, np.sqrt(R), n)
            forc_err = np.random.normal(0, np.sqrt(Q), n)

            obs = (sm_true + obs_err) / H_true
            forcing = precip_true + forc_err

            # --- get OL ts & error ---
            OL = np.full(n, np.nan)

            model = deepcopy(api)
            for t, f in enumerate(forcing):
                x = model.step(f)
                OL[t] = x

            R_true = obs_err.var() / H_true**2
            Q_true = forc_err.var()
            P_true = Q_true / (1 - gamma ** 2)
            P_OL_true = (np.mean((sm_true - OL)**2))

            x_ana_madkf, P_ana_madkf, R_est_madkf, Q_est_madkf, H_est_madkf, R_innov_madkf, checkvar_madkf, K = MadKF(api, forcing, obs, n_ens=n_ens, n_iter=n_iter)

            P_ana_true_madkf = np.mean((sm_true - x_ana_madkf) ** 2)
            P_ana_est_madkf = P_ana_madkf.mean()

            if n_iter == n_iter_arr[0]:
                R_est_rmsd = (np.mean((OL - H_true * obs) ** 2) - P_true) / H_true ** 2

                forc_pert = ['normal', 'additive', Q_true]
                obs_pert = ['normal', 'additive', R_est_rmsd]
                x_ana_enkf_rmsd, P_ana_enkf_rmsd, R_innov_enkf_rmsd, checkvar_enkf_rmsd, K = EnKF(api, forcing, obs, forc_pert, obs_pert, H=H_true, n_ens=n_ens)

                P_ana_est_enkf_rmsd = P_ana_enkf_rmsd.mean()
                P_ana_true_enkf_rmsd = np.mean((sm_true - x_ana_enkf_rmsd) ** 2)

                obs_pert = ['normal', 'additive', R_true]
                x_ana_enkf_true, P_ana_enkf_true, R_innov_enkf_true, checkvar_enkf_true, K = EnKF(api, forcing, obs, forc_pert, obs_pert, H=H_true, n_ens=n_ens)

                P_ana_est_enkf_true = P_ana_enkf_true.mean()
                P_ana_true_enkf_true = np.mean((sm_true - x_ana_enkf_true) ** 2)
            else:
                R_est_rmsd = np.nan

                x_ana_enkf_rmsd = np.full(len(forcing), np.nan)
                P_ana_est_enkf_rmsd = np.nan
                P_ana_true_enkf_rmsd = np.nan
                checkvar_enkf_rmsd = np.nan
                R_innov_enkf_rmsd = np.nan

                x_ana_enkf_true = np.full(len(forcing), np.nan)
                P_ana_est_enkf_true = np.nan
                P_ana_true_enkf_true = np.nan
                checkvar_enkf_true = np.nan
                R_innov_enkf_true = np.nan


            corr = pd.DataFrame({'truth': sm_true,
                                 'obs': obs,
                                 'OL': OL,
                                 'ana_madkf': x_ana_madkf,
                                 'ana_enkf_rmsd': x_ana_enkf_rmsd,
                                 'ana_enkf_true': x_ana_enkf_true}).corr()

            result = pd.DataFrame({'n_ens': n_ens,
                                   'n_iter': n_iter,
                                   'R_true': R_true,
                                   'Q_true': Q_true,
                                   'P_true': P_true,
                                   'R_est_rmsd': R_est_rmsd,
                                   'R_est_madkf': R_est_madkf,
                                   'Q_est_madkf': Q_est_madkf,
                                   'H_est_madkf': H_est_madkf,
                                   'H_true': H_true,
                                   'checkvar_madkf': checkvar_madkf,
                                   'checkvar_enkf_rmsd': checkvar_enkf_rmsd,
                                   'checkvar_enkf_true': checkvar_enkf_true,
                                   'R_innov_madkf': R_innov_madkf,
                                   'R_innov_enkf_rmsd': R_innov_enkf_rmsd,
                                   'R_innov_enkf_true': R_innov_enkf_true,
                                   'P_OL_true': P_OL_true,
                                   'P_ana_est_madkf': P_ana_est_madkf,
                                   'P_ana_true_madkf': P_ana_true_madkf,
                                   'P_ana_est_enkf_rmsd': P_ana_est_enkf_rmsd,
                                   'P_ana_true_enkf_rmsd': P_ana_true_enkf_rmsd,
                                   'P_ana_est_enkf_true': P_ana_est_enkf_true,
                                   'P_ana_true_enkf_true': P_ana_true_enkf_true,
                                   'corr_obs': corr['truth']['obs'],
                                   'corr_OL': corr['truth']['OL'],
                                   'corr_ana_madkf': corr['truth']['ana_madkf'],
                                   'corr_ana_enkf_rmsd': corr['truth']['ana_enkf_rmsd'],
                                   'corr_ana_enkf_true': corr['truth']['ana_enkf_true']}, index=(idx,))

            if (os.path.isfile(fname) == False):
                result.to_csv(fname, float_format='%0.4f')
            else:
                result.to_csv(fname, float_format='%0.4f', mode='a', header=False)

            print 'thread: %i, run: %i' % (thread, idx)

if __name__=='__main__':
    main()

