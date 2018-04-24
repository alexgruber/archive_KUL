

import os
import timeit
import platform

import numpy as np
import pandas as pd

from copy import deepcopy

from multiprocessing import Pool

from pyapi.api import API
from pyass.filter import MadEnKF

from myprojects.readers.gen_syn_data import generate_soil_moisture

def main():

    t = timeit.default_timer()

    n_runs = 48*6

    SNR_R = np.random.uniform(0.25, 4, n_runs)
    SNR_P = np.random.uniform(0.25, 4, n_runs)
    gamma = np.random.uniform(0.55, 0.95, n_runs)
    H_true = np.random.uniform(0.5, 2, n_runs)
    thread = np.arange(n_runs)

    args = zip(SNR_R, SNR_P, gamma, H_true, thread)

    Pool(48).map(run, args)

    print timeit.default_timer() - t

def run(args):

    if platform.system() == 'Windows':
        root = r'D:\work\API\MadEnKF\synthetic_experiment' + '\\'
    else:
        root = '/scratch/leuven/320/vsc32046/output/MadEnKF/synthetic_experiment/'

    SNR_R = args[0]
    SNR_P = args[1]
    gamma = args[2]
    H_true = args[3]
    thread = args[4]

    fname = root + 'result_%i.csv' % thread

    n = 1500

    api = API(gamma=gamma)

    n_ens_arr = [10, 30, 50, 70]
    n_iter_arr = [5, 7, 9, 11, 13, 15]

    idx = thread * len(n_ens_arr) * len(n_iter_arr) - 1
    for n_ens in n_ens_arr:
        for n_iter in n_iter_arr:

            t = timeit.default_timer()

            idx += 1

            sm_true, precip_true = generate_soil_moisture(n, gamma=gamma, scale=7)

            R = sm_true.var()/SNR_R
            Q = sm_true.var()/SNR_P * (1-gamma**2)

            obs_err = np.random.normal(0, np.sqrt(R), n)
            forc_err = np.random.normal(0, np.sqrt(Q), n)

            obs = (sm_true + obs_err) * H_true
            forcing = precip_true + forc_err

            # --- get OL ts & error ---
            OL = np.full(n, np.nan)

            model = deepcopy(api)
            for t, f in enumerate(forcing):
                x = model.step(f)
                OL[t] = x

            R_true = obs_err.var() * H_true**2
            Q_true = forc_err.var()
            P_OL_true = (np.mean((sm_true - OL)**2))

            SNR_R_true = sm_true.var() / obs_err.var()
            SNR_P_true = sm_true.var() / (forc_err.var() / (1-gamma**2))

            x_ana, P_ana, R_est, Q_est, H_est, checkvar = MadEnKF(api, forcing, obs, n_ens=n_ens, n_iter=n_iter)

            P_ana_true = np.mean((sm_true - x_ana) ** 2)
            P_ana_est = P_ana.mean()

            corr = pd.DataFrame({'truth': sm_true, 'obs': obs, 'OL': OL, 'ana': x_ana}).corr()

            result = pd.DataFrame({'SNR_P': SNR_P,
                                   'SNR_R': SNR_R,
                                   'SNR_R_true': SNR_R_true,
                                   'SNR_P_true': SNR_P_true,
                                   'n_ens': n_ens,
                                   'n_iter': n_iter,
                                   'R_true': R_true,
                                   'Q_true': Q_true,
                                   'R_est': R_est,
                                   'Q_est': Q_est,
                                   'H_est': H_est,
                                   'H_true': H_true,
                                   'checkvar': checkvar,
                                   'P_OL_true': P_OL_true,
                                   'P_ana_est': P_ana_est,
                                   'P_ana_true': P_ana_true,
                                   'corr_obs': corr['truth']['obs'],
                                   'corr_OL': corr['truth']['OL'],
                                   'corr_ana': corr['truth']['ana']}, index=(idx,))

            if (os.path.isfile(fname) == False):
                result.to_csv(fname, float_format='%0.4f')
            else:
                result.to_csv(fname, float_format='%0.4f', mode='a', header=False)

            print 'thread: %i, run: %i' % (thread, idx)

if __name__=='__main__':
    main()

