

import os

import numpy as np
import pandas as pd

from myprojects.validation_good_practice.data_readers.interface import reader, calc_anomaly, collocate
from myprojects.validation import estimate_tau, estimate_lag1_autocorr, calc_bootstrap_blocklength, bootstrap, tc

def main():
    io = reader()
    result_file = r'D:\work\validation_good_practice\confidence_invervals\ref_merra\result.csv'
    lut = pd.read_csv(r"D:\data_sets\EASE2_grid\grid_lut.csv", index_col=0)

    block_lengths = [1, 10, 25, 50]
    block_lengths_anom = [1, 5, 15, 25]
    sensors = ['MERRA2', 'ASCAT', 'AMSR2']
    n_samples = 1000

    cols_loc = ['row', 'col', 'n_abs', 'n_anom']

    cols_abs =  ['r2_abs_' + s for s in sensors] + \
                ['r2_abs_' + s + '_p50_bl_opt' for s in sensors] + \
                ['r2_abs_' + s + '_ci_l_bl_opt' for s in sensors] + \
                ['r2_abs_' + s + '_ci_u_bl_opt' for s in sensors] + \
                ['r2_abs_' + s + '_p50_bl_%i' % b for s in sensors for b in block_lengths] + \
                ['r2_abs_' + s + '_ci_l_bl_%i' % b for s in sensors for b in block_lengths] + \
                ['r2_abs_' + s + '_ci_u_bl_%i' % b for s in sensors for b in block_lengths] + \
                ['ubrmse_abs_' + s for s in sensors] + \
                ['ubrmse_abs_' + s + '_p50_bl_opt' for s in sensors] + \
                ['ubrmse_abs_' + s + '_ci_l_bl_opt' for s in sensors] + \
                ['ubrmse_abs_' + s + '_ci_u_bl_opt' for s in sensors] + \
                ['ubrmse_abs_' + s + '_p50_bl_%i' % b for s in sensors for b in block_lengths] + \
                ['ubrmse_abs_' + s + '_ci_l_bl_%i' % b for s in sensors for b in block_lengths] + \
                ['ubrmse_abs_' + s + '_ci_u_bl_%i' % b for s in sensors for b in block_lengths]

    cols_anom = ['r2_anom_' + s for s in sensors] + \
                ['r2_anom_' + s + '_p50_bl_opt' for s in sensors] + \
                ['r2_anom_' + s + '_ci_l_bl_opt' for s in sensors] + \
                ['r2_anom_' + s + '_ci_u_bl_opt' for s in sensors] + \
                ['r2_anom_' + s + '_p50_bl_%i' % b for s in sensors for b in block_lengths_anom] + \
                ['r2_anom_' + s + '_ci_l_bl_%i' % b for s in sensors for b in block_lengths_anom] + \
                ['r2_anom_' + s + '_ci_u_bl_%i' % b for s in sensors for b in block_lengths_anom] + \
                ['ubrmse_anom_' + s for s in sensors] + \
                ['ubrmse_anom_' + s + '_p50_bl_opt' for s in sensors] + \
                ['ubrmse_anom_' + s + '_ci_l_bl_opt' for s in sensors] + \
                ['ubrmse_anom_' + s + '_ci_u_bl_opt' for s in sensors] + \
                ['ubrmse_anom_' + s + '_p50_bl_%i' % b for s in sensors for b in block_lengths_anom] + \
                ['ubrmse_anom_' + s + '_ci_l_bl_%i' % b for s in sensors for b in block_lengths_anom] + \
                ['ubrmse_anom_' + s + '_ci_u_bl_%i' % b for s in sensors for b in block_lengths_anom]

    for cnt, (gpi, data) in enumerate(lut.iterrows()):
        print '%i / %i' % (cnt, len(lut))
        try:
            df = io.read(gpi, sensors=sensors)
            tau_abs = estimate_tau(df, n_lags=180)

            df_anom = calc_anomaly(df)
            tau_anom = estimate_tau(df_anom, n_lags=60)

            df_matched = collocate(df)
            ac_avg = estimate_lag1_autocorr(df_matched, tau_abs)
            bl = [calc_bootstrap_blocklength(df_matched, ac_avg),] + block_lengths
            bs_list = [bootstrap(df_matched, tmp_bl) for tmp_bl in bl]

            df_anom_matched = collocate(df_anom)
            ac_avg_anom = estimate_lag1_autocorr(df_anom_matched, tau_anom)
            bl_anom = [calc_bootstrap_blocklength(df_anom_matched, ac_avg_anom),] + block_lengths_anom
            bs_anom_list = [bootstrap(df_anom_matched, tmp_bl) for tmp_bl in bl_anom]

            res = pd.DataFrame(columns = cols_loc + cols_abs + cols_anom, index=(gpi,), dtype='float')
            res[['row','col']] = [data.ease2_row, data.ease2_col]
            res.loc[:, 'n_abs'] = len(df_matched)
            res.loc[:, 'n_anom'] = len(df_anom_matched)

            for i, (bs, bs_anom) in enumerate(zip(bs_list, bs_anom_list)):

                if ~np.isnan(bl[i]):
                    r2_abs = np.zeros((3,n_samples))
                    ubrmse_abs = np.zeros((3,n_samples))
                    for j in np.arange(n_samples):
                        try:
                            df = bs.next()
                            tc_res = tc(df)

                            r2_abs[:,j] = tc_res[1]
                            ubrmse_abs[:,j] = tc_res[2]
                        except:
                            pass

                    for k, s in enumerate(sensors):
                        if i == 0:
                            res.loc[:, 'r2_abs_' + s + '_p50_bl_opt'] = np.percentile(r2_abs[k,:],50)
                            res.loc[:, 'r2_abs_' + s + '_ci_l_bl_opt'] = np.percentile(r2_abs[k,:],2.5)
                            res.loc[:, 'r2_abs_' + s + '_ci_u_bl_opt'] = np.percentile(r2_abs[k,:],97.5)
                            res.loc[:, 'ubrmse_abs_' + s + '_p50_bl_opt'] = np.percentile(ubrmse_abs[k,:],50)
                            res.loc[:, 'ubrmse_abs_' + s + '_ci_l_bl_opt'] = np.percentile(ubrmse_abs[k,:],2.5)
                            res.loc[:, 'ubrmse_abs_' + s + '_ci_u_bl_opt'] = np.percentile(ubrmse_abs[k,:],97.5)
                        else:
                            res.loc[:, 'r2_abs_' + s + '_p50_bl_%i' % bl[i]] = np.percentile(r2_abs[k,:],50)
                            res.loc[:, 'r2_abs_' + s + '_ci_l_bl_%i' % bl[i]] = np.percentile(r2_abs[k,:],2.5)
                            res.loc[:, 'r2_abs_' + s + '_ci_u_bl_%i' % bl[i]] = np.percentile(r2_abs[k,:],97.5)
                            res.loc[:, 'ubrmse_abs_' + s + '_p50_bl_%i' % bl[i]] = np.percentile(ubrmse_abs[k,:],50)
                            res.loc[:, 'ubrmse_abs_' + s + '_ci_l_bl_%i' % bl[i]] = np.percentile(ubrmse_abs[k,:],2.5)
                            res.loc[:, 'ubrmse_abs_' + s + '_ci_u_bl_%i' % bl[i]] = np.percentile(ubrmse_abs[k,:],97.5)

                if ~np.isnan(bl_anom[i]):
                    r2_anom = np.zeros((3,n_samples))
                    ubrmse_anom = np.zeros((3,n_samples))
                    for j in np.arange(n_samples):
                        try:
                            df = bs_anom.next()
                            tc_res = tc(df)

                            r2_anom[:,j] = tc_res[1]
                            ubrmse_anom[:,j] = tc_res[2]
                        except:
                            pass

                    for k, s in enumerate(sensors):
                        if i == 0:
                            res.loc[:, 'r2_anom_' + s + '_p50_bl_opt'] = np.percentile(r2_anom[k,:],50)
                            res.loc[:, 'r2_anom_' + s + '_ci_l_bl_opt'] = np.percentile(r2_anom[k,:],2.5)
                            res.loc[:, 'r2_anom_' + s + '_ci_u_bl_opt'] = np.percentile(r2_anom[k,:],97.5)
                            res.loc[:, 'ubrmse_anom_' + s + '_p50_bl_opt'] = np.percentile(ubrmse_anom[k,:],50)
                            res.loc[:, 'ubrmse_anom_' + s + '_ci_l_bl_opt'] = np.percentile(ubrmse_anom[k,:],2.5)
                            res.loc[:, 'ubrmse_anom_' + s + '_ci_u_bl_opt'] = np.percentile(ubrmse_anom[k,:],97.5)
                        else:
                            res.loc[:, 'r2_anom_' + s + '_p50_bl_%i' % bl_anom[i]] = np.percentile(r2_anom[k,:],50)
                            res.loc[:, 'r2_anom_' + s + '_ci_l_bl_%i' % bl_anom[i]] = np.percentile(r2_anom[k,:],2.5)
                            res.loc[:, 'r2_anom_' + s + '_ci_u_bl_%i' % bl_anom[i]] = np.percentile(r2_anom[k,:],97.5)
                            res.loc[:, 'ubrmse_anom_' + s + '_p50_bl_%i' % bl_anom[i]] = np.percentile(ubrmse_anom[k,:],50)
                            res.loc[:, 'ubrmse_anom_' + s + '_ci_l_bl_%i' % bl_anom[i]] = np.percentile(ubrmse_anom[k,:],2.5)
                            res.loc[:, 'ubrmse_anom_' + s + '_ci_u_bl_%i' % bl_anom[i]] = np.percentile(ubrmse_anom[k,:],97.5)

            res_tc = tc(df_matched)
            res_tc_anom = tc(df_anom_matched)
            for k, s in enumerate(sensors):
                res.loc[:, 'r2_abs_' + s] = res_tc[1][k]
                res.loc[:, 'r2_anom_' + s] = res_tc_anom[1][k]
                res.loc[:, 'ubrmse_abs_' + s] = res_tc[2][k]
                res.loc[:, 'ubrmse_anom_' + s] = res_tc_anom[2][k]

        except:
            print 'GPI %i failed.' % gpi
            continue

        if (os.path.isfile(result_file) == False):
            res.to_csv(result_file, float_format='%0.3f')
        else:
            res.to_csv(result_file, float_format='%0.3f', mode='a', header=False)

if __name__=='__main__':
    main()
