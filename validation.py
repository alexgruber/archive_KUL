
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import pearsonr, norm, t, chi
import scipy.optimize as optimization

def tc(df, ref_ind=0):

    cov = df.dropna().cov().values

    ind = (0, 1, 2, 0, 1, 2)
    no_ref_ind = np.where(np.arange(3) != ref_ind)[0]

    snr = np.array([np.abs(((cov[i, i] * cov[ind[i + 1], ind[i + 2]]) /
                            (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) - 1)) ** (-1)
                        for i in np.arange(3)])

    snr_db = 10 * np.log10(snr)
    R2 = 1. / (1 + snr**(-1))

    err_var = np.array([
        np.abs(cov[i, i] -
               (cov[i, ind[i + 1]] * cov[i, ind[i + 2]]) / cov[ind[i + 1], ind[i + 2]])
        for i in np.arange(3)])

    beta = np.abs(np.array([cov[ref_ind, no_ref_ind[no_ref_ind != i][0]] /
                     cov[i, no_ref_ind[no_ref_ind != i][0]] if i != ref_ind
                     else 1 for i in np.arange(3)]))

    return snr_db, R2, np.sqrt(err_var) * beta, beta

def estimate_tau(in_df, n_lags=180):
    """ Estimate characteristic time lengths for pd.DataFrame columns """

    df = in_df.copy().resample('1D').last()
    n_cols = len(df.columns)

    # calculate auto-correlation for different lags
    rho = np.full((n_cols,n_lags), np.nan)
    for lag in np.arange(n_lags):
        for i,col in enumerate(df):
            Ser_l = df[col].copy()
            Ser_l.index += pd.Timedelta(lag, 'D')
            rho[i,lag] = df[col].corr(Ser_l)

    # Fit exponential function to auto-correlations and estimate tau
    tau = np.full(n_cols, np.nan)
    for i in np.arange(n_cols):
        try:
            ind = np.where(~np.isnan(rho[i,:]))[0]
            if len(ind) > 20:
                popt = optimization.curve_fit(lambda x, a: np.exp(a * x), np.arange(n_lags)[ind], rho[i,ind],
                                              bounds = [-1., -1. / n_lags])[0]
                tau[i] = np.log(np.exp(-1.)) / popt
        except:
            # If fit doesn't converge, fall back to the lag where calculated auto-correlation actually drops below 1/e
            ind = np.where(rho[i,:] < np.exp(-1))[0]
            tau[i] = ind[0] if (len(ind) > 0) else n_lags # maximum = # calculated lags

    # print tau
    # import matplotlib.pyplot as plt
    # xlim = [0,60]
    # ylim = [-0.4,1]
    # plt.figure(figsize=(14,9))
    # for i in np.arange(n_cols):
    #     plt.subplot(n_cols,1,i+1)
    #     plt.plot(np.arange(n_lags),rho[i,:])
    #     plt.plot(np.arange(0,n_lags,0.1),np.exp(-np.arange(0,n_lags,0.1)/tau[i]))
    #     plt.plot([tau[i],tau[i]],[ylim[0],np.exp(-1)],'--k')
    #     plt.xlim(xlim)
    #     plt.ylim(ylim)
    #     plt.text(xlim[1]-xlim[1]/15.,0.7,df.columns.values[i],fontsize=14)
    #
    # plt.tight_layout()
    # plt.show()

    return tau

def estimate_lag1_autocorr(df, tau=None):
    """ Estimate geometric average median lag-1 auto-correlation """

    # Get auto-correlation length for all time series
    if tau is None:
        tau = estimate_tau(df)

    # Calculate gemetric average lag-1 auto-corr
    avg_spc_t = np.median((df.index[1::] - df.index[0:-1]).days)
    ac = np.exp(-avg_spc_t/tau)
    avg_ac = ac.prod()**(1./len(ac))

    return avg_ac

def calc_bootstrap_blocklength(df, avg_ac=None):
    """ Calculate optimal block length [days] for block-bootstrapping of a data frame """

    # Get average lag-1 auto-correlation
    if avg_ac is None:
        avg_ac = estimate_lag1_autocorr(df)

    # Estimate block length (maximum 0.8 * data length)
    bl = min([round((np.sqrt(6) * avg_ac / (1 - avg_ac**2))**(2/3.)*len(df)**(1/3.)), round(0.8*len(df))])

    return bl

def bootstrap(df, bl):
    """ Bootstrap sample generator for a data frame with given block length [days] """

    N_df = len(df)
    t = df.index

    # build up list of all blocks (only consider block if number of data is at least half the block length)
    if bl > 1:
        blocks = list()
        for i in np.arange(N_df - int(bl / 2.)):
            ind = np.where((t >= t[i]) & (t < (t[i] + pd.Timedelta(bl, 'D'))))[0]
            if len(ind) > (bl / 2.):
                blocks.append(ind)
        N_blocks = len(blocks)

    # randomly draw a sample of blocks and trim it to df length
    while True:
        if bl == 1:
            ind = np.round(np.random.uniform(0, N_df-1, N_df)).astype('int')
        else:
            tmp_ind = np.round(np.random.uniform(0, N_blocks - 1, int(np.ceil(2. * N_df / bl)))).astype('int')
            ind = [i for block in np.array(blocks)[tmp_ind] for i in block]
        yield df.iloc[ind[0:N_df],:]

def correct_n(n, df):
    """ Corrects sample size based on avergae lag-1 auto-correlation. """

    # get geometric average median lag-1 auto-correlation
    rho = estimate_lag1_autocorr(df)

    return round(n * (1 - rho) / (1 + rho))


def bias(df, dropna=True, alpha=0.05, flatten=True):
    """"
    Calculates temporal mean biases and its confidence intervals based on Student's t-distribution,
    both with and without auto-correlation corrected sample size.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame whose k columns will be correlated
    dropna : boolean
        If false, temporal matching (dropna-based) will be done for each column-combination individually
    alpha : float [0,1]
        Significance level for the confidence intervals
    flatten : boolean
        If set, results are returned as pd.Series in case df only holds 2 columns

    Returns
    -------
    res : xr.DataArray
        (k x k x 7) Data Array holding the following statistics for each data set combination of df:
        bias : Temporal mean bias
        n, n_corr : original and auto-correlation corrected sample size
        CI_l, CI_l_corr, CI_u, CI_u_corr : lower and upper confidence levels with and without sample size correction

    res : pd.Series (if flatten is True and df contains only two columns)
        Series holding the above described statistics for the two input data sets.

    """

    if not isinstance(df,pd.DataFrame):
        print 'Error: Input is no pd.DataFrame.'
        return None

    if dropna is True:
        df.dropna(inplace=True)

    df.sort_index(inplace=True)

    cols = df.columns.values

    stats = ['bias', 'n', 'CI_l', 'CI_u', 'n_corr', 'CI_l_corr', 'CI_u_corr']
    dummy = np.full((len(cols),len(cols),len(stats)),np.nan)

    res = xr.DataArray(dummy, dims=['ds1','ds2','stats'], coords={'ds1':cols,'ds2':cols,'stats':stats})

    for ds1 in cols:
        for ds2 in cols:
            if ds1 == ds2:
                continue

            # get sample size
            tmpdf = df[[ds1,ds2]].dropna()
            n = len(tmpdf)
            res.loc[ds1, ds2, 'n'] = n
            res.loc[ds2, ds1, 'n'] = n
            if n < 5:
                continue

            # Calculate bias & ubRMSD
            diff = tmpdf[ds1].values - tmpdf[ds2].values
            bias = diff.mean()
            ubRMSD = diff.std(ddof=1)

            t_l, t_u = t.interval(1-alpha, n-1)
            CI_l = bias + t_l * ubRMSD / np.sqrt(n)
            CI_u = bias + t_u * ubRMSD / np.sqrt(n)

            res.loc[ds1, ds2, 'bias'] = bias
            res.loc[ds1, ds2, 'CI_l'] = CI_l
            res.loc[ds1, ds2, 'CI_u'] = CI_u

            n_corr = correct_n(n, tmpdf)
            res.loc[ds1, ds2, 'n_corr'] = n_corr
            res.loc[ds2, ds1, 'n_corr'] = n_corr
            if n_corr < 5:
                continue

            # Confidence intervals with corrected sample size
            t_l, t_u = t.interval(alpha, n_corr - 1)
            CI_l = bias + t_l * ubRMSD / np.sqrt(n_corr)
            CI_u = bias + t_u * ubRMSD / np.sqrt(n_corr)

            res.loc[ds1, ds2, 'CI_l_corr'] = CI_l
            res.loc[ds1, ds2, 'CI_u_corr'] = CI_u

    if flatten is True:
        if len(cols) == 2:
            res = pd.Series(res.loc[cols[0], cols[1], :], index=stats, dtype='float32')

    return res


def ubRMSD(df, dropna=True, alpha=0.05, flatten=True):
    """"
    Calculates the unbiased Root-Mean-Square-Difference and its confidence intervals based on the chi-distribution,
    both with and without auto-correlation corrected sample size.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame whose k columns will be correlated
    dropna : boolean
        If false, temporal matching (dropna-based) will be done for each column-combination individually
    alpha : float [0,1]
        Significance level for the confidence intervals
    flatten : boolean
        If set, results are returned as pd.Series in case df only holds 2 columns

    Returns
    -------
    res : xr.DataArray
        (k x k x 7) Data Array holding the following statistics for each data set combination of df:
        ubRMSD : unbiased Root-Mean-Square-Difference
        n, n_corr : original and auto-correlation corrected sample size
        CI_l, CI_l_corr, CI_u, CI_u_corr : lower and upper confidence levels with and without sample size correction

    res : pd.Series (if flatten is True and df contains only two columns)
        Series holding the above described statistics for the two input data sets.

    """

    if not isinstance(df,pd.DataFrame):
        print 'Error: Input is no pd.DataFrame.'
        return None

    if dropna is True:
        df.dropna(inplace=True)

    df.sort_index(inplace=True)

    cols = df.columns.values

    stats = ['ubRMSD', 'n', 'CI_l', 'CI_u', 'n_corr', 'CI_l_corr', 'CI_u_corr']
    dummy = np.full((len(cols),len(cols),len(stats)),np.nan)

    res = xr.DataArray(dummy, dims=['ds1','ds2','stats'], coords={'ds1':cols,'ds2':cols,'stats':stats})

    for ds1 in cols:
        for ds2 in cols:
            if ds1 == ds2:
                continue

            # get sample size
            tmpdf = df[[ds1,ds2]].dropna()
            n = len(tmpdf)
            res.loc[ds1, ds2, 'n'] = n
            res.loc[ds2, ds1, 'n'] = n
            if n < 5:
                continue

            # Calculate bias & ubRMSD
            diff = tmpdf[ds1].values - tmpdf[ds2].values
            ubRMSD = diff.std(ddof=1)

            chi_l, chi_u = chi.interval(1-alpha, n-1)
            CI_l = ubRMSD * np.sqrt(n-1) / chi_u
            CI_u = ubRMSD * np.sqrt(n-1) / chi_l

            res.loc[ds1, ds2, 'ubRMSD'] = ubRMSD
            res.loc[ds1, ds2, 'CI_l'] = CI_l
            res.loc[ds1, ds2, 'CI_u'] = CI_u

            n_corr = correct_n(n, tmpdf)
            res.loc[ds1, ds2, 'n_corr'] = n_corr
            res.loc[ds2, ds1, 'n_corr'] = n_corr
            if n_corr < 5:
                continue

            # Confidence intervals with corrected sample size
            chi_l, chi_u = chi.interval(1-alpha, n_corr - 1)
            CI_l = ubRMSD * np.sqrt(n_corr - 1) / chi_u
            CI_u = ubRMSD * np.sqrt(n_corr - 1) / chi_l

            res.loc[ds1, ds2, 'CI_l_corr'] = CI_l
            res.loc[ds1, ds2, 'CI_u_corr'] = CI_u

    if flatten is True:
        if len(cols) == 2:
            res = pd.Series(res.loc[cols[0], cols[1], :], index=stats, dtype='float32')

    return res


def Pearson_R(df, dropna=True, alpha=0.95, flatten=True):
    """"
    Calculates the Pearson correlation coefficient and its confidence intervals based on
    Fischer's z-transformation, both with and without auto-correlation corrected sample size.

    Parameters
    ----------
    df : pd.DataFrame
        Data Frame whose k columns will be correlated
    dropna : boolean
        If false, temporal matching (dropna-based) will be done for each column-combination individually
    alpha : float [0,1]
        Significance level for the confidence intervals
    flatten : boolean
        If set, results are returned as pd.Series in case df only holds 2 columns

    Returns
    -------
    res : xr.DataArray
        (k x k x 8) Data Array holding the following statistics for each data set combination of df:
        R, p : Pearson correlation coefficient and corresponding significance level
        n, n_corr : original and auto-correlation corrected sample size
        CI_l, CI_l_corr, CI_u, CI_u_corr : lower and upper confidence levels with and without sample size correction

    res : pd.Series (if flatten is True and df contains only two columns)
        Series holding the above described statistics for the two input data sets.

    """

    if not isinstance(df,pd.DataFrame):
        print 'Error: Input is no pd.DataFrame.'
        return None

    if dropna is True:
        df.dropna(inplace=True)

    df.sort_index(inplace=True)

    cols = df.columns.values

    stats = ['R', 'p', 'n', 'CI_l', 'CI_u', 'n_corr', 'CI_l_corr', 'CI_u_corr']
    dummy = np.full((len(cols),len(cols),len(stats)),np.nan)

    res = xr.DataArray(dummy, dims=['ds1','ds2','stats'], coords={'ds1':cols,'ds2':cols,'stats':stats})

    for ds1 in cols:
        for ds2 in cols:
            if ds1 == ds2:
                continue

            # get sample size
            tmpdf = df[[ds1,ds2]].dropna()
            n = len(tmpdf)
            res.loc[ds1, ds2, 'n'] = n
            res.loc[ds2, ds1, 'n'] = n
            if n < 5:
                continue

            # Calculate correlation and -significance
            R, p = pearsonr(tmpdf[ds1].values,tmpdf[ds2].values)

            # Fisher's z-transform for confidence intervals
            z = 0.5 * np.log((1+R)/(1-R))
            z_l, z_u = norm.interval(alpha, loc=z, scale=(n - 3) ** (-0.5))
            CI_l = (np.exp(2*z_l) - 1) / (np.exp(2*z_l) + 1)
            CI_u = (np.exp(2*z_u) - 1) / (np.exp(2*z_u) + 1)

            res.loc[ds1, ds2, 'R'] = R
            res.loc[ds1, ds2, 'p'] = p
            res.loc[ds1, ds2, 'CI_l'] = CI_l
            res.loc[ds1, ds2, 'CI_u'] = CI_u

            n_corr = correct_n(n, tmpdf)
            res.loc[ds1, ds2, 'n_corr'] = n_corr
            res.loc[ds2, ds1, 'n_corr'] = n_corr
            if n_corr < 5:
                continue

            # Confidence intervals with corrected sample size
            z_l, z_u = norm.interval(alpha, loc=z, scale=(n_corr - 3) ** (-0.5))
            CI_l = (np.exp(2 * z_l) - 1) / (np.exp(2 * z_l) + 1)
            CI_u = (np.exp(2 * z_u) - 1) / (np.exp(2 * z_u) + 1)

            res.loc[ds1, ds2, 'CI_l_corr'] = CI_l
            res.loc[ds1, ds2, 'CI_u_corr'] = CI_u

    if flatten is True:
        if len(cols) == 2:
            res = pd.Series(res.loc[cols[0], cols[1], :], index=stats, dtype='float32')

    return res



if __name__ == '__main__':

    from pyldas.interface import LDAS_io
    io = LDAS_io('ObsFcstAna','US_M36_SMOS40_DA_cal_scaled')
    ser1 = io.timeseries['obs_fcst'][0,40,40].to_series()
    ser2 = io.timeseries['obs_obs'][0,40,40].to_series()
    df = pd.DataFrame([ser1,ser2]).swapaxes(0,1)

    print bias(df)
    print ubRMSD(df)
    print Pearson_R(df)
    # for val in res.loc['obs_ana','obs_fcst',:]:
    #     print val.values