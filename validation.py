
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import pearsonr, norm, t, chi

from pyldas.interface import LDAS_io

def correct_n(n, df):
    """ Corrects sample size based on data auto-correlation. """

    # calculate time difference and the average 50% of lags
    delta = df.index[1::] - df.index[0:-1]
    lag_l, lag_u = pd.Series(delta).quantile([0.25, 0.75])
    lags = delta[(delta >= lag_l) & (delta <= lag_u)].unique()

    # Auto-correlation as the average AC of both data sets, averaged over all lags
    rho = 0.
    for lag in lags:
        idx = np.where(delta == lag)[0]
        rho_ds1, P_ds1 = pearsonr(df.iloc[idx, 0].values, df.iloc[idx + 1, 0].values)
        rho_ds2, P_ds2 = pearsonr(df.iloc[idx, 1].values, df.iloc[idx + 1, 1].values)

        if (P_ds1 < 0.05) & (rho_ds1 > 0):
            rho += rho_ds1
        if (P_ds2 < 0.05) & (rho_ds2 > 0):
            rho += rho_ds2
    rho /= 2 * len(lags)

    ## --- GDL version: multiply AC of the ts instead of averaging ---
    #     if (P_ds1 < 0.05) & (rho_ds1 > 0) & (P_ds2 < 0.05) & (rho_ds2 > 0):
    #         rho += rho_ds1 * rho_ds2
    # rho /= len(lags)

    return round(n * (1 - rho) / (1 + rho))


def bias(df, dropna=True, alpha=0.95, flatten=True):
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

            t_l, t_u = t.interval(alpha, n-1)
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


def ubRMSD(df, dropna=True, alpha=0.95, flatten=True):
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

            chi_l, chi_u = chi.interval(alpha, n-1)
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
            chi_l, chi_u = chi.interval(alpha, n_corr - 1)
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

    io = LDAS_io('ObsFcstAna','US_M36_SMOS40_DA_cal_scaled')
    ser1 = io.timeseries['obs_fcst'][0,40,40].to_series()
    ser2 = io.timeseries['obs_obs'][0,40,40].to_series()
    df = pd.DataFrame([ser1,ser2]).swapaxes(0,1)

    print bias(df)
    print ubRMSD(df)
    print Pearson_R(df)
    # for val in res.loc['obs_ana','obs_fcst',:]:
    #     print val.values