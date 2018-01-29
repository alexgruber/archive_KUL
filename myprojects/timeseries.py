
import numpy as np
import pandas as pd

def calc_clim(ts, n=4):
    """
    Calculates the mean seasonal cycle of a data set
    by fitting harmonics.
    (!! Leap years are not yet properly treated !!)

    Parameters
    ----------
    ts : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.
    n : int (optional)
        Number of harmonics that should be fitted.
        n=0 : long term mean
        n=1 : long term mean + annual cycle
        n=2 : long term mean + half-annual cycle
        n=3 : long term mean + annual + seasonal cycle

    Returns
    -------
    clim : pd.Series
        climatology of ts (without leap days)
    """

    T = 365

    xts = ts.dropna().copy()
    doys = xts.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xts.index.is_leap_year & (doys>59)] -= 1

    A = np.ones((len(doys),2*n+1))

    for j in np.arange(n)+1:
        A[:,j] = np.cos(j * 2 * np.pi * doys / T)
        A[:,j+n] = np.sin(j * 2 * np.pi * doys / T)

    A = np.matrix(A)
    y = np.matrix(xts.values).T
    try:
        x = np.array((A.T * A).I * A.T * y).flatten()
    except:
        x = np.full(2*n+1,np.nan)

    doys = np.arange(T)+1
    clim = pd.Series(index=np.arange(T)+1)
    clim[:] = x[0]
    for j in np.arange(n)+1:
        clim[:] += x[j] * np.cos(j * 2 * np.pi * doys / T) + x[j+n] * np.sin(j * 2 * np.pi * doys / T)

    return clim


def calc_pentadal_mean(ts, return_pentadal=True):
    """
    Calculates the mean seasonal cycle as long-term mean during each pentad of the year

    Parameters
    ----------
    ts : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.
    return_pentadal : boolean
        If set, only 73 values (each pentad) will be returned
        If not seet, 365 values are returned (constant within each pentad)

    Returns
    -------
    clim : pd.Series
        climatology of ts (without leap days)
    """

    xts = ts.dropna().copy()
    doys = xts.index.dayofyear.values

    # in leap years, subtract 1 for all days after Feb 28
    doys[xts.index.is_leap_year & (doys > 59)] -= 1

    ts_pentads = np.floor((doys-1)/5.)
    clim_pentads = np.floor((np.arange(365))/5.)

    clim = pd.Series(index=np.arange(73) + 1)
    if return_pentadal is True:
        tmp_clim = xts.groupby(ts_pentads, axis=0).mean()
        clim.loc[tmp_clim.index+1] = tmp_clim.values

    else:
        clim = pd.Series(index=np.arange(365) + 1)
        for pent in np.unique(clim_pentads):
            clim.iloc[clim_pentads == pent] = xts.iloc[ts_pentads == pent].mean()

    return clim


