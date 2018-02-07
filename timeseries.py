
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


def calc_pentadal_mean(ts):
    """
    Calculates the mean seasonal cycle as long-term mean within a 45 days moving average window
    for each pentad. A minimum of 20 data points are required.

    Parameters
    ----------
    ts : pd.Series w. DatetimeIndex
        Timeseries of which the climatology shall be calculated.

    Returns
    -------
    clim : pd.Series
        climatology of ts (without leap days)
    n_days : pd.Series
        the number of data points available within each window
    """

    xts = ts.dropna().copy()

    doys = xts.index.dayofyear.values
    doys[xts.index.is_leap_year & (doys > 59)] -= 1

    ts_pentad = np.floor((doys - 1) / 5.) + 1

    pentads = np.arange(73) + 1
    clim = pd.Series(index=pentads)
    n_data = pd.Series(index=pentads)
    for p in pentads:
        tmp_pentad = ts_pentad.copy()
        if p < 5:
            tmp_pentad[tmp_pentad > 10] -= 73
        if p > 69:
            tmp_pentad[tmp_pentad < 60] += 73
        n_data[p] = len(xts[(tmp_pentad >= p - 4) & (tmp_pentad <= p + 4)])

        # only store the climatology if more than 20 data points are available
        if n_data[p] >= 20:
            clim[p] = xts[(tmp_pentad >= p - 4) & (tmp_pentad <= p + 4)].values.mean()

    return clim, n_data

