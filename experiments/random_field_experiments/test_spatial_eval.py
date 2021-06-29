

import numpy as np
import pandas as pd

from pathlib import Path

from multiprocessing import Pool

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.signal import convolve2d

from myprojects.experiments.random_field_experiments.gaussian_random_fields import gaussian_random_field

def aggregate(arr, window, return_agg=True):
    dim = arr.shape[0]
    new_dim = dim/window
    assert (new_dim % 1) == 0, f'Array cannot be aggregated into {new_dim} grid cells.'
    new_dim = int(new_dim)
    new_arr = np.empty((dim,dim))
    if return_agg:
        new_arr_agg = np.empty((new_dim,new_dim))
    for i in range(new_dim):
        for j in range(new_dim):
            new_arr[i*window:(i+1)*window,j*window:(j+1)*window] = arr[i*window:(i+1)*window,j*window:(j+1)*window].mean()
            if return_agg:
                new_arr_agg[i,j] = arr[i*window:(i+1)*window,j*window:(j+1)*window].mean()
    if return_agg:
        return new_arr, new_arr_agg
    else:
        return new_arr

def lag_i_autocorr(x, i):
    return np.corrcoef(x[:-i], x[i:])[0][1]


def get_fields(n_f, k_m, k_c, seed=None, alpha=2.5):
    # Create gaussian field roughly bound between 0 and 0.8
    alpha = 2.5
    fname = Path(f'/Users/u0116961/data_sets/random_fields/alpha_{alpha}_size_{n_f}_seed_{seed}.npy')
    if fname.exists():
        field_f = np.load(fname)
        print('Field loaded...')
    else:
        np.random.seed(seed)
        field_f = gaussian_random_field(n_f, alpha=2.5)
        pl, pu = np.percentile(field_f, [0.2, 99.8])
        field_f = (field_f - pl) / (pu - pl) * (0.6 - 0.0)
        field_f = np.abs(field_f)
        if seed is not None:
            np.save(fname, field_f)
        print('Field created...')
    lag = 100

    # arr1 = np.array([lag_i_autocorr(field_f[:, idx], lag) for idx in range(n_f)])
    # print(f'1-km auto-corr: {arr1.mean():.3f}')

    # Aggregate field to medium and coarse scale
    field_m, field_m_agg = aggregate(field_f, k_m)
    field_c, field_c_agg = aggregate(field_f, k_c)

    return field_f, field_m, field_c, field_m_agg, field_c_agg

def test_spatial_eval():

    recreate_fields = False

    plot_map = True
    plot_bias = False

    plot_stations = False

    seed = 1

    # Coarse, medium, and fine scale [m]
    scl_c = 25000
    scl_m = 1000
    scl_f = 10

    # Number of coarse, medium, and fine scale grid cells
    n_c = 10
    n_m = int(n_c * scl_c / scl_m)
    n_f = int(n_c * scl_c / scl_f)
    print(f'Grid cells: ')
    print(f'{n_c} @ {scl_c} m')
    print(f'{n_m} @ {scl_m} m')
    print(f'{n_f} @ {scl_f} m')

    # Scaling factors
    k_m = int(scl_m / scl_f)
    k_c = int(scl_c / scl_f)

    # Create random fields at different resolutions.
    if plot_map or (not recreate_fields):
        field_f, field_m, field_c, field_m_agg, field_c_agg = get_fields(n_f, k_m, k_c, seed=seed)

    sns.set_context('talk', font_scale=0.8)
    ms = 13

    # ----------------------------- PLOT MAP -----------------------------

    if plot_map:

        vmin = 0.05
        vmax = 0.55
        f, (ax_c, ax_m) = plt.subplots(1,2, figsize=(13,6))
        sns.heatmap(field_m_agg, cmap='jet_r', ax=ax_m, vmin=vmin, vmax=vmax, cbar=False)
        sns.heatmap(field_c_agg, cmap='jet_r', ax=ax_c, vmin=vmin, vmax=vmax, cbar=False)

        ax_c.set_xticks([])
        ax_c.set_yticks([])

        ax_m.set_xticks([])
        ax_m.set_yticks([])

        plt.tight_layout()

    # ---- Plot histogram ----
    # print('Plotting histogram...')
    # f = plt.figure(figsize=(7,7))
    # ax = plt.gca()
    # sns.histplot(field_f.flatten(), bins=20, kde=False, ax=ax)
    # ax.set_ylabel('')

    # ----------------------------- Calculate Biases MAP -----------------------------

    # --- Point locations distributed within the same coarse grid cells ---
    bias_m = []
    bias_c = []
    n_stat = n_c**2

    for i in range(n_c):
        for j in range(n_c):

            # Create station coordinates in fine scale space
            xs = np.random.randint(k_c*i, k_c*(i+1), n_stat)
            ys = np.random.randint(k_c*j, k_c*(j+1), n_stat)
            # Transform coordinates to medium / coarse scale space
            xs_m = xs / k_m; ys_m = ys / k_m
            xs_c = xs / k_c; ys_c = ys / k_c
            # Calculate biases

            if recreate_fields:
                field_f, field_m, field_c, field_m_agg, field_c_agg = get_fields(n_f, k_m, k_c, seed=seed)

            # bias_m.append(np.mean(field_f[xs, ys] - field_m[xs,ys]))
            # bias_c.append(np.mean(field_f[xs, ys] - field_c[xs,ys]))
            bias_m.append(np.corrcoef(field_f[xs, ys],field_m[xs, ys])[0,1])
            bias_c.append(np.corrcoef(field_f[xs, ys],field_c[xs, ys])[0,1])

    bias_m_sub = np.array(bias_m)
    bias_c_sub = np.array(bias_c)

    if plot_map and plot_stations:
        ax_m.plot(xs_m, ys_m, marker='X', linestyle='', markersize=ms, markerfacecolor='white', markeredgecolor='grey')
        ax_c.plot(xs_c, ys_c, marker='X', linestyle='', markersize=ms, markerfacecolor='white', markeredgecolor='grey')

    # --- 1 Point location per grid cell ---
    bias_m = []
    bias_c = []
    n_stat = 1
    n_iters = n_c**2
    for it in np.arange(n_iters):
        xs = []
        ys = []
        for i in range(n_c):
            for j in range(n_c):
                # Create station coordinates in fine scale space
                xs.append(np.random.randint(k_c * i, k_c * (i + 1), n_stat))
                ys.append(np.random.randint(k_c * j, k_c * (j + 1), n_stat))
        xs = np.array(xs)
        ys = np.array(ys)
        # Transform coordinates to medium / coarse scale space-
        xs_m = xs / k_m; ys_m = ys / k_m
        xs_c = xs / k_c; ys_c = ys / k_c
        # Calculate biases
        if recreate_fields:
            field_f, field_m, field_c, field_m_agg, field_c_agg = get_fields(n_f, k_m, k_c, seed=seed)
        # bias_m.append(np.mean(field_f[xs, ys] - field_m[xs, ys]))
        # bias_c.append(np.mean(field_f[xs, ys] - field_c[xs, ys]))

        bias_m.append(np.corrcoef(field_f[xs, ys].flatten(),field_m[xs, ys].flatten())[0,1])
        bias_c.append(np.corrcoef(field_f[xs, ys].flatten(),field_c[xs, ys].flatten())[0,1])

    bias_m_dist = np.array(bias_m)
    bias_c_dist = np.array(bias_c)

    if plot_map and plot_stations:
        ax_m.plot(xs_m, ys_m, marker='X', linestyle='', markersize=15, markerfacecolor='black', markeredgecolor='grey')
        ax_c.plot(xs_c, ys_c, marker='X', linestyle='', markersize=15, markerfacecolor='black', markeredgecolor='grey')


    # ----------------------------- PLOT BIASES -----------------------------

    # ylim = [-0.03,0.03]
    ylim = [0.6,1]

    if plot_bias:
        _, (ax1, ax2) = plt.subplots(2,1, figsize=[15,8])

        idx = np.argsort(bias_m_sub)
        df = pd.DataFrame({'grid_cell': np.arange(n_c**2), '1 km': bias_m_sub[idx], '25 km': bias_c_sub[idx]})
        g = sns.lineplot(x='grid_cell', y='bias', hue='Scale', data=df.melt('grid_cell', df.columns[1::], 'Scale', 'bias'), ax=ax1)
        g.set(xticklabels=[])
        ax1.axhline(color='k', linestyle='--', linewidth=1)
        ax1.set_ylim(ylim)
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_title('Bias over multiple station per coarse cell')
        g.legend(loc='upper left')

        idx = np.argsort(bias_c_dist)
        df = pd.DataFrame({'grid_cell': np.arange(n_c**2), '1 km': bias_m_dist[idx], '25 km': bias_c_dist[idx]})
        g = sns.lineplot(x='grid_cell', y='bias', hue='Scale', data=df.melt('grid_cell', df.columns[1::], 'Scale', 'bias'), ax=ax2)
        g.set(xticklabels=[])
        ax2.axhline(color='k', linestyle='--', linewidth=1)
        ax2.set_ylim(ylim)
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_title('Bias for single stations per coarse cell')
        g.legend(loc='upper left')

        plt.tight_layout()

    plt.show()

if __name__=='__main__':

    test_spatial_eval()


