import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk', font_scale=0.8)
from matplotlib.colors import LogNorm

from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage.filters import gaussian_filter

import numpy as np
import pandas as pd

import sympy as sym


def P_K_diff_dependency():

    x, y, R, P, H, M, K = sym.symbols('x y R P H M K')

    upd = x + K * (y - H*x)

    dx = sym.diff(upd, x)
    dy = sym.diff(upd, y)
    dH = sym.diff(upd, H)

    Pupd_eq = dx**2*P + dy**2*R + dH**2*M

    Pupd = sym.lambdify((K, R, P, H, M, x), Pupd_eq, 'numpy')
    Keq = sym.lambdify((H, R, P, M, x), H * P / (R + H**2*P + x**2*M), 'numpy')

    P_val = 2
    Rs = 10**(np.linspace(-10,1,200)/10) * P_val
    Ms = 10**(np.linspace(-10,1,200)/10) * P_val
    Rs, Ms = np.meshgrid(Rs, Ms)

    Hs = np.full(Rs.shape, 1)

    f, axs = plt.subplots(2, 2, figsize=(14,10), sharex=True, sharey=True)

    # plt.suptitle('$\Delta$P$^+$(10% deviation from K$_{opt}$)', y=0.95)

    for i, (ax, k) in enumerate(zip(axs.flatten(),[0, 0.5, 1, 2])):

        # plt.subplot(2, 2, i+1)

        xs_x = np.full(Rs.shape, k * np.sqrt(Ms))

        Ks = Keq(Hs, Rs, P_val, Ms, xs_x)
        Pupd_grid = Pupd(Ks, Rs, P_val, Hs, Ms, xs_x) / P_val

        Pupd_grid2 = Pupd(Ks*1.1, Rs, P_val, Hs, Ms, xs_x) / P_val

        ax.set_yscale('log')
        ax.set_xscale('log')
        im = ax.pcolormesh(Rs / P_val, Ms / P_val, Pupd_grid2/Pupd_grid, cmap='viridis')
        im.set_clim(vmin=1, vmax=1.08)

        CS = ax.contour(Rs / P_val, Ms / P_val, Pupd_grid2/Pupd_grid, colors='white', linestyles='--', linewidths=1, levels=[1.02, 1.04, 1.06, 1.08])
        ax.clabel(CS, inline=True, fontsize=10)

        ax.set_title('(x$^-$)$^2$ = ' + f'{k} $\cdot$ $\Sigma_o$')

    f.text(0.5, 0.05, 'R / HP$^-$H$^T$', ha='center')
    f.text(0.06, 0.5, '$\Sigma_o$ / HP$^-$H$^T$', va='center', rotation='vertical')

    f.subplots_adjust(right=0.85)
    cbar_ax = f.add_axes([0.87, 0.25, 0.02, 0.5])
    f.colorbar(im, cax=cbar_ax)

    fout = r'H:\work\SMAP_DA_paper\plots\skill_gain_K_dependence.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()


def x_variation():

    x, y, R, P, H, M, K = sym.symbols('x y R P H M K')

    upd = x + K * (y - H*x)

    dx = sym.diff(upd, x)
    dy = sym.diff(upd, y)
    dH = sym.diff(upd, H)

    Pupd_eq = dx**2*P + dy**2*R + dH**2*M

    Pupd = sym.lambdify((K, R, P, H, M, x), Pupd_eq, 'numpy')
    Keq = sym.lambdify((H, R, P, M, x), H * P / (R + H**2*P + x**2*M), 'numpy')

    P_val = 2
    Rs = 10**(np.linspace(-10,10,100)/10) * P_val
    Ms = 10**(np.linspace(-10,10,100)/10) * P_val
    Rs, Ms = np.meshgrid(Rs, Ms)

    Hs = np.full(Rs.shape, 1)

    f, axs = plt.subplots(2, 2, figsize=(14,10), sharex=True, sharey=True)

    plt.suptitle('Relative skill gain (1 - P$^+$/P$^-$)', y=0.95)

    for i, (ax, k) in enumerate(zip(axs.flatten(),[0, 0.5, 1, 2])):

        # plt.subplot(2, 2, i+1)

        xs_x = np.full(Rs.shape, k * np.sqrt(Ms))

        Ks = Keq(Hs, Rs, P_val, Ms, xs_x)
        Pupd_grid = Pupd(Ks, Rs, P_val, Hs, Ms, xs_x) / P_val

        ax.set_yscale('log')
        ax.set_xscale('log')
        im = ax.pcolormesh(Rs / P_val, Ms / P_val, 1-Pupd_grid, cmap='viridis')
        im.set_clim(vmin=0, vmax=1)
        CS = ax.contour(Rs / P_val, Ms / P_val, 1-Pupd_grid, colors='white', linestyles='--', linewidths=1, levels=[0.2,0.4,0.6,0.8])
        ax.clabel(CS, inline=True, fontsize=10)

        ax.set_title(f'x$^2$ = {k} $\cdot$ M')

    f.text(0.5, 0.05, 'R / HP$^-$H$^T$', ha='center')
    f.text(0.06, 0.5, 'M / HP$^-$H$^T$', va='center', rotation='vertical')

    f.subplots_adjust(right=0.85)
    cbar_ax = f.add_axes([0.87, 0.25, 0.02, 0.5])
    f.colorbar(im, cax=cbar_ax)

    fout = '/Users/u0116961/Documents/skill_gain_M_dependence.png'
    f.savefig(fout, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()


def line_plot():

    x, y, R, P, H, M, K = sym.symbols('x y R P H M K')

    upd = x + K * (y - H*x)

    dx = sym.diff(upd, x)
    dy = sym.diff(upd, y)
    dH = sym.diff(upd, H)

    Keq = sym.lambdify((R, P, H), H * P / (R + H ** 2 * P), 'numpy')
    Pupd = sym.lambdify((R, P, H, K), dx**2*P + dy**2*R, 'numpy')

    Keq2 = sym.lambdify((H, R, P, M, x), H * P / (R + H ** 2 * P + x ** 2 * M), 'numpy')
    Pupd2 = sym.lambdify((K, R, P, H, M, x), dx ** 2 * P + dy ** 2 * R + dH ** 2 * M, 'numpy')

    Rs = 10**(np.linspace(-10,10,100)/10)
    Ps = np.ones(100)
    Hs = np.ones(100)

    Ks = Keq(Rs, Ps, Hs)
    Pus = Pupd(Rs, Ps, Hs, Ks)

    f = plt.figure(figsize=(10,8))

    plt.title('Relative skill gain')
    plt.plot(Rs, 1-Pus)
    plt.xscale('log')
    plt.xlabel('R / P')
    plt.ylabel('1 - P$^+$ / P$^-$')

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    P_K_diff_dependency()
    # x_variation()
    # line_plot()
