
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import t

def plot_R_p():

    n = 500

    ylim = [0, 1]
    xlim = [3, 100]

    Rs = np.linspace(ylim[0],ylim[1],n)
    Ns = np.linspace(xlim[0],xlim[1],n)

    p = np.empty((n,n))
    p05 = np.empty(n)
    p01 = np.empty(n)

    for i, N in enumerate(Ns):

        df = N-2

        tt = t.ppf(0.975, df)
        p05[i] = np.sqrt(tt**2 / (df+tt**2))

        tt = t.ppf(0.995, df)
        p01[i] = np.sqrt(tt**2 / (df+tt**2))

        for j, R in enumerate(Rs):

            tt = (R * np.sqrt((df)/(1-R**2)))

            p[j,i] = 2 * (1 - t.cdf(tt, df))

    Ns, Rs = np.meshgrid(Ns, Rs)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)

    im = ax.pcolormesh(Ns, Rs, p, cmap='jet')

    ax.plot(Ns[0,:], p05, '-k', linewidth=3)
    ax.plot(Ns[0,:], p01, '-k', linewidth=3)

    plt.xlim(xlim)
    plt.ylim(ylim)

    fontsize = 14

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.title('p-value', fontsize=fontsize+4)

    plt.xlabel('Sample size', fontsize=fontsize+2)
    plt.ylabel('Correlation', fontsize=fontsize+2)

    # cbaxes = fig.add_axes([0.1, 0.1, 0.8, 0.1])
    cb = fig.colorbar(im, pad=0.02)
    cb.ax.tick_params(labelsize=fontsize-2)

    plt.text(31, p05[np.argmin(abs(Ns[0,:]-30))]-0.03, 'p = 0.05', fontsize=fontsize+2, color='white')
    plt.text(31, p01[np.argmin(abs(Ns[0,:]-30))]-0.03, 'p = 0.01', fontsize=fontsize+2, color='white')

    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    plot_R_p()