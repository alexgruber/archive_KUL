
import numpy as np

from scipy.stats import pearsonr

def test_avg_ci():

    n_meas = 5000
    n_stat = 50
    n_pairs = n_stat * 2
    n_iter = 1000

    w = np.matrix(np.ones(n_stat)/n_stat)

    mu = np.random.uniform(-1.5, 1.5, n_pairs)
    c = np.matrix(np.random.uniform(-0.1, 0.9, n_pairs))
    C = c.T*c
    np.fill_diagonal(C, 1.)

    b_avg_arr = np.full(n_iter, np.nan)
    u_avg_arr = np.full(n_iter, np.nan)

    b_se_avg_arr = np.full(n_iter, np.nan)
    u_se_avg_arr = np.full(n_iter, np.nan)

    for i in np.arange(n_iter):

        data = np.random.multivariate_normal(mu, C, n_meas)

        b_arr = np.full(n_stat, np.nan)
        u_arr = np.full(n_stat, np.nan)

        b_se_arr = np.full(n_stat, np.nan)
        u_se_arr = np.full(n_stat, np.nan)

        R_b_arr = np.full((n_stat,n_stat), np.nan)
        R_u_arr = np.full((n_stat,n_stat), np.nan)

        for j,k in zip(np.arange(n_stat),np.arange(n_stat) + n_stat):

            diff = data[:,j] - data[:,k]
            bias = diff.mean()
            ubRMSD = diff.std(ddof=1)

            b_arr[j] = bias
            u_arr[j] = ubRMSD

            b_se_arr[j] = ubRMSD / np.sqrt(n_meas)
            u_se_arr[j] = ubRMSD / np.sqrt(2.*(n_meas-1))

            for l, m in zip(np.arange(j,n_stat), np.arange(j,n_stat) + n_stat):
                x = data[:, j]-data[:, k]
                y = data[:, l]-data[:, m]

                R_b_arr[j,l] = pearsonr(x, y)[0]
                R_u_arr[j,l] = pearsonr((x-x.mean())**2, (y-y.mean())**2)[0]

                if j != l:
                    R_b_arr[l,j] = R_b_arr[j,l]
                    R_u_arr[l,j] = R_u_arr[j,l]

        b_se = np.matrix(b_se_arr)
        u_se = np.matrix(u_se_arr)

        R_b_mat = np.matrix(R_b_arr)
        R_u_mat = np.matrix(R_u_arr)

        C_mat_b = np.multiply(R_b_mat, b_se.T*b_se)
        C_mat_u = np.multiply(R_u_mat, u_se.T*u_se)

        b_avg_arr[i] = b_arr.mean()
        u_avg_arr[i] = u_arr.mean()

        b_se_avg_arr[i] = np.sqrt(w*C_mat_b*w.T)
        u_se_avg_arr[i] = np.sqrt(w*C_mat_u*w.T)

    print 'true:   %.5f,    %.5f' % (b_avg_arr.std(), u_avg_arr.std())

    print 'est:    %.5f,    %.5f' % (b_se_avg_arr.mean(), u_se_avg_arr.mean())

if __name__=='__main__':
    test_avg_ci()