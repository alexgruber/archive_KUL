
import numpy as np

from scipy.stats import pearsonr

def test_avg_ci():

    n_meas = 500
    n_stat = 50
    n_pairs = n_stat * 2
    n_iter = 100

    w = np.matrix(np.ones(n_stat)/n_stat)

    mu = np.random.uniform(1.5, 3.5, n_pairs)
    c = np.matrix(np.random.uniform(0.4, 0.9, n_pairs))
    C_m = c.T*c
    np.fill_diagonal(C_m, 1.)

    z_avg_arr = np.full(n_iter, np.nan)
    z_se_avg_arr = np.full(n_iter, np.nan)

    for i in np.arange(n_iter):

        data = np.random.multivariate_normal(mu, C_m, n_meas)

        z_arr = np.full(n_stat, np.nan)
        z_se_arr = np.full(n_stat, np.nan)

        R_arr = np.full((n_stat,n_stat), np.nan)

        for j,k in zip(np.arange(n_stat),np.arange(n_stat) + n_stat):

            tmp_r = pearsonr(data[:,j],data[:,k])[0]

            z_arr[j] = 0.5 * np.log((1 + tmp_r) / (1 - tmp_r))
            z_se_arr[j] = (n_meas - 3) ** (-0.5)

            for l, m in zip(np.arange(j,n_stat), np.arange(j,n_stat) + n_stat):

                if j == l:
                    R_arr[j, l] = 1

                else:
                    rwx = pearsonr(data[:,j], data[:,l])[0]
                    rxy = pearsonr(data[:,j], data[:,k])[0]
                    rxz = pearsonr(data[:,j], data[:,m])[0]
                    rwy = pearsonr(data[:,l], data[:,k])[0]
                    rwz = pearsonr(data[:,l], data[:,m])[0]
                    ryz = pearsonr(data[:,k], data[:,m])[0]

                    A = rwx*ryz + rwy*rxz
                    B = -rxy * (rwx*rxz + rwy*ryz)
                    C = -rwz * (rwx*rwy + rxz*ryz)
                    D = rxy*rwz * (rwx + rxz + rwy + ryz) / 2.

                    R_arr[j,l] = (A + B + C + D) / ((1 - rxy**2)*(1 - rwz**2))

                    R_arr[l,j] = R_arr[j,l]

        z_se = np.matrix(z_se_arr)

        R_mat = np.matrix(R_arr)
        C_mat = np.multiply(R_mat, z_se.T*z_se)

        z_avg_arr[i] = z_arr.mean()
        z_se_avg_arr[i] = np.sqrt(w*C_mat*w.T)

    print 'true:   %.5f' % z_avg_arr.std()
    print 'est:    %.5f' % z_se_avg_arr.mean()

if __name__=='__main__':
    test_avg_ci()