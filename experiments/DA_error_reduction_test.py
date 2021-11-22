
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context('talk', font_scale=0.8)



T = 1000

t = np.arange(T)

# ylim=[-2,2]

# True signal
tc = 2 * np.sin(2*np.pi*t / (T/2))
tl = 1.2 * np.sin(2*np.pi*t / (T/4))
ts = 0.25 * np.sin(2*np.pi*t / (T/32))
true = tc + tl + ts


# Signal 1
c11 = tc.var() * 0.24
c22 = tl.var() * 0.35
c33 = ts.var() * 0.18
c12 = 0
c13 = 0
c23 = 0
C1 = np.array([[c11,c12,c13],[c12,c22,c23],[c13,c23,c33]])
err1 = np.random.multivariate_normal([0,0,0], C1, T)
c1 = tc + err1[:,0]
l1 = tl + err1[:,1]
s1 = ts + err1[:,2]
sig1 = c1 + l1 + s1


# Signal 2
c11 = tc.var() * 0.2
c22 = tl.var() * 0.08
c33 = ts.var() * 0.14
c12 = 0
c13 = 0
c23 = 0
C2 = np.array([[c11,c12,c13],[c12,c22,c23],[c13,c23,c33]])
err2 = np.random.multivariate_normal([0,0,0], C2, T)
c2 = tc + err2[:,0]
l2 = tl + err2[:,1]
s2 = ts + err2[:,2]
sig2 = c2 + l2 + s2


w1 = 0.5
w2 = 0.5


A = np.matrix([w1, w1, w1, w2, w2, w2])
S = np.matrix(np.hstack((np.vstack((C1, np.zeros((3,3)))), np.vstack((np.zeros((3,3)), C2)))))

sig = w1 * sig1 + w2 * sig2

err_est1 = np.mean((sig1-true)**2)
err_est2 = np.mean((sig2-true)**2)

err_est = np.mean((sig-true)**2)

print(f'RMSD Signal1: {err1.var(axis=0,ddof=1).sum()} / {err_est1}')
print(f'RMSD Signal2: {err2.var(axis=0,ddof=1).sum()} / {err_est2}')
print(f'RMSD Signal: {(A*S*A.T)[0,0]} / {err_est}')





# print(sig1.var(ddof=1), true.var(ddof=1) + err.var(axis=0, ddof=1).sum())

# plt.figure(figsize=(18,10))
#
# plt.subplot(4,1,1)
# plt.plot(t, c1)
# # plt.ylim(ylim)
#
# plt.subplot(4,1,2)
# plt.plot(t, l1)
# # plt.ylim(ylim)
#
# plt.subplot(4,1,3)
# plt.plot(t, s1)
# # plt.ylim(ylim)
#
# plt.subplot(4,1,4)
# plt.plot(t,sig1)
#
#
# plt.tight_layout()
# plt.show()
