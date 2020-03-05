
import numpy as np

ssm = np.load('/data_sets/ESA_CCI_SM/_aggregates/monthly_avg_active_v04.4_2007_2017.npy')

ssm = ssm.reshape(ssm.shape[0], ssm.shape[-2]*ssm.shape[-1])

# Extract gpis with all valid data
ind_valid = np.where(np.isnan(ssm).sum(axis=0) == 0)[0][0:100]
ssm = ssm[:,ind_valid]
n_gp = len(ind_valid)
# ssm[np.isnan(ssm)] = 0

#Standardize the ssm data
ssm_standardized = (ssm - np.nanmean(ssm, axis=1)[:,np.newaxis]) / np.nanstd(ssm, axis=1)[:,np.newaxis]

ssm_correlation = np.empty((n_gp,n_gp))

for i in range(n_gp):
    for j in range(n_gp):
        print(i,j)
        ssm_correlation[i,j] = 1/len(ssm)*ssm_standardized[:,i][np.newaxis,:]@ssm_standardized[:,j][:,np.newaxis]


#Calculate the correlation matrix
ssm_correlation = 1/len(ssm)*ssm_standardized.T@ssm_standardized

#Perform eigenanalysis
ssm_eigenvalues, ssm_eigenvectors = np.linalg.eig(ssm_correlation)

#Calculate principal components
ssm_pcs = ssm_standardized@ssm_eigenvectors

#Standardize the principal components
ssm_pcs = (ssm_pcs - np.mean(ssm_pcs, axis=0)) / np.std(ssm_pcs, axis=0)

#Calculate the EOFs in physical units
ssm_eigenvectors_physical = 1/len(ssm)*ssm_pcs.T@ssm

#Reshape the eigenvectors into the map shape
ssm_eigenvectors_physical = ssm_eigenvectors_physical.reshape(len(ssm_eigenvectors_physical), len(latitudes_tropics), len(longitudes_tropics))
ssm_eigenvectors = ssm_eigenvectors.reshape(len(ssm_eigenvectors), (len(latitudes_tropics), len(longitudes_tropics)))

#Calculate the variance explained by each eigenvector
ssm_eigenvalues_variance = ssm_eigenvalues/np.sum(ssm_eigenvalues)

fig1, ax1 = plt.subplots()

ax1.plot(ssm_eigenvalues_variance[:10],'o')

ax1.set_ylabel('Variance Explained\n(fractional)', color='0.75', fontsize=20)
ax1.set_xlabel('Eigenvector Number', color='0.75', fontsize=20)

ax1.tick_params(labelsize=15, color='0.75', labelcolor='0.75')