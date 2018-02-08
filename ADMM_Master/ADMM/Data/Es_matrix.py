import numpy as np

def Es_matrix(w,mu,s):
	dim1 = 3
	dim2 = w.shape[0]
	E_ = np.array([])
	for i in range(dim2/dim1):
		E_ = np.concatenate((E_,np.array([0,1,1])*mu[i]))
	E = E_[:,np.newaxis]
	return np.squeeze(E) * s #s=1
