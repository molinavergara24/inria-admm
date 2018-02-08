import numpy as np

def dicairano(A,M,A_T):
	eig = np.linalg.eigvals(M)
	
	eigmax = np.absolute( np.amax(eig) )
	eigmin = np.absolute( np.min(eig[np.nonzero(eig)]) )
	return np.sqrt(eigmax*eigmin)
