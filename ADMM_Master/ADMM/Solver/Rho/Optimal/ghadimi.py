import numpy as np
from scipy.sparse.linalg import eigs

def ghadimi(A,M,A_T):
	DUAL = np.dot(np.dot(A,np.linalg.inv(M)),A_T)

	try:
		eig,eigvect = eigs(DUAL) #sparse
	except:
		eig = np.linalg.eigvals(DUAL) #quite dense

	eigmax = np.absolute(np.amax(eig))
	eigmin = np.absolute(np.min(eig[np.nonzero(eig)]))
	return 1 / np.sqrt(eigmax*eigmin)
