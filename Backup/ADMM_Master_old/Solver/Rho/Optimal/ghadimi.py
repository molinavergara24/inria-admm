import numpy as np

def ghadimi(A,M,A_T):
	DUAL = np.dot(np.dot(A,np.linalg.inv(M)),A_T)
	eig = np.linalg.eigvals(DUAL)
	return 1 / np.sqrt(np.amax(eig)*np.amin(eig))
