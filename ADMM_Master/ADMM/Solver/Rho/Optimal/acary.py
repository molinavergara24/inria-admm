import numpy as np

def acary(A,M,A_T):
	M_norm1 = np.linalg.norm(M,1)	
	A_norm1 = np.linalg.norm(A,1)

	return M_norm1/A_norm1
