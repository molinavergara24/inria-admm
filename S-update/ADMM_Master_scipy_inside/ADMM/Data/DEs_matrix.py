import numpy as np
from scipy.sparse import csc_matrix

def DEs_matrix(w,mu,u,H_T):
	dim1 = 3
	dim2 = w.shape[0]

	u_per_contact = np.split(u,dim2/dim1)
	H_T_per_contact = np.split(H_T,dim2/dim1)

	for i in range(dim2/dim1):
		if i == 0:
			C1 = mu[i]/np.linalg.norm(u_per_contact[i][1:])
			C2 = np.dot(np.array([1,0,0]),u_per_contact[i])
			C3 = np.dot(np.array([[0,0,0],[0,1,0],[0,0,1]]), H_T_per_contact[i])
			DE_ = C1*C2*C3
		else:
			C1 = mu[i]/np.linalg.norm(u_per_contact[i][1:])
			C2 = np.dot(np.array([1,0,0]),u_per_contact[i])
			C3 = np.dot(np.array([[0,0,0],[0,1,0],[0,0,1]]), H_T_per_contact[i])
			DE_ = np.concatenate((DE_,C1*C2*C3))
	DE = np.transpose(DE_)

	return csc_matrix(DE)
