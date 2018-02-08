import numpy as np
from Numerics.projection import *

def ADMM_iteration(f,rho,A_T,b,xi,u,A,v,mu,r,s,M,k,dim1,dim2):

	#Cholesky factorization of M + rho * dot(M_T,M)
	P = M + rho[k] * np.dot(A_T,A)
	L = np.linalg.cholesky(P)
	L_T = np.transpose(L)

	################
	## v - update ##
	################
        RHS = -f + rho[k] * np.dot(A_T, -b - xi[k] + u[k])
        v_cholesky = np.linalg.solve(L, RHS)
	v.append(np.linalg.solve(L_T, v_cholesky)) #v[k+1]

	################
	## u - update ##
	################
	Av = np.dot(A,v[k+1])
	vector = Av + xi[k] + b
	u.append(projection(vector,mu,dim1,dim2)) #u[k+1]

	########################
	## residuals - update ##
	########################
	s.append(rho[k] * np.dot(A_T,(u[k+1]-u[k]))) #s[k+1]
	r.append(Av - u[k+1] + b) #r[k+1]

	#################
	## xi - update ##
	#################
	ratio = rho[k]/rho[k+1] #update of dual scaled variable with new rho
	xi.append(ratio*(xi[k] + r[k+1])) #xi[k+1]
