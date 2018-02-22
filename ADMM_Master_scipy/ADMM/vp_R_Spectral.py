'''
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*v'*M*v + f'*v + indicator(u)
%   subject to   u = Av + b
'''

def vp_R_Spectral(problem_data, rho_method):

	######################
	## IMPORT LIBRARIES ##
	######################

	#Math libraries
	import numpy as np
	from scipy.sparse import csc_matrix
	from scipy.sparse import csr_matrix
	from scipy.sparse import linalg

	#Timing
	import time

	#Import data
	from Data.read_fclib import *

	#Plot residuals
	from Solver.ADMM_iteration.Numerics.plot import *

	#Initial penalty parameter
	import Solver.Rho.Optimal

	#Max iterations and kind of tolerance
	from Solver.Tolerance.iter_totaltolerance import *

	#Acceleration
	from Solver.Acceleration.minusr_vp import *

	#Varying penalty parameter
	from Solver.Rho.Varying.Spectral_relaxed import *

	#b = Es matrix
	from Data.Es_matrix import *

	#Projection onto second order cone
	from Solver.ADMM_iteration.Numerics.projection import *

	##################################
	############# REQUIRE ############
	##################################

	start = time.clock()
	problem = hdf5_file(problem_data)

	M = problem.M.tocsc()
	f = problem.f
	A = csc_matrix.transpose(problem.H.tocsc())
	A_T = csr_matrix.transpose(A)
	w = problem.w
	mu = problem.mu

	#Dimensions (normal,tangential,tangential)
	dim1 = 3 
	dim2 = np.shape(w)[0]

	#Problem size
	n = np.shape(M)[0]
	p = np.shape(w)[0]

	b = 0.1 * Es_matrix(w,mu,np.ones([p,])) / np.linalg.norm(Es_matrix(w,mu,np.ones([p,])))

	#################################
	############# SET-UP ############
	#################################

	#Set-up of vectors
	v = [np.zeros([n,])]
	u = [np.zeros([p,])] #this is u tilde, but in the notation of the paper is used as hat [np.zeros([10,0])]
	u_hat = [np.zeros([p,])] #u_hat[0] #in the notation of the paper this used with a underline
	xi = [np.zeros([p,])] 
	xi_hat = [np.zeros([p,])]
	xiG = [np.zeros([p,])]
	r = [np.zeros([p,])] #primal residual
	rG = [np.zeros([p,])]
	s = [np.zeros([p,])] #dual residual
	r_norm = [0]
	s_norm = [0]
	tau = [1] #over-relaxation
	e = [] #restart
	rho = []

	#Optimal penalty parameter
	rho_string = 'Solver.Rho.Optimal.' + rho_method + '(A,M,A_T)'
	rh = eval(rho_string)
	rho.append(rh) #rho[0]

	################
	## ITERATIONS ##
	################

	for k in range(MAXITER):

		#Super LU factorization of M + rho * dot(M_T,M)
		if rho[k] != rho[k-1] or k == 0:
			P = M + rho[k] * csc_matrix.dot(A_T,A)
			LU = linalg.splu(P)
			LU_old = LU
		else:
			LU = LU_old

		################
		## v - update ##
		################
		RHS = -f + rho[k] * csc_matrix.dot(A_T, -w - b - xi_hat[k] + u_hat[k])
		v.append(LU.solve(RHS)) #v[k+1]

		################
		## u - update ##
		################
		Av = csr_matrix.dot(A,v[k+1])
		vector = Av + xi_hat[k] + w + b
		u.append(projection(vector,mu,dim1,dim2)) #u[k+1]

		########################
		## residuals - update ##
		########################
		s.append(rho[k] * csc_matrix.dot(A_T,(u[k+1]-u_hat[k]))) #s[k+1]
		r.append(Av - u[k+1] + w + b) #r[k+1]

		#################
		## xi - update ##
		#################
		ratio = rho[k-1]/rho[k] #update of dual scaled variable with new rho
		xi.append(ratio*(xi_hat[k] + r[k+1])) #xi[k+1]

		####################
		## stop criterion ##
		####################
		pri_evalf = np.amax(np.array([np.linalg.norm(csr_matrix.dot(A,v[k+1])),np.linalg.norm(u[k+1]),np.linalg.norm(w + b)]))
		eps_pri = np.sqrt(p)*ABSTOL + RELTOL*pri_evalf

		dual_evalf = np.linalg.norm(rho[k] * csc_matrix.dot(A_T,xi[k+1]))
		eps_dual = np.sqrt(n)*ABSTOL + RELTOL*dual_evalf

		r_norm.append(np.linalg.norm(r[k+1]))
		s_norm.append(np.linalg.norm(s[k+1]))
		if r_norm[k+1]<=eps_pri and s_norm[k+1]<=eps_dual:
			break

		######################################
		## accelerated ADMM without restart ##
		######################################
		minusr(tau,u,u_hat,xi,xi_hat,k,ratio)

		#################################
		## Spectral parameter - update ##
		#################################
		rho.append(penalty(A,Av,u,w,b,xi,ratio,rG,xiG,rho,v,k))

		#end rutine

	end = time.clock()
	####################
	## REPORTING DATA ##
	####################
	#plotit(r,s,start,end,'With acceleration / Without restarting for '+problem_data+' for rho: '+rho_method)

	time = end - start
	return time
