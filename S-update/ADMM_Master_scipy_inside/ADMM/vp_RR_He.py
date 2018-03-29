'''
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*v'*M*v + f'*v + indicator(u)
%   subject to   u = Av + b
'''

def vp_RR_He(problem_data, rho_method):

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
	from Solver.Acceleration.plusr_vp import *

	#Varying penalty parameter
	from Solver.Rho.Varying.He import *

	#b = Es matrix
	from Data.Es_matrix import *

	#Db = DEs matrix (derivative)
	from Data.DEs_matrix import *

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

	b = [Es_matrix(w,mu,np.zeros([p,]))]

	#################################
	############# SET-UP ############
	#################################

	#Set-up of vectors
	v = [np.zeros([n,])]
	u = [np.zeros([p,])] #this is u tilde, but in the notation of the paper is used as hat [np.zeros([10,0])]
	u_hat = [np.zeros([p,])] #u_hat[0] #in the notation of the paper this used with a underline
	xi = [np.zeros([p,])] 
	xi_hat = [np.zeros([p,])]
	r = [np.zeros([p,])] #primal residual
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

	#Plot
	rho_plot = []	
	b_plot = []
	u_bin_plot = []
	xi_bin_plot = []
	siconos_plot = []

	################
	## ITERATIONS ##
	################

	for k in range(MAXITER):
		print k
	
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
		RHS = -f + rho[k] * csc_matrix.dot(A_T, -w - b[k] - xi_hat[k] + u_hat[k])
		v.append(LU.solve(RHS)) #v[k+1]

		################
		## b - update ##
		################
		Av = csr_matrix.dot(A,v[k+1])
		b.append(Es_matrix(w,mu,Av + w))

		################
		## u - update ##
		################
		vector = Av + xi_hat[k] + w + b[k+1]
		u.append(projection(vector,mu,dim1,dim2)) #u[k+1]

		########################
		## residuals - update ##
		########################
		s.append(rho[k] * csc_matrix.dot(A_T,(u[k+1]-u_hat[k]))) #s[k+1]
		r.append(Av - u[k+1] + w + b[k+1]) #r[k+1]

		#################
		## xi - update ##
		#################
		ratio = rho[k-1]/rho[k] #update of dual scaled variable with new rho
		xi.append(ratio*(xi_hat[k] + r[k+1])) #xi[k+1]

		###################################
		## accelerated ADMM with restart ##
		###################################
		plusr(tau,u,u_hat,xi,xi_hat,k,e,rho,ratio)

		################################
		## penalty parameter - update ##
		################################
		r_norm.append(np.linalg.norm(r[k+1]))
		s_norm.append(np.linalg.norm(s[k+1]))
		rho.append(penalty(rho[k],r_norm[k+1],s_norm[k+1]))

		####################
		## stop criterion ##
		####################
		pri_evalf = np.amax(np.array([np.linalg.norm(csr_matrix.dot(A,v[k+1])),np.linalg.norm(u[k+1]),np.linalg.norm(w + b[k+1])]))
		eps_pri = np.sqrt(p)*ABSTOL + RELTOL*pri_evalf

		dual_evalf = np.linalg.norm(rho[k] * csc_matrix.dot(A_T,xi[k+1]))
		eps_dual = np.sqrt(n)*ABSTOL + RELTOL*dual_evalf

		R = -rho[k]*xi[k+1]
		N1 = csc_matrix.dot(M, v[k+1]) - csc_matrix.dot(A_T, R) + f
		N2 = u[k+1] - projection(u[k+1] - R, mu, dim1, dim2)  
		N1_norm = np.linalg.norm(N1)
		N2_norm = np.linalg.norm(N2)
		siconos_plot.append(np.sqrt( N1_norm**2 + N2_norm**2 ))

		if k == MAXITER-1: #r_norm[k+1]<=eps_pri and s_norm[k+1]<=eps_dual

			for element in range(len(u)):
				#Relative velocity
				u_proj = projection(u[element],mu,dim1,dim2)

				u_proj_contact = np.split(u_proj,dim2/dim1)
				u_contact = np.split(u[element],dim2/dim1)			

				u_count = 0.0
				for contact in range(dim2/dim1):
					if np.allclose(u_contact[contact], u_proj_contact[contact], rtol=0.1, atol=0.0):
						u_count += 1.0
			
				u_bin = 100 * u_count / (dim2/dim1)
				u_bin_plot.append(u_bin)

				#Reaction
				xi_proj = projection(-1.0 * xi[element],1/mu,dim1,dim2)

				xi_proj_contact = np.split(xi_proj,dim2/dim1)
				xi_contact = np.split(-1.0 * xi[element],dim2/dim1)			

				xi_count = 0.0
				for contact in range(dim2/dim1):
					if np.allclose(xi_contact[contact], xi_proj_contact[contact], rtol=0.1, atol=0.0):
						xi_count += 1.0
			
				xi_bin = 100 * xi_count / (dim2/dim1)
				xi_bin_plot.append(xi_bin)					

			for element in range(len(r_norm)):
				rho_plot.append(rho[element])
				b_plot.append(np.linalg.norm(b[element]))

			#R = -rho[k]*xi[k+1]
			#N1 = csc_matrix.dot(M, v[k+1]) - csc_matrix.dot(A_T, R) + f
			#N2 = u[k+1] - projection(u[k+1] - R, mu, dim1, dim2)  
			#N1_norm = np.linalg.norm(N1)
			#N2_norm = np.linalg.norm(N2)

			#print np.sqrt( N1_norm**2 + N2_norm**2 )
			print b_plot[-1]
			print b[-1][:3]
			print b[-1][-3:]				
			break

		b_per_contact_j1 = np.split(b[k+1],dim2/dim1)
		b_per_contact_j0 = np.split(b[k],dim2/dim1)
		count = 0
		for j in range(dim2/dim1):
			if np.linalg.norm(b_per_contact_j1[j] - b_per_contact_j0[j]) / np.linalg.norm(b_per_contact_j0[j]) > 1e-03:
				count += 1
		if count < 1: #k == MAXITER-1

			for element in range(len(u)):
				#Relative velocity
				u_proj = projection(u[element],mu,dim1,dim2)

				u_proj_contact = np.split(u_proj,dim2/dim1)
				u_contact = np.split(u[element],dim2/dim1)			

				u_count = 0.0
				for contact in range(dim2/dim1):
					if np.allclose(u_contact[contact], u_proj_contact[contact], rtol=0.1, atol=0.0):
						u_count += 1.0
			
				u_bin = 100 * u_count / (dim2/dim1)
				u_bin_plot.append(u_bin)

				#Reaction
				xi_proj = projection(-1.0 * xi[element],1/mu,dim1,dim2)

				xi_proj_contact = np.split(xi_proj,dim2/dim1)
				xi_contact = np.split(-1.0 * xi[element],dim2/dim1)			

				xi_count = 0.0
				for contact in range(dim2/dim1):
					if np.allclose(xi_contact[contact], xi_proj_contact[contact], rtol=0.1, atol=0.0):
						xi_count += 1.0
			
				xi_bin = 100 * xi_count / (dim2/dim1)
				xi_bin_plot.append(xi_bin)					

			for element in range(len(r_norm)):
				rho_plot.append(rho[element])
				b_plot.append(np.linalg.norm(b[element]))
	
			#R = -rho[k]*xi[k+1]
			#N1 = csc_matrix.dot(M, v[k+1]) - csc_matrix.dot(A_T, R) + f
			#N2 = u[k+1] - projection(u[k+1] - R, mu, dim1, dim2)  
			#N1_norm = np.linalg.norm(N1)
			#N2_norm = np.linalg.norm(N2)

			#print np.sqrt( N1_norm**2 + N2_norm**2 )	

			print b_plot[-1]
			print b[-1][:3]
			print b[-1][-3:]				
			break	
	
		#end rutine

	end = time.clock()

	####################
	## REPORTING DATA ##
	####################

	f, axarr = plt.subplots(4, sharex=True)
	f.suptitle('Internal update with vp_RR_He (Di Cairano)')

	axarr[0].semilogy(b_plot)
	axarr[0].set(ylabel='||Phi(s)||')

	axarr[1].plot(rho_plot)
	axarr[1].set(ylabel='Rho')

	axarr[2].semilogy(r_norm, label='||r||')
	axarr[2].semilogy(s_norm, label='||s||')
	axarr[2].legend()
	axarr[2].set(ylabel='Residuals')

	axarr[3].semilogy(siconos_plot)
	axarr[3].set(xlabel='Iteration', ylabel='SICONOS error')
	plt.show()

	plt.show()

	#print b[-1]
	#print np.linalg.norm(b[-1])
	#plotit(r,b,start,end,'With acceleration / Without restarting for '+problem_data+' for rho: '+rho_method)
	#plotit(r,s,start,end,'Internal update with vp_RR_He (Di Cairano)')

	time = end - start
	print 'Total time: ', time
	return time
