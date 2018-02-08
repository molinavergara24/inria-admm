'''
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*v'*M*v + f'*v + indicator(u)
%   subject to   u = Av + b
'''

def vp_R_He(problem_data, rho_method):

	######################
	## IMPORT LIBRARIES ##
	######################

	#Math libraries
	import numpy as np

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

	#Stop criterion
	from Solver.Tolerance.stop_criterion_vp import *

	#Acceleration
	from Solver.Acceleration.minusr import *

	#Varying penalty parameter
	from Solver.Rho.Varying.He import *

	#ADMM iteration
	from Solver.ADMM_iteration.vp_R_iteration import *

	#b = Es matrix
	from Data.Es_matrix import *

	##################################
	############# REQUIRE ############
	##################################

	problem = hdf5_file(problem_data)

	M = (problem.M.toarray() + np.transpose(problem.M.toarray()))/2.0 #Symmetric part
	f = problem.f
	A = np.transpose(problem.H.toarray())
	A_T = np.transpose(A)
	w = problem.w
	mu = problem.mu

	dim1 = 3 #dimensions (normal,tangential,tangential)
	dim2 = w.shape[0]
	b = Es_matrix(w,mu,1)

	#################################
	############# SET-UP ############
	#################################

	#Problem size
	n = np.shape(M)[0]
	p = np.shape(b)[0]

	#Set-up of vectors
	v = [np.zeros([n,1])]
	u = [np.zeros([dim2,])] #this is u tilde, but in the notation of the paper is used as hat [np.zeros([10,0])]
	u_hat = [np.zeros([dim2,])] #u_hat[0] #in the notation of the paper this used with a underline
	xi = [np.zeros([dim2,])] 
	xi_hat = [np.zeros([dim2,])]
	r = [np.zeros([n,1])] #primal residual
	s = [np.zeros([n,1])] #dual residual
	r_norm = [0]
	s_norm = [0]
	tau = [1] #over-relaxation
	e = [] #restart
	rho = []

	#Optimal penalty parameter
	rho_string = 'Solver.Rho.Optimal.' + rho_method + '(A,M,A_T)'
	rh = eval(rho_string)
	rho.append(rh) #rho[0]
	rho.append(rh) #rho[1]

	################
	## ITERATIONS ##
	################
	start = time.clock()

	for k in range(MAXITER):

		####################
		## ADMM iteration ##
		####################
		ADMM_iteration(f,rho,A_T,b,xi,u,A,v,mu,r,s,xi_hat,u_hat,M,k,dim1,dim2)

		####################
		## stop criterion ##
		####################
		if stopcriterion(A,A_T,v,u,b,xi,r,s,r_norm,s_norm,p,n,ABSTOL,RELTOL,rho,k) == 'break':
			break	

		######################################
		## accelerated ADMM without restart ##
		######################################
		minusr(tau,u,u_hat,xi,xi_hat,k)

		################################
		## penalty parameter - update ##
		################################
		rho.append(penalty(rho[k+1],r_norm[k+1],s_norm[k+1]))

		#end rutine

	end = time.clock()
	####################
	## REPORTING DATA ##
	####################
	#plotit(r,s,start,end,'With acceleration / Without restarting'+problem_data)

	time = end - start
	return time
