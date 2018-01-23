'''
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*v'*M*v + f'*v + indicator(u)
%   subject to   u = Av + b
%
% The solution is returned in the vector v.
%
% More information can be found in the paper linked at:
% [1] https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
% [2] https://link.springer.com/content/pdf/10.1023%2FA%3A1004603514434.pdf
% [3] https://link.springer.com/content/pdf/10.1023%2FA%3A1017522623963.pdf
'''

######################
## IMPORT LIBRARIES ##
######################

import numpy as np
import matplotlib.pyplot as plt
import time

##########################
## FUNCTION DEFINITIONS ##
##########################

#Projection onto second order cone
def projection(vector):
	normvec = np.linalg.norm(vector[1:])
	lambda_1 = vector[0] - normvec
	lambda_2 = vector[0] + normvec

	zero = 1e-12 #how close is to zero
	if np.linalg.norm(vector[1:]) > zero:
		e_1 = vector[1]/normvec
		e_2 = vector[2]/normvec
		u_1 = 0.5 * np.array([1,-e_1,-e_2])
		u_2 = 0.5 * np.array([1,e_1,e_2])		
	else:
		e = np.sqrt(2)*0.5
		u_1 = 0.5 * np.array([1,-e,-e])
		u_2 = 0.5 * np.array([1,e,e])	

	return np.maximum(0,lambda_1) * u_1 + np.maximum(0,lambda_2) * u_2

#####################################################
############# TERMS / NOT A FUNCTION YET ############
#####################################################

#3 balls falling in a vertical line
#mass=1,radius=1,gravity=9.8

#M = np.array([[1,0,0],[0,1,0],[0,0,1]]) #mass matrix
#f = np.array([-9.8,-9.8,-9.8]) #external forces
#A = np.array([[1,0,0],[-1,1,0],[-1,-1,1]]) #position terms
#b = np.array([-1,-2,-3]) #radius terms

M = np.array([[1,0,0],[0,5,0],[0,0,25]]) #mass matrix
f = np.array([-9.8,-9.8,-9.8]) #external forces
A = np.array([[1,0,0],[-1,1,0],[-1,-1,1]]) #position terms
b = np.array([-1,-3,-4]) #radius terms

#######################################
############# SOLVER TERMS ############
#######################################

MAXITER = 1000
ABSTOL = 1e-04
RELTOL = 1e-04

ITER = []
TIMING = []

for i in range(14):

	########################################
	############# REQUIRE TERMS ############
	########################################

	#Set-up of vectors
	v = []

	u = []

	xi = []

	r = [] #primal residual
	s = [] #dual residual
	e = [] #restart

	#Value of v, u and xi
	v.append(np.array([0,0,0])) #v[0]

	u.append(np.array([0,0,0])) #u[0] #this is u tilde, but in the notation of the paper is used as hat

	xi.append(np.array([0,0,0])) #xi[0]

	r.append(np.array([0,0,0])) #r[0]
	s.append(np.array([0,0,0])) #s[0]

	#Value of parameters
	eta = 1
		
	#Varying penalty parameter (comparison with paper [2])
	rho = 10**(-5+i) #augmented Lagrangian penalty parameter
	
	########################################
	## TERMS COMMON TO ALL THE ITERATIONS ##
	########################################

	#Transpose matrix of M
	M_T = np.transpose(M)

	#Cholesky factorization of M + rho * dot(M_T,M)
	A_T = np.transpose(A)
	L = np.linalg.cholesky(M + rho * np.dot(A_T,A))
	L_T = np.transpose(L)

	################
	## ITERATIONS ##
	################

	start = time.clock()

	for k in range(MAXITER):

		################
		## v - update ##
		################
		RHS = -f + rho * np.dot(A_T, -b - xi[k] + u[k])
		v_cholesky = np.linalg.solve(L, RHS)
		v.append(np.linalg.solve(L_T, v_cholesky)) #v[k+1]

		################
		## u - update ##
		################
		Av = np.dot(A,v[k+1])
		vector = Av + xi[k] + b
		u.append(projection(vector)) #u[k+1]

		########################
		## residuals - update ##
		########################
		s.append(rho * np.dot(A_T,(u[k+1]-u[k]))) #s[k+1]
		r.append(Av - u[k+1] + b) #r[k+1]

		#################
		## xi - update ##
		#################
		xi.append(xi[k] + r[k+1]) #xi[k+1]

		####################
		## stop criterion ##
		####################
		pri_evalf = np.array([np.linalg.norm(v[k]),np.linalg.norm(u[k]),np.linalg.norm(b)])
		eps_pri = np.sqrt(3)*ABSTOL + RELTOL*np.amax(pri_evalf)

		dual_evalf = rho * np.dot(A_T,xi[k])
		eps_dual = np.sqrt(3)*ABSTOL + RELTOL*np.linalg.norm(dual_evalf)

		r_norm = np.linalg.norm(r[k+1])
		s_norm = np.linalg.norm(s[k+1])
		
		end = time.clock()

		if r_norm<=eps_pri and s_norm<=eps_dual:
			break
		
		#end rutine

	####################
	## REPORTING DATA ##
	####################
	
	ITER.append(len(r)-1)
	TIMING.append((end-start)*10**3)

################
## PRINT DATA ##
################

for j in range(14):
	print 'Rho:',10**(-5+j),', iterations:',ITER[j],', total time:',TIMING[j]
