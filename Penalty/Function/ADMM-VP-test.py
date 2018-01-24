'''
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*v'*M*v + f'*v + indicator(u)
%   subject to   u = Av + b
%
% The solution is returned in the vector v.
%
% More information can be found in the paper linked at:
% https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
% https://link.springer.com/content/pdf/10.1023%2FA%3A1004603514434.pdf
% https://link.springer.com/content/pdf/10.1023%2FA%3A1017522623963.pdf
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

#Penalty parameter
def penalty(rho,r_norm,s_norm):
	mu = 10.0
	factor = 2.0

	if r_norm > mu * s_norm:
		return rho*factor
	if s_norm > mu * r_norm:
		return rho/factor
	else:
		return rho

#####################################################
############# TERMS / NOT A FUNCTION YET ############
#####################################################

#3 balls in a vertical line
#mass=1,radius=1,gravity=9.8

#M = np.array([[1,0,0],[0,1,0],[0,0,1]]) #mass matrix
#f = np.array([-9.8,-9.8,-9.8]) #external forces
#A = np.array([[1,0,0],[-1,1,0],[-1,-1,1]]) #position terms
#b = np.array([-1,-2,-3]) #radius terms

mass = 100

M = np.array([[1,0,0],[0,1,0],[0,0,1]]) * mass #mass matrix
f = np.array([-9.8,-9.8,-9.8]) * mass #external forces
A = np.array([[1,0,0],[-1,1,0],[-1,-1,1]]) #position terms
b = np.array([-1,-2,-3]) #radius terms

#######################################
############# SOLVER TERMS ############
#######################################

MAXITER = 1000
ABSTOL = 1e-04
RELTOL = 1e-04

ITER = []
TIMING = []

POWERVAR = 14
for i in range(POWERVAR):

	########################################
	############# REQUIRE TERMS ############
	########################################

	#Set-up of vectors
	v = []

	u = []
	u_hat = []

	xi = []
	xi_hat = []

	r = [] #primal residual
	s = [] #dual residual
	tau = [] #over-relaxation
	e = [] #restart

	rho = [] #augmented Lagrangian penalty parameter

	#Value of v, u and xi
	v.append(np.array([0,0,0])) #v[0]

	u.append(np.array([0,0,0])) #u[0] #this is u tilde, but in the notation of the paper is used as hat
	u_hat.append(u[0]) #u_hat[0] #in the notation of the paper this used with a underline

	xi.append(np.array([0,0,0])) #xi[0]
	xi_hat.append(xi[0]) #xi_hat[0]

	r.append(np.array([0,0,0])) #r[0]
	s.append(np.array([0,0,0])) #s[0]
	tau.append(1) #tau[0]

	#Value of parameters
	eta = 1

	#Varying penalty parameter (comparison with paper [2])
	rho.append(10**(-5+i)) #rho[0]
	rho.append(10**(-5+i)) #rho[1]

	########################################
	## TERMS COMMON TO ALL THE ITERATIONS ##
	########################################

	#Transpose matrix of M and A
	M_T = np.transpose(M)
	A_T = np.transpose(A)

	################
	## ITERATIONS ##
	################

	start = time.clock()

	for k in range(MAXITER):
	
		#Cholesky factorization of M + rho * dot(A_T,A)
		L = np.linalg.cholesky(M + rho[k] * np.dot(A_T,A))
		L_T = np.transpose(L)

		################
		## v - update ##
		################
		RHS = -f + rho[k] * np.dot(A_T, -b - xi_hat[k] + u_hat[k])
		v_cholesky = np.linalg.solve(L, RHS)
		v.append(np.linalg.solve(L_T, v_cholesky)) #v[k+1]

		################
		## u - update ##
		################
		Av = np.dot(A,v[k+1])
		vector = Av + xi_hat[k] + b
		u.append(projection(vector)) #u[k+1]

		########################
		## residuals - update ##
		########################
		s.append(rho[k] * np.dot(A_T,(u[k+1]-u_hat[k]))) #s[k+1]
		r.append(Av - u[k+1] + b) #r[k+1]

		#################
		## xi - update ##
		#################
		ratio = rho[k]/rho[k+1] #update of dual scaled variable with new rho
		xi.append(ratio * (xi_hat[k] + r[k+1])) #xi[k+1]

		####################
		## stop criterion ##
		####################
		pri_evalf = np.array([np.linalg.norm(np.dot(A,v[k+1])),np.linalg.norm(u[k+1]),np.linalg.norm(b)])
		eps_pri = np.sqrt(3)*ABSTOL + RELTOL*np.amax(pri_evalf)

		dual_evalf = rho[k] * np.dot(A_T,xi[k+1])
		eps_dual = np.sqrt(3)*ABSTOL + RELTOL*np.linalg.norm(dual_evalf)

		r_norm = np.linalg.norm(r[k+1])
		s_norm = np.linalg.norm(s[k+1])
		if r_norm<=eps_pri and s_norm<=eps_dual:
			break

		###################################
		## accelerated ADMM with restart ##
		###################################
		e.append(np.square(np.linalg.norm(xi[k+1]-xi_hat[k])) + rho * np.square(np.linalg.norm(u[k+1]-u_hat[k]))) #e[k]

		if e[k] < eta * e[k-1]:
			tau.append(0.5 * (1 + np.sqrt(1 + 4 * np.square(tau[k])))) #tau[k+1]
			alpha = (tau[k] - 1) / tau[k+1]
			u_hat.append(u[k+1] + alpha * (u[k+1] - u[k])) #u_hat[k+1]
			xi_hat.append(xi[k+1] + alpha * (xi[k+1] - xi[k])) #xi_hat[k+1]
		else:
			tau.append(1) #tau[k+1]
			u_hat.append(u[k]) #u_hat[k+1]
			xi_hat.append(xi[k]) #xi_hat[k+1]
			e[k] = e[k-1] / eta

		################################
		## penalty parameter - update ##
		################################
		rho.append(penalty(rho[k+1],r_norm,s_norm))
		#rho.append(10**(-5+i))	
	
		#end rutine

	end = time.clock()

	####################
	## REPORTING DATA ##
	####################
	
	ITER.append(len(r)-1)
	TIMING.append((end-start)*10**3)

################
## PRINT DATA ##
################

for j in range(POWERVAR):
	print 'Rho:',10**(-5+j),', iterations:',ITER[j],', total time [ms]:',TIMING[j]

