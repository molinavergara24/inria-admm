import numpy as np

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
'''

##########################
## FUNCTION DEFINITIONS ##
##########################

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

M = np.array([[1,0,0],[0,1,0],[0,0,1]])
f = np.array([1,1,1])
A = np.array([[1,0,0],[0,1,0],[0,0,1]])
b = np.array([1,1,1])

#######################################
############# SOLVER TERMS ############
#######################################

MAXITER = 3

########################################
############# REQUIRE TERMS ############
########################################

#Dimension of the problem
n = M.shape[1] #for the projection calculus is assumed as 3 and in the set-up of vectors also

#Set-up of vectors
v = np.zeros([n,1])
u = np.zeros([n,1])
u_hat = np.zeros([n,1])
xi = np.zeros([n,1])
xi_hat = np.zeros([n,1])

tau = np.zeros(MAXITER)
e = np.zeros(MAXITER)
print(v[1])
'''
#Value of u and xi
u[0] = np.array([0,0,0]) #this is u tilde, but in the notation of the paper is used as hat
u_hat[0] = u[0] #in the notation of the paper this used with a underline

xi[0] = np.array([0,0,0])
xi_hat[0] = xi[0]

#Value of parameters
tau[0] = 1
rho = 1
eta = 1

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

for k in range(MAXITER-1):

	################
	## v - update ##
	################

        RHS = -f + rho * np.dot(A_T, -b - xi_hat[k] + u_hat[k])
        v_cholesky = np.linalg.solve(L, RHS)
	v[k+1] = np.linalg.solve(L_T, v_cholesky)

	################
	## u - update ##
	################
	Av = np.dot(A,v[k+1])
	vector = Av + xi_hat[k] + b
	u[k+1]= projection(vector)

	#################
	## xi - update ##
	#################

	xi[k+1] = xi_hat[k] + Av - u[k+1] + b

	###################################
	## accelerated ADMM with restart ##
	###################################

	e[k] = np.square(np.linalg.norm(xi[k+1]-xi_hat[k])) + rho * np.square(np.linalg.norm(u[k+1]-u_hat[k]))

	if e[k] < eta * e[k-1]:
		tau[k+1] = 0.5 * (1 + np.sqrt(1 + 4 * np.square(tau[k])))
		alpha = (tau[k] - 1) / tau[k+1]
		u_hat[k+1] = u[k+1] + alpha * (u[k+1] - u[k])
		xi_hat[k+1] = xi[k+1] + alpha * (xi[k+1] - xi[k])
	else:
		tau[k+1] = 1
		u_hat[k+1] = u[k]
		xi_hat[k+1] = xi[k]
		e[k] = e[k-1] / eta

	#end rutine

####################
## REPORTING DATA ##
####################

import tabulate as tb

results = []
for i in range(len(v)):
	results.append(i,v[i])
	
print tb(results, headers=['Iteration', 'v'])

'''




