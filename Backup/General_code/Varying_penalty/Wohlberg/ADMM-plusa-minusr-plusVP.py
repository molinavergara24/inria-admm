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
from read_fclib import *

##########################
## FUNCTION DEFINITIONS ##
##########################

#Projection onto dual second order cone
def projection(vector):
	normx2 = np.linalg.norm(vector[1:])
	x1 = vector[0]

	if normx2 <= (-1/mu)*x1:
		project =  0		
	if normx2 <= (1/mu)*x1:
		project = vector	
	else:
		x2 = vector[1:]
		project = (mu**2)/(1+mu**2) * (x1 + (1/mu)*normx2) * np.concatenate([np.array([1]),(1/mu)*x2*(1/normx2)])

	return project

#Penalty parameter
def penalty(rho,r_norm,s_norm):
	mu = 10.0
	factormax = 2.0

	ratiosqrt = np.sqrt(r_norm / s_norm)
	if 1.0 <= ratiosqrt and ratiosqrt < factormax:
		factor = ratiosqrt
	if 1.0/factormax < ratiosqrt and ratiosqrt < 1.0:
		factor = 1.0/ratiosqrt
	else:
		factor = factormax	
	
	if r_norm > mu * s_norm:
		rhos = rho*factor
	if s_norm > mu * r_norm:
		rhos = rho/factor
	else:
		rhos = rho

	return rhos

#####################################################
############# TERMS / NOT A FUNCTION YET ############
#####################################################

M = problem.M.toarray()
f = problem.f
A = np.transpose(problem.H.toarray())
mu = problem.mu
b = np.array([0,1,1]) * mu * 1

#######################################
############# SOLVER TERMS ############
#######################################

MAXITER = 1000
ABSTOL = 0
RELTOL = 1e-04

########################################
############# REQUIRE TERMS ############
########################################

#Set-up of vectors
n = np.shape(M)[0]

v = [np.zeros([n,1])]

u = [np.array([0,1,1])] #this is u tilde, but in the notation of the paper is used as hat
u_hat = [np.array([0,1,1])]

xi = [np.array([0,1,1])]
xi_hat = [np.array([0,1,1])]

r = [np.zeros([n,1])] #primal residual
s = [np.zeros([n,1])] #dual residual

#Parameters
rho = []
tau = []
tau.append(1) #tau[0]
eta = 1

#Transpose matrix of A
A_T = np.transpose(A)

#Optimal penalty parameter by Ghadimi
DUAL = np.dot(np.dot(A,np.linalg.inv(M)),A_T)
eig = np.linalg.eigvals(DUAL)
rh = 1 / np.sqrt(np.amax(eig)*np.amin(eig))
rho.append(rh) #rho[0]
rho.append(rh) #rho[1]

################
## ITERATIONS ##
################

start = time.clock()

for k in range(MAXITER):

	#Cholesky factorization of M + rho * dot(M_T,M)
	P = M + rho[k] * np.dot(A_T,A)
	L = np.linalg.cholesky(P)
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
	xi.append(ratio*(xi_hat[k] + r[k+1])) #xi[k+1]

	####################
	## stop criterion ##
	####################
	pri_evalf = np.array([np.linalg.norm(np.dot(A,v[k+1])),np.linalg.norm(u[k+1]),np.linalg.norm(b)])
	eps_pri = np.sqrt(3)*ABSTOL + RELTOL*np.amax(pri_evalf)

	dual_evalf = rho[k] * np.dot(A_T,xi[k+1])
	eps_dual = np.sqrt(3)*ABSTOL + RELTOL*np.linalg.norm(dual_evalf)

	r_norm = np.linalg.norm(r[k+1])
	s_norm = np.linalg.norm(s[k+1])

	end = time.clock()

	if r_norm<=eps_pri and s_norm<=eps_dual:
		break

	###################################
	## accelerated ADMM with restart ##
	###################################

	tau.append(0.5 * (1 + np.sqrt(1 + 4 * np.square(tau[k])))) #tau[k+1]
	alpha = (tau[k] - 1) / tau[k+1]
	u_hat.append(u[k+1] + alpha * (u[k+1] - u[k])) #u_hat[k+1]
	xi_hat.append(xi[k+1] + alpha * (xi[k+1] - xi[k])) #xi_hat[k+1]

	################################
	## penalty parameter - update ##
	################################
	rho.append(penalty(rho[k+1], r_norm/np.amax(pri_evalf), s_norm/np.linalg.norm(dual_evalf)))

	#end rutine

####################
## REPORTING DATA ##
####################

R = [np.linalg.norm(k) for k in r[1:]]
S = [np.linalg.norm(k) for k in s[1:]]
plt.plot(R, label='||r||')
plt.hold(True)
plt.plot(S, label='||s||')
plt.hold(True)
plt.ylabel('Residuals')
plt.xlabel('Iteration')
#plt.xlim(xmax=30)
plt.text(len(r)/2,3*(np.amax(S)+np.amax(R))/4,'N_iter = '+str(len(r)-1))
plt.text(len(r)/2,2*(np.amax(S)+np.amax(R))/4,'Total time = '+str((end-start)*10**3)+' ms')
plt.text(len(r)/2,1*(np.amax(S)+np.amax(R))/4,'Time_per_iter = '+str(((end-start)/(len(r)-1))*10**3)+' ms')
plt.title('With acceleration / Without restarting')
plt.legend()
plt.show()
