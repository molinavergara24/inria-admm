'''
% Solves the following problem via ADMM:
%
%   minimize     (1/2)*v'*M*v + f'*v + indicator(u)
%   subject to   u = Av + b
'''

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
from Solver.Rho.Optimal.ghadimi import *

#Max iterations and kind of tolerance
from Solver.Tolerance.iter_totaltolerance import *

#Stop criterion
from Solver.Tolerance.stop_criterion import *

#Acceleration
from Solver.Acceleration.minusr import *

#ADMM iteration
from Solver.ADMM_iteration.cp_R_iteration import *

#####################################################
############# TERMS / NOT A FUNCTION YET ############
#####################################################

M = problem.M.toarray()
f = problem.f
A = np.transpose(problem.H.toarray())
A_T = np.transpose(A)
mu = problem.mu
b = np.array([0,1,1]) * mu * 1

########################################
############# REQUIRE TERMS ############
########################################

#Problem size
n = np.shape(M)[0]
p = np.shape(b)[0]

#Set-up of vectors
v = [np.zeros([n,1])]
u = [np.array([0,0,0])] #this is u tilde, but in the notation of the paper is used as hat
u_hat = [np.array([0,0,0])] #u_hat[0] #in the notation of the paper this used with a underline
xi = [np.array([0,0,0])]
xi_hat = [np.array([0,0,0])]
r = [np.zeros([n,1])] #primal residual
s = [np.zeros([n,1])] #dual residual
r_norm = [0]
s_norm = [0]
tau = [1] #over-relaxation
e = [] #restart

#Optimal penalty parameter
rho = ghadimi(A,M,A_T)

########################################
## TERMS COMMON TO ALL THE ITERATIONS ##
########################################
start = time.clock()

#Cholesky factorization of M + rho * dot(M_T,M)
P = M + rho * np.dot(A_T,A)
L = np.linalg.cholesky(P)
L_T = np.transpose(L)

################
## ITERATIONS ##
################
for k in range(MAXITER):

	####################
	## ADMM iteration ##
	####################
	ADMM_iteration(f,rho,A_T,b,xi,u,L,L_T,A,v,mu,r,s,k,xi_hat,u_hat)

	####################
	## stop criterion ##
	####################
	if stopcriterion(A,A_T,v,u,b,xi,r,s,r_norm,s_norm,p,n,ABSTOL,RELTOL,rho,k) == 'break':
		break

	######################################
	## accelerated ADMM without restart ##
	######################################
	minusr(tau,u,u_hat,xi,xi_hat,k)

	#end rutine

end = time.clock()
####################
## REPORTING DATA ##
####################
plotit(r,s,start,end,'With acceleration / Without restarting')
