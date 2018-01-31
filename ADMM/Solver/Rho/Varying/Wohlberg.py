import numpy as np

def penalty(rho,r_norm,s_norm,mu):
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
