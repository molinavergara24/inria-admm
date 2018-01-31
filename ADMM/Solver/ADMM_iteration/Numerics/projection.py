import numpy as np

def projection(vector,mu):
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

