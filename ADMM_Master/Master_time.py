#########################################################
#################### MASTER FUNCTION ####################
#########################################################

import ADMM
def master(solver, problem, rho_method):
	STRING = 'ADMM.' + solver + '("' + problem + '", "' + rho_method + '")'
	return eval(STRING)

#########################################################
###################### IMPORT DATA ######################
#########################################################

#Import all the problems hdf5
import os 
all_problems = os.listdir("ADMM/Data/box_stacks/")
all_problems.sort()

#Import all the solvers
all_solvers = ['cp_N', 'cp_R', 'cp_RR', 'vp_N_He', 'vp_R_He', 'vp_RR_He', 'vp_N_Spectral', 'vp_R_Spectral', 'vp_RR_Spectral', 'vp_N_Wohlberg', 'vp_R_Wohlberg', 'vp_RR_Wohlberg']

#########################################################
######################### CODE ##########################
#########################################################

#Import librearies
import numpy as np
import pickle

#Definition of list
dict_master = []
rho_optimal = ['acary','dicairano','ghadimi','normal']

#Time problem/solver
for each_problem in all_problems:
	print(each_problem)
	dict_problem = []		
	for each_solver in all_solvers:
		print(each_solver)
		dict_solver = {'problem': each_problem, 'solver': each_solver}
		for each_rho in rho_optimal:
			try:
				timing = master(each_solver, each_problem, each_rho)
			except:
				timing = np.nan #NaN
			dict_solver[each_rho+' (time)'] = timing
		dict_problem.append(dict_solver)
	dict_master.append(dict_problem)

#Save the data
pickle.dump(dict_master, open("time_solver.p", "wb"))
