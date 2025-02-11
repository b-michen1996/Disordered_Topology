import kwant
import numpy as np
import os
import gc
import sys

def make_system_DC_CI_3_terminal(Nx, Ny, Nx_lead_start, Nx_lead_stop, r_1, r_2, W, t_lead):
	"""Creates system with the given parameters and then adds random impurities."""
	# values for lattice constant and number of orbitals
	a = 1
	
	# matrices acting in orbital space to build the system
	sigma_x = np.array([[0, 1],
				[1, 0]])
	
	sigma_y = np.array([[0, -1j],
				[1j, 0]])
	
	sigma_z = np.array([[1, 0],
				[0, -1]])
	
	sigma_0 = np.array([[1, 0],
				[0, 1]])	
	
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	
	lat = kwant.lattice.square(a, norbs = 2)
	syst = kwant.Builder()

	# initialize by setting zero on-site terms
	syst[(lat(jx,jy) for jx in range (Nx) for jy in range(Ny))] = 0 * sigma_0

	# hoppings and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy

					# set disorder
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + ((r_1 + r_2)/2) * sigma_z

					# set hoppings 
					if jx < (Nx-1):                
						syst[lat(jx,jy), lat(jx+1,jy)] = (1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z 
					
					if jx < (Nx-2):                
						syst[lat(jx,jy), lat(jx+2,jy)] = - (1/2) * sigma_z
						  													  							 
					if jy < (Ny-1):    
						syst[lat(jx,jy), lat(jx,jy+1)] = (1/(2j)) * sigma_y - (1/2) * sigma_z
	
	# construct left lead
	lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
		
	#initialize with on-site terms to avoid error 
	lead[(lat(0,jy) for jy in range(Ny))] = 0 * sigma_0
	lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead * sigma_0
	#lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead * sigma_0
		
	# attach left lead 
	syst.attach_lead(lead)
	
	# reverse left lead and attach it on the right
	syst.attach_lead(lead.reversed())
	
	# construct top lead
	lead_top = kwant.Builder(kwant.TranslationalSymmetry((0, a)))
	
	lead_top[(lat(jx, Ny) for jx in range(Nx_lead_start, Nx_lead_stop))] = 0 * sigma_0
	lead_top[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead * sigma_0
	
	# attach top lead
	syst.attach_lead(lead_top)
	
	#kwant.plotter.plot(syst)
											 
	return syst


def export_conductance_DC_CI_3_terminal(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, Nx_lead_start, Nx_lead_stop, r_1, r_2, t_lead, run_nr):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "G_DC_CI_3T_results/DC_CI_3T_run_" + str(run_nr)
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	# generate and save va
	E_vals = np.linspace(E_min, E_max, n_E)
	W_vals = np.linspace(W_min, W_max, n_W)
	
	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')  
	np.savetxt(foldername + "/W_vals.txt", W_vals, delimiter=' ')  
	
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("W_min " + str(W_min) + str("\n"))
	f_parameter.write("W_max " + str(W_max) + str("\n"))
	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	
	f_parameter.write("n_W " + str(n_W) + str("\n"))
	f_parameter.write("n_E " + str(n_E) + str("\n"))
	f_parameter.write("n_inst " + str(n_inst) + str("\n"))
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))	
	f_parameter.write("Nx_lead_start " + str(Nx_lead_start) + str("\n"))
	f_parameter.write("Nx_lead_stop " + str(Nx_lead_stop) + str("\n"))
	
	f_parameter.write("r_1 " + str(r_1) + str("\n"))	
	f_parameter.write("r_2 " + str(r_2) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
				
	f_parameter.close()
					
	result_L_R = np.zeros((n_E, n_W))
	result_R_L = np.zeros((n_E, n_W))
	result_L_T = np.zeros((n_E, n_W))
	result_T_L = np.zeros((n_E, n_W))
	result_R_T = np.zeros((n_E, n_W))
	result_T_R = np.zeros((n_E, n_W))
		
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):
			# generate System
			syst = make_system_DC_CI_3_terminal(Nx, Ny, Nx_lead_start, Nx_lead_stop, r_1, r_2, W_curr, t_lead)
			syst = syst.finalized()
			
			# calculate transmission for all energy values
			data_j_inst_L_R = np.zeros(n_E)			
			data_j_inst_R_L = np.zeros(n_E)	
			data_j_inst_L_T = np.zeros(n_E)			
			data_j_inst_T_L = np.zeros(n_E)			
			data_j_inst_R_T = np.zeros(n_E)			
			data_j_inst_T_R = np.zeros(n_E)			
			
			for j_E in range(n_E):
				energy = E_vals[j_E]
				smatrix = kwant.smatrix(syst, energy)					
				data_j_inst_L_R[j_E] = smatrix.transmission(1, 0)
				data_j_inst_R_L[j_E] = smatrix.transmission(0, 1)

				data_j_inst_L_T[j_E] = smatrix.transmission(2, 0)
				data_j_inst_T_L[j_E] = smatrix.transmission(0, 2)

				data_j_inst_R_T[j_E] = smatrix.transmission(2, 1)
				data_j_inst_T_R[j_E] = smatrix.transmission(1, 2)

				sys.stdout.write(f"j_inst = {j_inst + 1} of {n_inst}, j_E = {j_E + 1} of {n_E} \n")
				sys.stdout.flush()
			
			# collect garbage
			del syst
			gc.collect()
									
			result_L_R[:, j_W] += (1 / n_inst) * data_j_inst_L_R
			result_R_L[:, j_W] += (1 / n_inst) * data_j_inst_R_L 

			result_L_T[:, j_W] += (1 / n_inst) * data_j_inst_L_T
			result_T_L[:, j_W] += (1 / n_inst) * data_j_inst_T_L 

			result_R_T[:, j_W] += (1 / n_inst) * data_j_inst_R_T
			result_T_R[:, j_W] += (1 / n_inst) * data_j_inst_T_R 
		
		np.savetxt(foldername + "/result_L_R.txt", result_L_R, delimiter=' ')  
		np.savetxt(foldername + "/result_R_L.txt", result_R_L, delimiter=' ')  
		np.savetxt(foldername + "/result_L_T.txt", result_L_T, delimiter=' ')  
		np.savetxt(foldername + "/result_T_L.txt", result_T_L, delimiter=' ')  
		np.savetxt(foldername + "/result_R_T.txt", result_R_T, delimiter=' ')  
		np.savetxt(foldername + "/result_T_R.txt", result_T_R, delimiter=' ')  
								
	return 
	

def main():
	W_min = 0.
	W_max = 4

	E_min = -1.5
	E_max = 1.5
	
	n_W = 41
	n_E = 65
	n_inst = 2

	Nx = 60
	Ny = 60
	
	Nx_lead_start = 0
	Nx_lead_stop = Nx
	
		
	r_1 = 1.9
	r_2 = 1
	
	t_lead = 5
	
	
	run_nr = 5
	
	W = 0
	
	export_conductance_DC_CI_3_terminal(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, Nx_lead_start, Nx_lead_stop, r_1, r_2, t_lead, run_nr)
	
	
if __name__ == '__main__':
	main()




	