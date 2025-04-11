import os
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import kwant
import numpy as np
import gc
import sys


sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])
sigma_0 = np.array([[1, 0],[0, 1]])	


def make_system_DC_CI_V2_4_terminal(Nx, Ny, x_lead_start, x_lead_stop, y_lead_start, y_lead_stop, r, epsilon_1, epsilon_2, gamma, W, t_lead, t_lead_trans = 0.):
	"""Creates system with the given parameters and then adds random impurities."""
	# values for lattice constant and number of orbitals
	a = 1
	
	Nx = int(Nx)
	Ny = int(Ny)
	x_lead_start = int(x_lead_start)
	y_lead_start = int(y_lead_start)
	x_lead_stop = int(x_lead_stop)
	y_lead_stop = int(y_lead_stop)
	
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	
	lat = kwant.lattice.square(a, norbs = 2)
	syst = kwant.Builder()
	
	mat_os = r * sigma_z
	hop_mat_dx = (gamma * (epsilon_1 + epsilon_2/2)/(2j)) * sigma_x
	hop_mat_2dx = - (1/2) * sigma_z -(gamma * epsilon_2/(8j)) * sigma_x
	hop_mat_dy = (gamma * (epsilon_1 + epsilon_2/2)/(2j)) * sigma_y - ((epsilon_1 + epsilon_2/2) / 2) * sigma_z
	hop_mat_dx_dy = -(gamma * epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z
	hop_mat_dx_mdy = +(gamma * epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z
		
	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy					
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + mat_os
								
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_mat_dx
	syst[kwant.builder.HoppingKind((2, 0), lat, lat)] = hop_mat_2dx
	
	# Hopping in y-direction
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_mat_dy

	# Hopping in (x+y)-direction
	syst[kwant.builder.HoppingKind((1, 1), lat, lat)] = hop_mat_dx_dy
	
	# Hopping in (x-y)-direction
	syst[kwant.builder.HoppingKind((1, -1), lat, lat)] = hop_mat_dx_mdy
	
	# construct left lead
	lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
		
	#initialize with on-site terms to avoid error 
	lead[(lat(0,jy) for jy in range(x_lead_start, x_lead_stop))] = 0 * sigma_0
	lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead * sigma_0
	lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead_trans * sigma_0
				
	# attach left lead 
	syst.attach_lead(lead)
	
	# reverse left lead and attach it on the right
	syst.attach_lead(lead.reversed())
	
	# construct top lead
	lead_top = kwant.Builder(kwant.TranslationalSymmetry((0, a)))
	
	lead_top[(lat(jx, Ny) for jx in range(y_lead_start, y_lead_stop))] = 0 * sigma_0
	lead_top[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead * sigma_0
	lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead_trans * sigma_0
	
	# attach top lead
	syst.attach_lead(lead_top)
	
	# reverse top lead and attach it on the bottom
	syst.attach_lead(lead_top.reversed())
		
	#kwant.plotter.plot(syst)
											 
	return syst



def make_system_DC_CI_V3_4_terminal(Nx, Ny, x_lead_start, x_lead_stop, y_lead_start, y_lead_stop, r, epsilon_1, epsilon_2, gamma, gamma_2, W, t_lead, t_lead_trans = 0.):
	"""Creates system with the given parameters and then adds random impurities."""
	# values for lattice constant and number of orbitals
	a = 1
	
	Nx = int(Nx)
	Ny = int(Ny)
	x_lead_start = int(x_lead_start)
	y_lead_start = int(y_lead_start)
	x_lead_stop = int(x_lead_stop)
	y_lead_stop = int(y_lead_stop)

	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	
	lat = kwant.lattice.square(a, norbs = 2)
	syst = kwant.Builder()
	
	mat_os = gamma_2 * r * sigma_z
	hop_mat_dx = gamma/(2j) * sigma_x
	hop_mat_2dx = - gamma_2 *(1/2) * sigma_z
	hop_mat_dy = ((epsilon_1 + epsilon_2/2)/(2j)) * sigma_y - ((epsilon_1 + epsilon_2/2) / 2) * sigma_z
	hop_mat_dx_dy = -(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z
	hop_mat_dx_mdy = +(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z
		
	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy					
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + mat_os
								
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_mat_dx
	syst[kwant.builder.HoppingKind((2, 0), lat, lat)] = hop_mat_2dx
	
	# Hopping in y-direction
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_mat_dy

	# Hopping in (x+y)-direction
	syst[kwant.builder.HoppingKind((1, 1), lat, lat)] = hop_mat_dx_dy
	
	# Hopping in (x-y)-direction
	syst[kwant.builder.HoppingKind((1, -1), lat, lat)] = hop_mat_dx_mdy
	
	# construct left lead
	lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
		
	#initialize with on-site terms to avoid error 
	lead[(lat(0,jy) for jy in range(x_lead_start, x_lead_stop))] = 0 * sigma_0
	lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead * sigma_0
	lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead_trans * sigma_0
				
	# attach left lead 
	syst.attach_lead(lead)
	
	# reverse left lead and attach it on the right
	syst.attach_lead(lead.reversed())
	
	# construct top lead
	lead_top = kwant.Builder(kwant.TranslationalSymmetry((0, a)))
	
	lead_top[(lat(jx, Ny) for jx in range(y_lead_start, y_lead_stop))] = 0 * sigma_0
	lead_top[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead * sigma_0
	lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead_trans * sigma_0
	
	# attach top lead
	syst.attach_lead(lead_top)
	
	# reverse top lead and attach it on the bottom
	syst.attach_lead(lead_top.reversed())
		
	#kwant.plotter.plot(syst)
											 
	return syst


def get_conductance(cond_mat):
	"""Get elements of the conductance tensor from 4 lead conduction matrix for current between lead 0 and 1 and lead 2 and 3. Lead 0 to 1. 
	Starting counterclockwise from the left edge of a square sample, the leads are assumed to be attached in the order 0,1,2,3."""
	
	# Set V_3 = 0 and solve I = G V for I_0 = - I_1 = 1 and I_2 = 0, which implies I_3 = 0. This yields
	# V_0, V_1, and V_2, from which the conductance elements are calculated as as sigma_xy = I_0 / (V_3 - V_2) = -1 / V_2 
	# and sigma_xx = I_0 / (V_0 - V_1)
	cond_mat_truc_i = cond_mat[:-1, :-1]
	V_i = np.linalg.solve(cond_mat_truc_i, [1,-1, 0])   
	sigma_xx = 1 / (V_i[0] - V_i[1])
	sigma_xy = -1 / V_i[2]
	
	
	# Set V_0 = 0 and solve I = G V for I_2 = - I_3 = 1 and I_1 = 0, which implies I_0 = 0. This yields
	# V_1, V_2, and V_3, from which the transverse conductance is calculated as as sigma_yx = I_2 / (V_1 - V_0) = -1 / V_1 and
	# sigma_yy = I_2 / (V_2 - V_3)
	cond_mat_truc_ii = cond_mat[1:, 1:]
	V_ii = np.linalg.solve(cond_mat_truc_ii, [0, 1,-1])
	sigma_yy = 1 / (V_ii[1] - V_ii[2])
	sigma_yx = -1 / V_ii[0]
	
	return sigma_xx, sigma_xy, sigma_yx, sigma_yy


def export_Hall_cond_DC_CI_V2(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, Nx_lead_start, Nx_lead_stop, Ny_lead_start, Ny_lead_stop,  r, epsilon_1, epsilon_2, gamma, t_lead, run_nr, t_lead_trans = 0.):
	"""Compute 4-terminal conductance matrix for disorder realizations of the DC_CI V2 model and export"""
	# create folder
	foldername = "G_DC_CI_V2_Hall_cond_results/G_DC_CI_V2_Hall_cond_run_" + str(run_nr)

	print(f"Starting G_DC_CI_V2_Hall_cond run {run_nr}")
	
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
	f_parameter.write("x_lead_start " + str(Nx_lead_start) + str("\n"))
	f_parameter.write("x_lead_stop " + str(Nx_lead_stop) + str("\n"))
	f_parameter.write("y_lead_start " + str(Ny_lead_start) + str("\n"))
	f_parameter.write("y_lead_stop " + str(Ny_lead_stop) + str("\n"))
	
	f_parameter.write("r " + str(r) + str("\n"))	
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))	
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("t_lead_trans " + str(t_lead_trans) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	
	f_parameter.write("Version V2" + str("\n"))
				
	f_parameter.close()

	result = np.zeros((n_E, 4, 4, n_W))
	result_sigma_xx = np.zeros((n_E, n_W))
	result_sigma_xy = np.zeros((n_E, n_W))
	result_sigma_yx = np.zeros((n_E, n_W))
	result_sigma_yy = np.zeros((n_E, n_W))	
							
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):
			# generate System
			syst = make_system_DC_CI_V2_4_terminal(Nx, Ny, Nx_lead_start, Nx_lead_stop, Ny_lead_start, Ny_lead_stop, r, epsilon_1, epsilon_2, gamma, W_curr, t_lead, t_lead_trans)
			syst = syst.finalized()
			
			# calculate conductance matrix for all energy values						
			for j_E in range(n_E):
				energy = E_vals[j_E]
				smatrix = kwant.smatrix(syst, energy)					
				
				cond_mat_curr = smatrix.conductance_matrix()
				
				result[j_E,:,:, j_W] += (1 / n_inst) * cond_mat_curr
								
				sigma_xx, sigma_xy, sigma_yx, sigma_yy = get_conductance(cond_mat_curr)

				result_sigma_xx[j_E, j_W] = (1 / n_inst) * sigma_xx
				result_sigma_xy[j_E, j_W] = (1 / n_inst) * sigma_xy
				result_sigma_yx[j_E, j_W] = (1 / n_inst) * sigma_yx
				result_sigma_yy[j_E, j_W] = (1 / n_inst) * sigma_yy								

			np.savetxt(foldername + "/result_sigma_xx.txt", result_sigma_xx, delimiter=' ')
			np.savetxt(foldername + "/result_sigma_xy.txt", result_sigma_xy, delimiter=' ')
			np.savetxt(foldername + "/result_sigma_yx.txt", result_sigma_yx, delimiter=' ')
			np.savetxt(foldername + "/result_sigma_yy.txt", result_sigma_yy, delimiter=' ') 
			
			np.save(foldername + "/result.npy", result)  
						
			# collect garbage
			del syst
			gc.collect()

			print(f"j_inst = {j_inst + 1} of {n_inst}, j_W = {j_W + 1} of {n_W}, W = {W_curr} ")	
																
	return 


def export_Hall_cond_DC_CI_V3(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, Nx_lead_start, Nx_lead_stop, Ny_lead_start, Ny_lead_stop,  r, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, t_lead_trans = 0.):
	"""Compute 4-terminal conductance matrix for disorder realizations of the DC_CI V2 model and export"""
	# create folder
	foldername = "G_DC_CI_V3_Hall_cond_results/G_DC_CI_V3_Hall_cond_run_" + str(run_nr)

	print(f"Starting G_DC_CI_V3_Hall_cond run {run_nr}")
	
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
	f_parameter.write("x_lead_start " + str(Nx_lead_start) + str("\n"))
	f_parameter.write("x_lead_stop " + str(Nx_lead_stop) + str("\n"))
	f_parameter.write("y_lead_start " + str(Ny_lead_start) + str("\n"))
	f_parameter.write("y_lead_stop " + str(Ny_lead_stop) + str("\n"))
	
	f_parameter.write("r " + str(r) + str("\n"))	
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))	
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))	
	f_parameter.write("gamma_2 " + str(gamma_2) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("t_lead_trans " + str(t_lead_trans) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	
	f_parameter.write("Version V2" + str("\n"))
				
	f_parameter.close()

	result = np.zeros((n_E, 4, 4, n_W))
	result_sigma_xx = np.zeros((n_E, n_W))
	result_sigma_xy = np.zeros((n_E, n_W))
	result_sigma_yx = np.zeros((n_E, n_W))
	result_sigma_yy = np.zeros((n_E, n_W))
								
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):
			# generate System
			syst = make_system_DC_CI_V3_4_terminal(Nx, Ny, Nx_lead_start, Nx_lead_stop, Ny_lead_start, Ny_lead_stop, r, epsilon_1, epsilon_2, gamma, gamma_2, W_curr, t_lead, t_lead_trans)
			syst = syst.finalized()
			
			# calculate conductance matrix for all energy values						
			for j_E in range(n_E):
				energy = E_vals[j_E]
				smatrix = kwant.smatrix(syst, energy)					
				
				cond_mat_curr = smatrix.conductance_matrix()
				
				result[j_E,:,:, j_W] += (1 / n_inst) * cond_mat_curr
								
				sigma_xx, sigma_xy, sigma_yx, sigma_yy = get_conductance(cond_mat_curr)

				result_sigma_xx[j_E, j_W] = (1 / n_inst) * sigma_xx
				result_sigma_xy[j_E, j_W] = (1 / n_inst) * sigma_xy
				result_sigma_yx[j_E, j_W] = (1 / n_inst) * sigma_yx
				result_sigma_yy[j_E, j_W] = (1 / n_inst) * sigma_yy								

			np.savetxt(foldername + "/result_sigma_xx.txt", result_sigma_xx, delimiter=' ')
			np.savetxt(foldername + "/result_sigma_xy.txt", result_sigma_xy, delimiter=' ')
			np.savetxt(foldername + "/result_sigma_yx.txt", result_sigma_yx, delimiter=' ')
			np.savetxt(foldername + "/result_sigma_yy.txt", result_sigma_yy, delimiter=' ') 
			
			np.save(foldername + "/result.npy", result)  
						
			# collect garbage
			del syst
			gc.collect()

			print(f"j_inst = {j_inst + 1} of {n_inst}, j_W = {j_W + 1} of {n_W}, W = {W_curr} ")	
																
	return 

	

def main():

	W_min = 0.
	W_max = 8

	E_min = 0
	E_max = 3.5
	
	n_W = 21
	n_E = 35
	n_inst = 1

	Nx = 150
	Ny = 150
	
	x_lead_start = 50
	x_lead_stop = 100
	y_lead_start = 50
	y_lead_stop = 100
	
		
	r = 1
	epsilon_1 = 0.3
	epsilon_2 = 1
	
	gamma = 2
	
	t_lead = 5
	t_lead_trans = 5
	
	run_nr = 13
		
	#export_Hall_cond_DC_CI_V2(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, x_lead_start, x_lead_stop, y_lead_start, y_lead_stop, r, epsilon_1, epsilon_2, gamma, t_lead, run_nr, t_lead_trans)
	

	W_min = 0.
	W_max = 6

	E_min = -3
	E_max = 3.
	
	n_W = 41
	n_E = 61
	n_inst = 1

	Nx = 150
	Ny = 150
	
	x_lead_start = 50
	x_lead_stop = 100
	y_lead_start = 50
	y_lead_stop = 100
	
		
	r = 1.5
	epsilon_1 = 0.3
	epsilon_2 = 2
	
	gamma = 2
	gamma_2 = 0.3
	
	t_lead = 5
	t_lead_trans = 5
	
	run_nr = 1
		
	export_Hall_cond_DC_CI_V3(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, x_lead_start, x_lead_stop, y_lead_start, y_lead_stop, r, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, t_lead_trans)
	
if __name__ == '__main__':
	main()




	