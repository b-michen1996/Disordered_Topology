import os
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import kwant
import numpy as np
import gc

sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])
sigma_0 = np.array([[1, 0],[0, 1]])	


def make_system_final(Nx, Ny, N_lead_start, N_lead_stop, epsilon_1, epsilon_2, gamma, gamma_2, W, t_lead, PBC = 1., direction = "x", t_lead_trans = 0.):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
	# values for lattice constant and number of orbitals
	a = 1	
	
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	
	lat = kwant.lattice.square(a, norbs = 2)
	syst = kwant.Builder()
	
	mat_os = 0
	hop_mat_dx = gamma/(2j) * sigma_x 
	hop_mat_dy = ((epsilon_1 + epsilon_2/2)/(2j)) * sigma_y - ((epsilon_1 + epsilon_2/2) / 2) * sigma_z + (gamma_2 /2) * sigma_x 
	hop_mat_dx_dy = -(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z
	hop_mat_dx_mdy = +(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z

	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy					
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + mat_os
								
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_mat_dx
	
	# Hopping in y-direction
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_mat_dy

	# Hopping in (x+y)-direction
	syst[kwant.builder.HoppingKind((1, 1), lat, lat)] = hop_mat_dx_dy
	
	# Hopping in (x-y)-direction
	syst[kwant.builder.HoppingKind((1, -1), lat, lat)] = hop_mat_dx_mdy
	
	if direction == "x":
		# Set boundary terms along y-direction
		syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = PBC * hop_mat_dy
		
		# Set boundary terms along (x+y)-direction
		syst[kwant.builder.HoppingKind((1, -(Ny-1)), lat, lat)] = PBC * hop_mat_dx_dy
		
		# Set boundary terms along (x-y)-direction
		syst[kwant.builder.HoppingKind((1, (Ny-1)), lat, lat)] = PBC * hop_mat_dx_mdy
		
		# construct left lead
		lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
			
		#initialize with on-site terms to avoid error 
		lead[(lat(0,jy) for jy in range(N_lead_start, N_lead_stop))] = 0 * sigma_0

		# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
		lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead * sigma_0
		lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead_trans * sigma_0
		if (N_lead_start == 0 and N_lead_stop == Ny):
			lead[kwant.builder.HoppingKind((0, Ny - 1), lat, lat)] = PBC * t_lead_trans * sigma_0
		
							
		# attach left lead 
		syst.attach_lead(lead)

		# reverse left lead and attach it on the right
		syst.attach_lead(lead.reversed())

	elif direction == "y":
		# Set boundary terms along x-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = PBC * hop_mat_dx
		
		# Set boundary terms along (x+y)-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), 1), lat, lat)] = PBC * hop_mat_dx_dy
		
		# Set boundary terms along (x-y)-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), -1), lat, lat)] = PBC * hop_mat_dx_mdy
		
		# construct bottom lead
		lead = kwant.Builder(kwant.TranslationalSymmetry((0, -a)))
			
		#initialize with on-site terms to avoid error 
		lead[(lat(jx,0) for jx in range(N_lead_start, N_lead_stop))] = 0 * sigma_0

		# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
		lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead * sigma_0
		lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead_trans * sigma_0
		if (N_lead_start == 0 and N_lead_stop == Nx):
			lead[kwant.builder.HoppingKind((Nx - 1, 0), lat, lat)] = PBC * t_lead_trans * sigma_0
							
		# attach left lead 
		syst.attach_lead(lead)

		# reverse left lead and attach it on the right
		syst.attach_lead(lead.reversed())
				
	#kwant.plotter.plot(syst)
											 
	return syst.finalized()



def export_conductance_final_model(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, N_lead_start, N_lead_stop, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, PBC = 1., direction = "x", t_lead_trans = 0.):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "G_final_results/G_final_run_" + str(run_nr)

	print(f"Starting G_final_ run {run_nr}")
	
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
	f_parameter.write("N_lead_start " + str(N_lead_start) + str("\n"))	
	f_parameter.write("N_lead_stop " + str(N_lead_stop) + str("\n"))	
	
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))	
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))	
	f_parameter.write("gamma_2 " + str(gamma_2) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("t_lead_trans " + str(t_lead_trans) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	
	f_parameter.write("PBC " + str(PBC) + str("\n"))
	f_parameter.write("direction " + str(direction) + str("\n"))
			
	f_parameter.close()
					
	result = np.zeros((n_E, n_W))
	result = np.zeros((n_E, n_W))
		
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):
			# generate System
			syst = make_system_final(Nx, Ny, N_lead_start, N_lead_stop, epsilon_1, epsilon_2, gamma, gamma_2, W_curr, t_lead, PBC, direction, t_lead_trans)			
			
			# calculate transmission for all energy values
			data_j_inst = np.zeros(n_E)			
			for j_E in range(n_E):
				energy = E_vals[j_E]
				smatrix = kwant.smatrix(syst, energy)					
				data_j_inst[j_E] = smatrix.transmission(1, 0)
				print(f"j_inst = {j_inst + 1} of {n_inst}, j_W = {j_W + 1} of {n_W}, W = {W_curr}, j_E = {j_E + 1} of {n_E}, W = {W_curr}, E = {energy} ")	
			
			# collect garbage
			del syst
			gc.collect()
												
			result[:, j_W] += (1 / n_inst) * data_j_inst 
			
			np.savetxt(foldername + "/result.txt", result, delimiter=' ')  		
								
	return 


def main():
	
	W_min = 0
	W_max = 6
	
	E_min = -3.
	E_max = 3.
	
	n_W = 41
	n_E = 51
	n_inst = 1
		
	epsilon_1 = -0.3
	epsilon_2 = 2

	gamma = 2.
	gamma_2 = 0.8
	
	t_lead = 5
	t_lead_trans = 5
	
	PBC = 0.	
	  
	Nx = 200
	Ny = 100
	
	N_lead_start = 25
	N_lead_stop = 75
	
	direction = "x"
	
	run_nr = 9

	export_conductance_final_model(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, N_lead_start, N_lead_stop, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, PBC, direction, t_lead_trans)


if __name__ == '__main__':
	main()




	
