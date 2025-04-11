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


def string_contents(array):
#= Return string_contents of contents of array with spaces between them =#
	result = ""
	for x in array:
		result = result + str(x) + " "
	

	return result  


def make_system_final(Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, W, t_lead, PBC = 1., direction = "x", t_lead_trans = 0.):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
	# values for lattice constant and number of orbitals
	a = 1	
	
	Nx = int(Nx)
	Ny = int(Ny)
	
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	
	lat = kwant.lattice.square(a, norbs = 2)
	syst = kwant.Builder()
	
	mat_os = 0
	hop_mat_dx = gamma/(2j) * sigma_x
	hop_mat_dy = ((epsilon_1 + epsilon_2/2)/(2j)) * sigma_y - ((epsilon_1 + gamma_2 + epsilon_2/2) / 2) * sigma_z
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
		lead[(lat(0,jy) for jy in range(0, Ny))] = 0 * sigma_0

		# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
		lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead * sigma_0
		lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead_trans * sigma_0
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
		lead[(lat(jx,0) for jx in range(0, Nx))] = 0 * sigma_0

		# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
		lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead * sigma_0
		lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead_trans * sigma_0
		lead[kwant.builder.HoppingKind((Nx - 1, 0), lat, lat)] = PBC * t_lead_trans * sigma_0
							
		# attach left lead 
		syst.attach_lead(lead)

		# reverse left lead and attach it on the right
		syst.attach_lead(lead.reversed())
				
	#kwant.plotter.plot(syst)
											 
	return syst.finalized()


def export_size_scaling_final(W_vals, E, n_inst, Nx_vals, Ny_vals, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, PBC = 1., direction = "x", t_lead_trans = 0.):
	"""Compute conductance for disorder realizations of DC_CI_V2 model and export. W_vals, Nx_vals, and Ny_vals should be passed
	as 1d arrays of similar size."""
	# create folder
	foldername = "size_scaling_final_results/size_scaling_final_run_" + str(run_nr)

	print(f"Starting size_scaling_final_ run {run_nr}")
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("W " + string_contents(W_vals) + str("\n"))
	f_parameter.write("E " + str(E) + str("\n"))
	
	f_parameter.write("n_inst " + str(n_inst) + str("\n"))
	
	f_parameter.write("Nx " + string_contents(Nx_vals) + str("\n"))	
	f_parameter.write("Ny " + string_contents(Ny_vals) + str("\n"))		
	
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))	
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))	
	f_parameter.write("gamma_2 " + str(gamma) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("t_lead_trans " + str(t_lead_trans) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	
	f_parameter.write("PBC " + str(PBC) + str("\n"))
	f_parameter.write("direction " + str(direction) + str("\n"))
			
	f_parameter.close()
					
	n_vals = len(W_vals)
	result = np.zeros(n_vals)
		
	for j_val in range(0, n_vals):
		W_curr = W_vals[j_val]		
		Nx_curr = Nx_vals[j_val]		
		Ny_curr = Ny_vals[j_val]	
			
		
		for j_inst in range(0, n_inst):
			# generate System
			syst = make_system_final(Nx_curr, Ny_curr, epsilon_1, epsilon_2, gamma, gamma_2, W_curr, t_lead, PBC, direction, t_lead_trans)						
			
			# calculate transmission
			smatrix = kwant.smatrix(syst, E)					
			result[j_val] += (1 / n_inst) * smatrix.transmission(1, 0)
									
			np.savetxt(foldername + "/result.txt", result, delimiter=' ')  		
			print(f"j_inst = {j_inst + 1} of {n_inst}, j_val = {j_val + 1} of {n_vals}, W = {W_curr}, Nx = {Nx_curr}, Ny = {Ny_curr}")	
			
			# collect garbage
			del syst
			gc.collect()						
								
	return 


def main():
	
	n_inst = 1  
		
	epsilon_1 = 0.5
	epsilon_2 = 5
	
	gamma = 4.
	gamma_2 = 1
		
	t_lead = 10	
	t_lead_trans = 10
	
	PBC = 0.	
		
	direction = "x"
			
	n_vals = 26

	W_vals = 6. * np.ones(n_vals)
	Ny_vals = 50 * np.ones(n_vals) + 10 * np.array(range(0, n_vals))
	Nx_vals =  3 * Ny_vals
	
	E = 2.
	
	run_nr = 4
	
	export_size_scaling_final(W_vals, E, n_inst, Nx_vals, Ny_vals, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, PBC, direction, t_lead_trans)
	
	
	
if __name__ == '__main__':
	main()




	
