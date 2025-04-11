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


def make_system_final_flux(Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, disorder, t_lead, direction = "x", transverse_disp = False):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport. Set by transverse_disp whether there should be
	a transverse dispersion in the leads or not. """
	# values for lattice constant and number of orbitals
	a = 1
	
	Nx = int(Nx)
	Ny = int(Ny)
	
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
	
	# Flux hopping around boundary defined as function to make it variable
	def hop_dx_phi_PBC(site1, site2, phi):
		res = np.exp(1j * phi) * hop_mat_dx
		return res 
		
	def hop_dy_phi_PBC(site1, site2, phi):
		res = np.exp(1j * phi) * hop_mat_dy
		return res  
	
	def hop_dx_dy_phi_PBC(site1, site2, phi):
		res = np.exp(1j * phi) * hop_mat_dx_dy
		return res  
		
	def hop_dx_mdy_phi_PBC_x(site1, site2, phi):
		# This is for generalized BC along x direction, i.e. transport along y.
		res = np.exp(1j * phi) * hop_mat_dx_mdy
		return res  
	
	def hop_dx_mdy_phi_PBC_y(site1, site2, phi):
		# This is for generalized BC along y direction, i.e. transport along x. There, 
		# we hop in negative y direction and should thus pick up the negative phase.
		res = np.exp(-1j * phi) * hop_mat_dx_mdy
		return res  
	
	if direction == "x":
		# Set boundary terms along y-direction
		syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = hop_dy_phi_PBC
		
		# Set boundary terms along (x+y)-direction
		syst[kwant.builder.HoppingKind((1, -(Ny-1)), lat, lat)] = hop_dx_dy_phi_PBC
		
		# Set boundary terms along (x-y)-direction
		syst[kwant.builder.HoppingKind((1, (Ny-1)), lat, lat)] = hop_dx_mdy_phi_PBC_y
		
		# construct left lead
		lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
			
		#initialize with on-site terms to avoid error 
		lead[(lat(0,jy) for jy in range(0, Ny))] = 0 * sigma_0

		# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
		lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead * sigma_0
		if transverse_disp:
			lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead * sigma_0
			lead[kwant.builder.HoppingKind((0, -(Ny - 1)), lat, lat)] = t_lead * sigma_0
							
		# attach left lead 
		syst.attach_lead(lead)

		# reverse left lead and attach it on the right
		syst.attach_lead(lead.reversed())

	elif direction == "y":
		# Set boundary terms along x-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = hop_dx_phi_PBC
		
		# Set boundary terms along (x+y)-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), 1), lat, lat)] = hop_dx_dy_phi_PBC
		
		# Set boundary terms along (x-y)-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), -1), lat, lat)] = hop_dx_mdy_phi_PBC_x
		
		# construct bottom lead
		lead = kwant.Builder(kwant.TranslationalSymmetry((0, -a)))
			
		#initialize with on-site terms to avoid error 
		lead[(lat(jx,0) for jx in range(0, Nx))] = 0 * sigma_0

		# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
		lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead * sigma_0
		if transverse_disp:
			lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead * sigma_0
			lead[kwant.builder.HoppingKind((0, -(Ny - 1)), lat, lat)] = t_lead * sigma_0
							
		# attach left lead 
		syst.attach_lead(lead)

		# reverse left lead and attach it on the right
		syst.attach_lead(lead.reversed())
				
	#kwant.plotter.plot(syst)
											 
	return syst.finalized()


def winding_number(syst, E, n_phi):
	"""Calculate winding number for syst at energy E."""
	phi_vals = np.linspace(0, 2 * np.pi, n_phi)
	phase_vals = np.zeros(n_phi)
	
	# dictionary with all parameters that are needed for the functions appearing in syst, here only one is needed
	p= {"phi" : phi_vals[0]}
	
	# calculate first datapoint
	r_matrix_0 = kwant.smatrix(syst, E, params=p).submatrix(0, 0)
	phase_r_matrix_0 = np.angle(np.linalg.det(r_matrix_0))
	phase_vals[0] = phase_r_matrix_0

	sum_phases =  0
	min_EV = np.zeros(n_phi)	
	
	EVs_0 = np.linalg.eigvals(r_matrix_0)
	min_EV[0] = np.min(np.abs(EVs_0))
	
	for j in range(1, n_phi):	
		# Update value in dictionary
		p["phi"] = phi_vals[j]
				
		# Get reflection matrix 
		r_matrix = kwant.smatrix(syst, E, params=p).submatrix(0, 0)
		
		# get complex angle of determinant 
		phase_r_matrix = np.angle(np.linalg.det(r_matrix))
		phase_vals[j] = phase_r_matrix
		
		d_phase = phase_vals[j] - phase_vals[j-1]
		
		# Calculate integration step, take care that angle may jump
		if d_phase > 1.5:
			d_phase = phase_vals[j] - phase_vals[j-1] - 2 * np.pi
		if d_phase < -1.5:
			d_phase = phase_vals[j] - phase_vals[j-1] + 2 * np.pi
		
		sum_phases +=d_phase
		
		# get absolute value of smallest eigenvalue
		EVs = np.linalg.eigvals(r_matrix)
		min_EV[j] = np.min(np.abs(EVs))
	
	return sum_phases / (2 * np.pi), np.min(min_EV)
		

def export_winding_number_final(W, E_min, E_max, n_E, n_inst, n_phi, Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, direction = "x", transverse_disp =  False):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "Winding_number_final/Winding_number_final_run_" + str(run_nr)
	print(f"Starting Wnum_final run {run_nr}")
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	# generate and save va
	E_vals = np.linspace(E_min, E_max, n_E)
	
	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')  
	
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("W " + str(W) + str("\n"))
	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	
	f_parameter.write("n_E " + str(n_E) + str("\n"))
	f_parameter.write("n_inst " + str(n_inst) + str("\n"))
	f_parameter.write("n_phi " + str(n_phi) + str("\n"))
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))	
	
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))	
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))
	f_parameter.write("gamma_2 " + str(gamma_2) + str("\n"))
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	f_parameter.write("direction " + str(direction) + str("\n"))	
	f_parameter.write("transverse_disp " + str(transverse_disp) + str("\n"))	
				
	f_parameter.close()
					
	result = np.zeros((n_E, n_inst))
	min_EV_r_matrix = np.zeros((n_E, n_inst))	
		
	for j_inst in range(0, n_inst):
		# generate system
		disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
		syst = make_system_final_flux(Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, disorder, t_lead, direction, transverse_disp)
				
		# calculate winidng number for all energy values	
		for j_E in range(n_E):
			energy = E_vals[j_E]
			w_num, min_EV = winding_number(syst, energy, n_phi)			
			result[j_E, j_inst] = w_num
			min_EV_r_matrix[j_E, j_inst] = min_EV
			
			np.savetxt(foldername + "/result.txt", result, delimiter=' ')  
			np.savetxt(foldername + "/min_EV_r_matrix.txt", min_EV_r_matrix, delimiter=' ')  
			
			sys.stdout.write(f"j_inst = {j_inst + 1} of {n_inst}, j_E = {j_E + 1} of {n_E} \n")
			sys.stdout.flush()
					
		# collect garbage
		del syst
		gc.collect()
													
	return 


def export_winding_number_final_single_vals(W_vals, E_vals, n_inst, n_phi, Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, direction = "x", transverse_disp =  False):
	"""Compute conductance for disorder realizations of the double cone CI model and export. W_vals and E_vals should be
	1d arrays of similar size."""
	# create folder
	foldername = "Winding_number_final_single_vals/Winding_number_final_single_vals_run_" + str(run_nr)
	print(f"Starting Wnum_final_single_vals run {run_nr}")
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')  
	np.savetxt(foldername + "/W_vals.txt", W_vals, delimiter=' ')  
	
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("n_inst " + str(n_inst) + str("\n"))
	f_parameter.write("n_phi " + str(n_phi) + str("\n"))
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))	
	
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))	
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))
	f_parameter.write("gamma_2 " + str(gamma_2) + str("\n"))
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	f_parameter.write("direction " + str(direction) + str("\n"))	
	f_parameter.write("transverse_disp " + str(transverse_disp) + str("\n"))	
				
	f_parameter.close()
					
	n_vals = len(E_vals)
	result = np.zeros((n_vals, n_inst))
	result_mean = np.zeros((n_vals, 2))
	
	min_EV_r_matrix = np.zeros((n_vals, n_inst))	
	
	# loop over all parameter values
	for j in range(n_vals):
		energy = E_vals[j]
		W = W_vals[j]

		# calculate winding number for random disorder instances		
		for j_inst in range(0, n_inst):		
			# generate system
			disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
			syst = syst = make_system_final_flux(Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, disorder, t_lead, direction, transverse_disp)
		
			w_num, min_EV = winding_number(syst, energy, n_phi)			
			result[j, j_inst] = w_num
			min_EV_r_matrix[j, j_inst] = min_EV
			
			np.savetxt(foldername + "/result.txt", result, delimiter=' ')  
			np.savetxt(foldername + "/min_EV_r_matrix.txt", min_EV_r_matrix, delimiter=' ')  
			
			sys.stdout.write(f"j_inst = {j_inst + 1} of {n_inst}, j_E = {j + 1} of {n_vals} \n")
			sys.stdout.flush()
					
		# collect garbage
		del syst
		gc.collect()
	
	# calculate mean and std
	result_mean[:,0] = np.mean(result, axis = 1)
	result_mean[:,1] = np.std(result, axis = 1)
	
	np.savetxt(foldername + "/result_mean.txt", result_mean, delimiter=' ')  
													
	return 

	

def main():
	W = 2

	E_min = -2
	E_max = 2
	
	n_E = 21
	n_inst = 3

	Nx = 50
	Ny = 50
		
	epsilon_1 = 0.5
	epsilon_2 = 5
	
	gamma = 4
	gamma_2 = 1
	
	t_lead = 10
	
	n_phi = 20	
	
	run_nr = 1
	
	direction = "x"
	
	transverse_disp = True
	
	#export_winding_number_final(W, E_min, E_max, n_E, n_inst, n_phi, Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, direction, transverse_disp)
	
	E_vals = [-2., 0, 2.]	
	W_vals = [4, 4, 4]	
	
	n_inst = 5

	Nx = 200
	Ny = 50
		
	gamma = 4
	gamma_2 = 1
	
	t_lead = 10
	
	n_phi = 20	
	
	run_nr = 1
	
	direction = "x"
	
	transverse_disp = True
	
	export_winding_number_final_single_vals(W_vals, E_vals, n_inst, n_phi, Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, t_lead, run_nr, direction, transverse_disp)
	


	
if __name__ == '__main__':
	main()




	