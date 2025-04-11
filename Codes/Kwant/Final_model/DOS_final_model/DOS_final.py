import os
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import kwant
import numpy as np
import gc
import random


sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])
sigma_0 = np.array([[1, 0],[0, 1]])

sigma_x_3b = np.array([[0,1,0], [1,0,0], [0,0,0]])
sigma_y_3b = np.array([[0,-1j,0], [1j,0,0], [0,0,0]])
sigma_z_3b = np.array([[1,0,0], [0,-1,0], [0,0,0]])
M_coupling = np.array([[0,0,1], [0,0,0], [1,0,0]])

def string_contents(array):
#= Return string_contents of contents of array with spaces between them =#
	result = ""
	for x in array:
		result = result + str(x) + " "	

	return result  


def make_system_final_V2(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, PBC = 1.):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
	# values for lattice constant and number of orbitals
	a = 1
	
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	rand_flux_vals = (U/2) * np.pi * np.ones((Nx, Ny)) - U * np.pi * np.random.random_sample((Nx, Ny))
		
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
	
	# Set boundary terms along x-direction
	syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = PBC * hop_mat_dx

	# Set boundary terms along y-direction
	syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = PBC * hop_mat_dy
	
	# Set boundary terms along (x+y)-direction
	syst[kwant.builder.HoppingKind((-(Nx-1), 1), lat, lat)] = PBC * hop_mat_dx_dy
	syst[kwant.builder.HoppingKind((1, -(Ny-1)), lat, lat)] = PBC * hop_mat_dx_dy
	syst[kwant.builder.HoppingKind((-(Nx-1), -(Ny-1)), lat, lat)] = PBC * hop_mat_dx_dy
	
	# Set boundary terms along (x-y)-direction
	syst[kwant.builder.HoppingKind((-(Nx-1), -1), lat, lat)] = PBC * hop_mat_dx_mdy
	syst[kwant.builder.HoppingKind((1, (Ny-1)), lat, lat)] = PBC * hop_mat_dx_mdy
	syst[kwant.builder.HoppingKind((-(Nx-1), (Ny-1)), lat, lat)] = PBC * hop_mat_dx_mdy
	
	syst = syst.finalized()
																 
	return syst


def make_system_final_V2_rand_flux(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, U, PBC = 1.):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
	# values for lattice constant and number of orbitals
	a = 1
	
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	rand_flux_vals = (U/2) * np.pi * np.ones((Nx, Ny)) - U * np.pi * np.random.random_sample((Nx, Ny))
		
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
	def hop_dx(site_1, site_2):
		jx = site_1.tag[0]
		jy = site_1.tag[1]
		phase = np.exp(1j * (rand_flux_vals[jx,jy])) 
		return phase * hop_mat_dx
	
	def hop_dx_PBC(site_1, site_2):
		jx = site_1.tag[0]
		jy = site_1.tag[1]
		phase = np.exp(1j * (rand_flux_vals[jx,jy])) 
		return PBC * phase * hop_mat_dx
	
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_dx
	
	# Hopping in y-direction
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_mat_dy	
						
	# Hopping in (x+y)-direction
	def hop_dx_dy(site_1, site_2):
		jx = site_1.tag[0]
		jy = site_1.tag[1]
		phase = np.exp(1j * (rand_flux_vals[jx,jy])) 
		return phase * hop_mat_dx_dy
	
	def hop_dx_dy_PBC(site_1, site_2):
		jx = site_1.tag[0]
		jy = site_1.tag[1]
		phase = np.exp(1j * (rand_flux_vals[jx,jy])) 
		return PBC * phase * hop_mat_dx_dy
	
	syst[kwant.builder.HoppingKind((1, 1), lat, lat)] = hop_dx_dy
	
	# Hopping in (x-y)-direction
	def hop_dx_mdy(site_1, site_2):
		jx = site_1.tag[0]
		jy = site_1.tag[1]
		phase = np.exp(1j * (rand_flux_vals[jx,jy])) 
		return phase * hop_mat_dx_mdy
	
	def hop_dx_mdy_PBC(site_1, site_2):
		jx = site_1.tag[0]
		jy = site_1.tag[1]
		phase = np.exp(1j * (rand_flux_vals[jx,jy])) 
		return PBC * phase * hop_mat_dx_mdy
	
	# Hopping in (x-y)-direction
	syst[kwant.builder.HoppingKind((1, -1), lat, lat)] = hop_dx_mdy
	
	# Set boundary terms along x-direction
	syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = hop_dx_PBC

	# Set boundary terms along y-direction
	syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = PBC * hop_mat_dy
	
	# Set boundary terms along (x+y)-direction
	syst[kwant.builder.HoppingKind((-(Nx-1), 1), lat, lat)] = hop_dx_dy_PBC
	syst[kwant.builder.HoppingKind((1, -(Ny-1)), lat, lat)] = hop_dx_dy_PBC
	syst[kwant.builder.HoppingKind((-(Nx-1), -(Ny-1)), lat, lat)] = hop_dx_dy_PBC
	
	# Set boundary terms along (x-y)-direction
	syst[kwant.builder.HoppingKind((-(Nx-1), -1), lat, lat)] = hop_dx_mdy_PBC
	syst[kwant.builder.HoppingKind((1, (Ny-1)), lat, lat)] = hop_dx_mdy_PBC
	syst[kwant.builder.HoppingKind((-(Nx-1), (Ny-1)), lat, lat)] = hop_dx_mdy_PBC
	
	syst = syst.finalized()
																 
	return syst



def DOS_final(E_vals, Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2,  W, U, PBC = 1., n_moments_KPM = 150, num_vec = None):
	"""Calculate typical and average DOS. The parameter num_vec tells us over how many vectors the sampling is
	carried out, num_vec = None means over all sites of the system."""
	# make system
	syst = make_system_final_V2_rand_flux(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, U, PBC)
	# vector factory for local vectors
	if num_vec == None:
		# vector factory runs over all sites of the system
		vector_factory = kwant.kpm.LocalVectors(syst)
		print(f'This factory has {len(vector_factory)} vectors.')
	else:
		# vector factory contains num_vec vectors at random sites of the system, determined by where
		all_sites = range(0, Nx * Ny)
		sample_sites= random.sample(all_sites, k = num_vec)
		
		vector_factory = kwant.kpm.LocalVectors(syst, where = sample_sites)
			
	# get local DOS. Setting mean = False returns a vector of densities for each
	# of the sites provided by the vector factory which corresponds to the local
	# DOS at this size withouth the normalizing prefactor 1 / (2 Nx Ny).
	# The bulk DOS is thus obtained from the arithmetic average over all these 
	# entries. It should suffice to average only over a couple, however for a 
	# localized systen the local DOS will fluctuate strongly, so one should
	# ideally sample over all sites of the system, i.e. take num_vec = 2 Nx Ny.
	local_dos = kwant.kpm.SpectralDensity(syst, num_vectors= num_vec, num_moments= n_moments_KPM,
                                      vector_factory=vector_factory,
                                      mean=False,
									accumulate_vectors = False)
	
	# array for local dos at each energy. First index is energy, second is the
	# index of the local vector used for sampling. We put the nan values that arise
	# when the energy is outside the estimated range of the matrix to 
	# zero to avoid errors. These values will produce a runtime warning, but we can
	# ignore that.
	rho_loc = local_dos(energy = E_vals)
	rho_loc[np.isnan(rho_loc)] = 1E-10
	
	# calculate average / bulk DOS as arithmetic mean of all the local DOS values. 		
	rho_av = np.sum(rho_loc, axis = 1)
	
	# calculate typical DOS as average of the logarithm of the local DOS
	rho_typ = np.sum(np.log(rho_loc), axis = 1)
	
	rho_IPR = np.sum(np.square(rho_loc), axis = 1)
	
	if num_vec == None:
		rho_av = rho_av / (2 * Nx * Ny)
		rho_typ = rho_typ / (2 * Nx * Ny)
		rho_IPR = rho_IPR / (2 * Nx * Ny)
	else: 
		rho_av = rho_av / (num_vec)
		rho_typ = rho_typ / (num_vec)
		rho_IPR = rho_IPR / (num_vec)
		
	# collect garbage
	del syst
	del local_dos
	gc.collect()
	
	return rho_av, np.exp(rho_typ), rho_IPR


def export_DOS_final_V2(W, U, E_min, E_max, n_E, n_inst, Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, run_nr, PBC = 1., n_vec_KPM = 10, n_moments_KPM = 150):
	"""Compute DOS for disorder realizations of the DC_CI_V3 model and export."""
	# create folder
	foldername = "DOS_final_V2_results/DOS_final_V2_run_" + str(run_nr)

	print(f"Starting DOS_final_V2 run {run_nr}")
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	# generate and save values
	E_vals = np.linspace(E_min, E_max, n_E)
	
	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')  
	
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("W " + str(W) + str("\n"))
	f_parameter.write("U " + str(U) + str("\n"))
	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	
	f_parameter.write("n_inst " + str(n_inst) + str("\n"))
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))		
	
	f_parameter.write("r " + str(r) + str("\n"))	
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))	
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))	
	f_parameter.write("gamma_2 " + str(gamma_2) + str("\n"))
	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	f_parameter.write("n_vec_KPM " + str(n_vec_KPM) + str("\n"))	
	f_parameter.write("n_moments_KPM " + str(n_moments_KPM) + str("\n"))
	
	f_parameter.write("PBC " + str(PBC) + str("\n"))
				
	f_parameter.close()
					
	rho_av = np.zeros(n_E)
	rho_typ = np.zeros(n_E)	
	rho_IPR = np.zeros(n_E)	

	for j_inst in range(0, n_inst):
		# generate System
		rho_av_curr, rho_typ_curr, rho_IPR_curr = DOS_final(E_vals, Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, U, PBC, n_moments_KPM, n_vec_KPM)
		
		rho_av += (1 / n_inst) * rho_av_curr
		rho_typ += (1 / n_inst) * rho_typ_curr
		rho_IPR += (1 / n_inst) * rho_IPR_curr
								
		np.savetxt(foldername + "/rho_av.txt", rho_av, delimiter=' ')  		
		np.savetxt(foldername + "/rho_typ.txt", rho_typ, delimiter=' ')  		
		np.savetxt(foldername + "/rho_IPR.txt", rho_IPR, delimiter=' ')  		
		print(f"j_inst = {j_inst + 1} of {n_inst}")
												
	return 



def main():	
	E_min = -3
	E_max = 3
	
	n_E = 1000
	
	W = 0
	U = 2
		
	Nx = 100
	Ny = 100
			
	n_moments_KPM = 100
	n_vec_KPM = 200
	
	r = 0.
	epsilon_1 = -0.3
	epsilon_2 = 2

	gamma = 2.
	gamma_2 = 0.2
	
	PBC = 1.
	
	n_inst = 1

	run_nr = 2
	
	export_DOS_final_V2(W, U, E_min, E_max, n_E, n_inst, Nx, Ny,  r, epsilon_1, epsilon_2, gamma, gamma_2, run_nr, PBC, n_vec_KPM, n_moments_KPM)
	
if __name__ == '__main__':
	main()




	