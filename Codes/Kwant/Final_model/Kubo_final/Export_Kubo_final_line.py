import os
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import kwant
import numpy as np
import random
import gc
from matplotlib import pyplot as plt

from kpm_tools.bloch import wrap_velocity

# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})


sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])
sigma_0 = np.array([[1, 0],[0, 1]])	


def make_system_final(Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, W):
	"""Creates system with the given parameters and then adds random impurities."""
		
	# define custom square lattice with two sublattices, because the custom velocity operator does not handle
	# orbitals and the matrix-valued hopping elements
	prim_vecs = [(1,0), (0,1)]	
	basis = [(0,0), (0,0.5)]	
	name=["a", "b"]
	lat = kwant.lattice.general(prim_vecs, basis, name, norbs = 1)
	
	lat_a, lat_b = lat.sublattices
		
	syst = kwant.builder.Builder(kwant.lattice.TranslationalSymmetry([Nx, 0], [0, Ny]))	
	
	# Onsite terms and impurities
	disorder = W * np.ones((Nx, Ny, 2)) - 2 * W * np.random.random_sample((Nx, Ny, 2))
	
	for jx in range(Nx):
			for jy in range(Ny):					
					syst[lat_a(jx,jy)] = disorder[jx,jy,0] 
					syst[lat_b(jx,jy)] = disorder[jx,jy,1] 
	
	# matrices that contain hoppings	
	mat_os = 0
	hop_mat_dx = gamma/(2j) * sigma_x
	hop_mat_dy = ((epsilon_1 + epsilon_2/2)/(2j)) * sigma_y - ((epsilon_1 + epsilon_2/2) / 2) * sigma_z + (gamma_2 /2) * sigma_x 
	hop_mat_dx_dy = -(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z
	hop_mat_dx_mdy = +(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z
	
	lat_list = (lat_a, lat_b)
	for l_1 in range(2):
		for l_2 in range(2):
			lat_l1 = lat_list[l_1]
			lat_l2 = lat_list[l_2]
			
			# Hopping in x-direction		
			syst[kwant.builder.HoppingKind((1, 0), lat_l1, lat_l2)] = hop_mat_dx[l_1, l_2]
			
			# Hopping in y-direction
			syst[kwant.builder.HoppingKind((0, 1), lat_l1, lat_l2)] = hop_mat_dy[l_1, l_2]
			
			# Hopping in (x+y)-direction
			syst[kwant.builder.HoppingKind((1, 1), lat_l1, lat_l2)] = hop_mat_dx_dy[l_1, l_2]

			# Hopping in (x-y)-direction
			syst[kwant.builder.HoppingKind((1, -1), lat_l1, lat_l2)] = hop_mat_dx_mdy[l_1, l_2]			
	
	#kwant.plotter.plot(syst)
											 
	return syst, lat


def get_cond_final_wraparound(Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, W, E_vals, n_vec_KPM = 10, n_moments_KPM = 100, sigma_entries = [1], local = False):
	"""Get linear response conductivity tensor for all energies from KPM method."""
	system, lat = make_system_final(Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, W)
	
	# get velocity operators 
	velocity_builder = wrap_velocity(system)
	vx = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[1, 0]), sparse=True)
	vy = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[0, 1]), sparse=True)
	
	# wrap system and finalize 
	system = kwant.wraparound.wraparound(system).finalized()
	area_normalization = Nx * Ny
	
	vector_factory = kwant.kpm.RandomVectors(system)
	if local:
		# vector factory contains num_vec vectors at random sites of the system, determined by where
		all_sites = range(0, Nx * Ny)
		sample_sites= random.sample(all_sites, k = n_vec_KPM)
		
		#vector_factory = kwant.kpm.LocalVectors(system, where = sample_sites)
		vector_factory = kwant.kpm.LocalVectors(system)
		
		area_normalization = 0.5
	
	cond_elements = [(vx,vx), (vx,vy), (vy,vx),(vy,vy)]
	result = list()
	for j in sigma_entries:
		v1_j = cond_elements[j][0]		
		v2_j = cond_elements[j][1]		
		# calculate conductivity (params=dict(k_x=0, k_y=0) is needed to avoid some cryptic error?)
		conductivity = kwant.kpm.conductivity(system, params=dict(k_x=0, k_y=0), 
											alpha=v1_j,
											beta=v2_j,
											vector_factory = vector_factory,
											num_vectors=n_vec_KPM,
											num_moments = n_moments_KPM,
											accumulate_vectors = False)
		
		result.append(np.array([conductivity(energy, temperature=0.).real / area_normalization for energy in E_vals]))
		# collect garbage 
		del conductivity
		gc.collect()	
		
	# collect garbage
	del system, vx, vy, cond_elements	
	gc.collect()
	
	return result


def export_conductance_final_line(W, E_min, E_max, n_E, n_inst, Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, run_nr, n_vec_KPM = 10, n_moments_KPM = 100, sigma_entries = [0,1,2,3], local = False):
	"""Calculate Conductance from Kubo Formula"""
	# create folder
	foldername = "Kubo_final_line_results/Kubo_final_line_run_" + str(run_nr)

	print(f"Starting Kubo_final_line run {run_nr}")
	
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
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))	
	
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))	
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))	
	f_parameter.write("gamma_2 " + str(gamma_2) + str("\n"))	
	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	f_parameter.write("n_vec_KPM " + str(n_vec_KPM) + str("\n"))	
	f_parameter.write("n_moments_KPM " + str(n_moments_KPM) + str("\n"))
	f_parameter.write("sigma_entries " + str(sigma_entries) + str("\n"))
	f_parameter.write("local " + str(local) + str("\n"))	
			
	f_parameter.close()
					
	result_sigma_xx = np.zeros((n_E, n_inst))
	result_sigma_xy = np.zeros((n_E, n_inst))
	result_sigma_yx = np.zeros((n_E, n_inst))
	result_sigma_yy = np.zeros((n_E, n_inst))
	
	results = list((result_sigma_xx, result_sigma_xy, result_sigma_yx, result_sigma_yy))
	results_filenames = ("/result_sigma_xx.txt", "/result_sigma_xy.txt", "/result_sigma_yx.txt", "/result_sigma_yy.txt")
			
	for j in range(4):
		filename = results_filenames[j]
		np.savetxt(foldername + filename, results[j], delimiter=' ')  			
		
	for j_inst in range(0, n_inst):			
		# get conductance			
		cond = get_cond_final_wraparound(Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, W, E_vals, n_vec_KPM, n_moments_KPM, sigma_entries, local)					

		for j_entry in range(len(sigma_entries)):								
			cond_element = cond[j_entry]										
			entry_index = sigma_entries[j_entry]				
			results[entry_index][:,j_inst] = cond_element
			
			filename = results_filenames[entry_index]
			np.savetxt(foldername + filename, results[entry_index], delimiter=' ')  	

		del cond
		gc.collect()	
			
		print(f"j_inst = {j_inst + 1} of {n_inst}")	
											
	return 


def main():	
	W = 1.2

	E_min = -3
	E_max = 3
	
	n_E = 1000
	n_inst = 10
	
	epsilon_1 = 0.3
	epsilon_2 = 2

	gamma = 2
	gamma_2 = 0.2
	
	Nx = 50
	Ny = 50
	
	n_vec_KPM = 10
	n_moments_KPM = 400

	run_nr = 26
		
	sigma_entries = [0,1, 2, 3]
	#sigma_entries = [1]
	local = False
	
	export_conductance_final_line(W, E_min, E_max, n_E, n_inst, Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, run_nr, n_vec_KPM, n_moments_KPM, sigma_entries, local)
	

	
	
if __name__ == '__main__':
	main()




	
