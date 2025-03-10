import os
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import kwant
import numpy as np
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

def make_system_DC_CI(Nx, Ny, r_1, r_2, gamma, W, phi = 0):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
		
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
					syst[lat_a(jx,jy)] = disorder[jx,jy,0] + (r_1 + r_2)/2 
					syst[lat_b(jx,jy)] = disorder[jx,jy,1] - (r_1 + r_2)/2 
	
	
	# Hopping in x-direction
	
	# set (1,0) hopping term gamma * (1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z 
	syst[kwant.builder.HoppingKind((1, 0), lat_a, lat_a)] = (r_1 - r_2)/4
	syst[kwant.builder.HoppingKind((1, 0), lat_b, lat_b)] = -(r_1 - r_2)/4
	syst[kwant.builder.HoppingKind((1, 0), lat_a, lat_b)] = gamma * 1/(2j)
	syst[kwant.builder.HoppingKind((1, 0), lat_b, lat_a)] = gamma * 1/(2j)
	
	# set (2,0) hopping term -(1/2) * sigma_z
	syst[kwant.builder.HoppingKind((2, 0), lat_a, lat_a)] = -1/2
	syst[kwant.builder.HoppingKind((2, 0), lat_b, lat_b)] = 1/2
		
	# Hopping in y-direction
	
	def hop_y_sz_a(site_1, site_2):
		jy = site_1.tag[0]
		phase = np.exp(1j * phi * jy) 
		return - phase * 1/2
	
	def hop_y_sz_b(site_1, site_2):
		jy = site_1.tag[0]
		phase = np.exp(1j * phi * jy) 
		return phase * 1/2
	
	def hop_y_sy_ab(site_1, site_2):
		jy = site_1.tag[0]
		phase = np.exp(1j * phi * jy) 
		return -phase * gamma * 1/2
	
	def hop_y_sy_ba(site_1, site_2):
		jy = site_1.tag[0]
		phase = np.exp(1j * phi * jy) 
		return phase * gamma * 1/2
	
	# set (0,1) hopping term gamma * (1/(2j)) * sigma_y - (1/2) * sigma_z
	syst[kwant.builder.HoppingKind((0, 1), lat_a, lat_a)] = hop_y_sz_a
	syst[kwant.builder.HoppingKind((0, 1), lat_b, lat_b)] = hop_y_sz_b
	syst[kwant.builder.HoppingKind((0, 1), lat_a, lat_b)] = hop_y_sy_ab
	syst[kwant.builder.HoppingKind((0, 1), lat_b, lat_a)] = hop_y_sy_ba
		
	#kwant.plotter.plot(syst)
											 
	return syst, lat


def make_system_DC_CI_V2(Nx, Ny, r, epsilon_1, epsilon_2, gamma, W):
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
					syst[lat_a(jx,jy)] = disorder[jx,jy,0] + r 
					syst[lat_b(jx,jy)] = disorder[jx,jy,1] - r
	
	# matrices that contain hoppings
	# matrices acting in orbital space to build the system
	sigma_x = np.array([[0, 1],
				[1, 0]])
	
	sigma_y = np.array([[0, -1j],
				[1j, 0]])
	
	sigma_z = np.array([[1, 0],
				[0, -1]])
	
	sigma_0 = np.array([[1, 0],
				[0, 1]])	
	
	hop_mat_dx = (gamma * (epsilon_1 + epsilon_2/2)/(2j)) * sigma_x
	hop_mat_2dx = - (1/2) * sigma_z
	hop_mat_dy = (gamma * (epsilon_1 + epsilon_2/2)/(2j)) * sigma_y - ((epsilon_1 + epsilon_2/2) / 2) * sigma_z
	hop_mat_2dy = -(gamma * epsilon_2/(8j)) * sigma_y 
	hop_mat_dx_dy = -(gamma * epsilon_2/(8j)) * sigma_x + (epsilon_2 / 8) * sigma_z
	hop_mat_dx_mdy = (gamma * epsilon_2/(8j)) * sigma_x + (epsilon_2 / 8) * sigma_z
	
	lat_list = (lat_a, lat_b)
	for l_1 in range(2):
		for l_2 in range(2):
			lat_l1 = lat_list[l_1]
			lat_l2 = lat_list[l_2]
			
			# Hopping in x-direction		
			syst[kwant.builder.HoppingKind((1, 0), lat_l1, lat_l2)] = hop_mat_dx[l_1, l_2]
			syst[kwant.builder.HoppingKind((2, 0), lat_l1, lat_l2)] = hop_mat_2dx[l_1, l_2]
			
			# Hopping in y-direction
			syst[kwant.builder.HoppingKind((0, 1), lat_l1, lat_l2)] = hop_mat_dy[l_1, l_2]
			syst[kwant.builder.HoppingKind((0, 2), lat_l1, lat_l2)] = hop_mat_2dy[l_1, l_2]
			
			# Hopping in (x+y)-direction
			syst[kwant.builder.HoppingKind((1, 1), lat_l1, lat_l2)] = hop_mat_dx_dy[l_1, l_2]

			# Hopping in (x-y)-direction
			syst[kwant.builder.HoppingKind((1, -1), lat_l1, lat_l2)] = hop_mat_dx_mdy[l_1, l_2]			
	
	#kwant.plotter.plot(syst)
											 
	return syst, lat


def get_cond_DC_CI_V2_wraparound(Nx, Ny, r, epsilon_1, epsilon_2, gamma, W, E_vals, n_vec_KPM = 10, n_moments_KPM = 100, sigma_entries = [1]):
	"""Get linear response conductivity tensor for all energies from KPM method."""
	system, lat = make_system_DC_CI_V2(Nx, Ny, r, epsilon_1, epsilon_2, gamma, W)
	
	# get velocity operators 
	velocity_builder = wrap_velocity(system)
	vx = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[1, 0]), sparse=True)
	vy = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[0, 1]), sparse=True)
	
	# wrap system and finalize 
	system = kwant.wraparound.wraparound(system).finalized()
	area_normalization = Nx * Ny
	
	cond_elements = [(vx,vx), (vx,vy), (vy,vx) ,(vy,vy)]
	result = list()
	for j in sigma_entries:
		v1_j = cond_elements[j][0]		
		v2_j = cond_elements[j][1]		
		# calculate conductivity (params=dict(k_x=0, k_y=0) is needed to avoid some cryptic error?)
		conductivity = kwant.kpm.conductivity(system, params=dict(k_x=0, k_y=0), 
											alpha=v1_j,
											beta=v2_j,
											num_vectors=n_vec_KPM,
											num_moments = n_moments_KPM)
		
		result.append(np.array([conductivity(energy, temperature=0.).real / area_normalization for energy in E_vals]))
		# collect garbage 
		del conductivity
		gc.collect()	
		
	# collect garbage
	del system, vx, vy, cond_elements	
	gc.collect()
	
	return result


def get_cond_DC_CI_wraparound(Nx, Ny, r_1, r_2, gamma, W, E_vals, phi = 0, n_vec_KPM = 10, n_moments_KPM = 100, sigma_entries = [1]):
	"""Get linear response conductivity tensor for all energies from KPM method."""
	system, lat = make_system_DC_CI(Nx, Ny, r_1, r_2, gamma, W, phi)
	
	# get velocity operators 
	velocity_builder = wrap_velocity(system)
	vx = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[1, 0]), sparse=True)
	vy = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[0, 1]), sparse=True)
	
	# wrap system and finalize 
	system = kwant.wraparound.wraparound(system).finalized()
	area_normalization = Nx * Ny
	
	cond_elements = [(vx,vx), (vx,vy), (vy,vx) ,(vy,vy)]
	result = list()
	for j in sigma_entries:
		v1_j = cond_elements[j][0]		
		v2_j = cond_elements[j][1]		
		# calculate conductivity (params=dict(k_x=0, k_y=0) is needed to avoid some cryptic error?)
		conductivity = kwant.kpm.conductivity(system, params=dict(k_x=0, k_y=0), 
											alpha=v1_j,
											beta=v2_j,
											num_vectors=n_vec_KPM,
											num_moments = n_moments_KPM)
		
		result.append(np.array([conductivity(energy, temperature=0.).real / area_normalization for energy in E_vals]))
		# collect garbage 
		del conductivity
		gc.collect()	
		
	# collect garbage
	del system, vx, vy, cond_elements	
	gc.collect()
	
	return result


def export_conductance_DC_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r_1, r_2, gamma, run_nr, n_vec_KPM = 10, n_moments_KPM = 100, sigma_entries = [0,1,2,3]):
	"""Compute conductance for disorder realizations of the double cone CI model frome Kubo formula and export"""
	# create folder
	foldername = "Kubo_DC_CI_results/Kubo_DC_CI_run_" + str(run_nr)

	print(f"Starting Kubo_DC_CI run {run_nr}")
	
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
	
	f_parameter.write("r_1 " + str(r_1) + str("\n"))	
	f_parameter.write("r_2 " + str(r_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))	
	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	f_parameter.write("n_vec_KPM " + str(n_vec_KPM) + str("\n"))	
	f_parameter.write("n_moments_KPM " + str(n_moments_KPM) + str("\n"))
	f_parameter.write("sigma_entries " + str(sigma_entries) + str("\n"))		
			
	f_parameter.close()
					
	result_sigma_xx = np.zeros((n_E, n_W))
	result_sigma_xy = np.zeros((n_E, n_W))
	result_sigma_yx = np.zeros((n_E, n_W))
	result_sigma_yy = np.zeros((n_E, n_W))
	
	results = (result_sigma_xx, result_sigma_xy, result_sigma_yx, result_sigma_yy)
	results_filenames = ("/result_sigma_xx.txt", "/result_sigma_xy.txt", "/result_sigma_yx.txt", "/result_sigma_yy.txt")
	
	phi = 0
	
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):			
			# get conductance			
			cond = get_cond_DC_CI_wraparound(Nx, Ny, r_1, r_2, gamma, W_curr, E_vals, phi, n_vec_KPM, n_moments_KPM, sigma_entries)					

			for j_entry in range(len(sigma_entries)):								
				cond_element = cond[j_entry]										
				entry_index = sigma_entries[j_entry]						
				results[entry_index][:, j_W] += (1 / n_inst) * cond_element
				
				filename = results_filenames[entry_index]
				np.savetxt(foldername + filename, results[entry_index], delimiter=' ')  	

			del cond
			gc.collect()	
				
			print(f"j_inst = {j_inst + 1} of {n_inst}, j_W = {j_W + 1} of {n_W}, W = {W_curr}")			
								
	return 


def export_conductance_DC_CI_V2(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r, epsilon_1, epsilon_2, gamma, run_nr, n_vec_KPM = 10, n_moments_KPM = 100, sigma_entries = [0,1,2,3]):
	"""Compute conductance for disorder realizations of the double cone CI model frome Kubo formula and export"""
	# create folder
	foldername = "Kubo_DC_CI_V2_results/Kubo_DC_CI_V2_run_" + str(run_nr)

	print(f"Starting Kubo_DC_CI_V2 run {run_nr}")
	
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
	
	f_parameter.write("r " + str(r) + str("\n"))	
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))	
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))	
	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	f_parameter.write("n_vec_KPM " + str(n_vec_KPM) + str("\n"))	
	f_parameter.write("n_moments_KPM " + str(n_moments_KPM) + str("\n"))
	f_parameter.write("sigma_entries " + str(sigma_entries) + str("\n"))	
			
	f_parameter.close()
					
	result_sigma_xx = np.zeros((n_E, n_W))
	result_sigma_xy = np.zeros((n_E, n_W))
	result_sigma_yx = np.zeros((n_E, n_W))
	result_sigma_yy = np.zeros((n_E, n_W))
	
	results = (result_sigma_xx, result_sigma_xy, result_sigma_yx, result_sigma_yy)
	results_filenames = ("/result_sigma_xx.txt", "/result_sigma_xy.txt", "/result_sigma_yx.txt", "/result_sigma_yy.txt")
			
	for j in range(4):
		filename = results_filenames[j]
		np.savetxt(foldername + filename, results[j], delimiter=' ')  			
		
	area_normalization = Nx * Ny	
	
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):			
			# get conductance			
			cond = get_cond_DC_CI_V2_wraparound(Nx, Ny, r, epsilon_1, epsilon_2, gamma, W_curr, E_vals, n_vec_KPM, n_moments_KPM, sigma_entries)					

			for j_entry in range(len(sigma_entries)):								
				cond_element = cond[j_entry]										
				entry_index = sigma_entries[j_entry]						
				results[entry_index][:, j_W] += (1 / n_inst) * cond_element
				
				filename = results_filenames[entry_index]
				np.savetxt(foldername + filename, results[entry_index], delimiter=' ')  	

			del cond
			gc.collect()	
				
			print(f"j_inst = {j_inst + 1} of {n_inst}, j_W = {j_W + 1} of {n_W}, W = {W_curr}")	
											
	return 

	

def main():
	W_min = 0.
	W_max = 4

	E_min = -5
	E_max = 5
	
	n_W = 40
	n_E = 1000
	n_inst = 1
		
	Nx = 100
	Ny = 100
		
	n_vec_KPM = 100
	n_moments_KPM = 150

	r_1 = 2.3
	r_2 = 1
	gamma = 2
	
	run_nr = 2
	
	export_conductance_DC_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r_1, r_2, gamma, run_nr, n_vec_KPM, n_moments_KPM)
	
	
	r = 1.5
	epsilon_1 = 0.3
	epsilon_2 = 2
	gamma = 1.5
	
	run_nr = 4
	
	E_min = -5
	E_max = 5
	
	sigma_entries = [0,1, 2, 3]
		
	#export_conductance_DC_CI_V2(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r, epsilon_1, epsilon_2, gamma, run_nr, n_vec_KPM, n_moments_KPM, sigma_entries)
	#gc.collect()	
	
	
if __name__ == '__main__':
	main()




	
