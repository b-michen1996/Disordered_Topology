import os
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import kwant
import numpy as np
import gc
from matplotlib import pyplot as plt

from kpm_tools import bloch
from kpm_tools.bloch import wrap_velocity
from kpm_tools.hamiltonians import haldane_pbc

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



def make_system_CI(Nx, Ny, r, W):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""	
	disorder = W * np.ones((Nx, Ny, 2)) - 2 * W * np.random.random_sample((Nx, Ny, 2))
	
	# define custom square lattice with two sublattices, because the custom velocity operator does not handle
	# orbitals and the matrix-valued hopping elements
	prim_vecs = [(1,0), (0,1)]	
	basis = [(0,0), (0,0.5)]	
	name=["a", "b"]
	lat = kwant.lattice.general(prim_vecs, basis, name, norbs = 1)
	
	lat_a, lat_b = lat.sublattices
		
	syst = kwant.builder.Builder(kwant.lattice.TranslationalSymmetry([Nx, 0], [0, Ny]))	
	
	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):					
					syst[lat_a(jx,jy)] = disorder[jx,jy,0] + r 
					syst[lat_b(jx,jy)] = disorder[jx,jy,1] - r 
	
	
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat_a, lat_a)] = - 1/2
	syst[kwant.builder.HoppingKind((1, 0), lat_b, lat_b)] = 1/2	
	syst[kwant.builder.HoppingKind((1, 0), lat_a, lat_b)] = 1/(2j)
	syst[kwant.builder.HoppingKind((1, 0), lat_b, lat_a)] = 1/(2j)
		
	# Hopping in y-direction	
	syst[kwant.builder.HoppingKind((0, 1), lat_a, lat_a)] = - 1/2
	syst[kwant.builder.HoppingKind((0, 1), lat_b, lat_b)] = 1/2	
	syst[kwant.builder.HoppingKind((0, 1), lat_a, lat_b)] = -1/(2)
	syst[kwant.builder.HoppingKind((0, 1), lat_b, lat_a)] = 1/(2)
		
	#kwant.plotter.plot(syst)
											 
	return syst, lat


def get_cond_CI_wraparound(Nx, Ny, r, W, n_vec_KPM = 10, n_moments_KPM = 100):
	"""Get linear response conductivity tensor for all energies from KPM method."""
	system, lat = make_system_CI(Nx, Ny, r, W)
	
	# get velocity operators 
	velocity_builder = wrap_velocity(system)
	vx = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[1, 0]), sparse=True)
	vy = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[0, 1]), sparse=True)
	
	# wrap system and finalize 
	system = kwant.wraparound.wraparound(system).finalized()
	
	# calculate conductivity (params=dict(k_x=0, k_y=0) is needed to avoid some cryptic error?)
	conductivity_xy = kwant.kpm.conductivity(system, params=dict(k_x=0, k_y=0), 
										  alpha=vx,
										  beta=vy,
										  num_vectors=n_vec_KPM,
										  num_moments = n_moments_KPM)
	# collect garbage
	del system
	gc.collect()
	
	return conductivity_xy


def get_cond_DC_CI_V2_wraparound(Nx, Ny, r, epsilon_1, epsilon_2, gamma, W, n_vec_KPM = 10, n_moments_KPM = 100):
	"""Get linear response conductivity tensor for all energies from KPM method."""
	system, lat = make_system_DC_CI_V2(Nx, Ny, r, epsilon_1, epsilon_2, gamma, W)
	
	# get velocity operators 
	velocity_builder = wrap_velocity(system)
	vx = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[1, 0]), sparse=True)
	vy = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[0, 1]), sparse=True)
	
	# wrap system and finalize 
	system = kwant.wraparound.wraparound(system).finalized()
	
	#local_vecs = kwant.kpm.LocalVectors(system, where = None)
	
	# calculate conductivity (params=dict(k_x=0, k_y=0) is needed to avoid some cryptic error?)
	conductivity_xy = kwant.kpm.conductivity(system, params=dict(k_x=0, k_y=0), 
										  alpha=vx,
										  beta=vy,
										  num_vectors=n_vec_KPM,
										  num_moments = n_moments_KPM)
	# collect garbage
	del system
	gc.collect()
	
	return conductivity_xy


def get_cond_DC_CI_wraparound(Nx, Ny, r_1, r_2, gamma, W, phi = 0, n_vec_KPM = 10, n_moments_KPM = 100):
	"""Get linear response conductivity tensor for all energies from KPM method."""
	system, lat = make_system_DC_CI(Nx, Ny, r_1, r_2, gamma, W, phi)
	
	# get velocity operators 
	velocity_builder = wrap_velocity(system)
	vx = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[1, 0]), sparse=True)
	vy = velocity_builder.hamiltonian_submatrix(params=dict(k_x=0, k_y=0, direction=[0, 1]), sparse=True)
	
	# wrap system and finalize 
	system = kwant.wraparound.wraparound(system).finalized()
	
	# calculate conductivity (params=dict(k_x=0, k_y=0) is needed to avoid some cryptic error?)
	conductivity_xy = kwant.kpm.conductivity(system, params=dict(k_x=0, k_y=0), 
										  alpha=vx,
										  beta=vy,
										  num_vectors=n_vec_KPM,
										  num_moments = n_moments_KPM)
	# collect garbage
	del system
	gc.collect()
	
	return conductivity_xy


def get_DOS(Nx, Ny, r_1, r_2, gamma, W, PBC = 1., n_vec_KPM = 10, n_moments_KPM = 100, edge_offset = 20):
	"""Get DOS for all energies from KPM method."""
	system, lat = make_system_DC_CI(Nx, Ny, r_1, r_2, gamma, W, PBC)
	#system, lat = make_system_CI(Nx, Ny, r_1, W, PBC)

	dos = kwant.kpm.SpectralDensity(system, num_vectors=n_vec_KPM, num_moments = n_moments_KPM)
	#function for sampling
	def	where(s):
		s_x = s.pos[0]
		s_y = s.pos[1]
		return (abs(s_x - Nx /2) > edge_offset) & (abs(s_y - Ny /2) > edge_offset)
	
	def where_2(s):
		return True

	# collect garbage
	del system
	gc.collect()
			
	return dos, lat
	

def plot_cond_CI(Nx, Ny, r, W, n_vec_KPM = 10, n_moments_KPM = 100):
	"""Plot conductance of disordered CI with PBC."""		
	cond = get_cond_CI_wraparound(Nx, Ny, r, W, n_vec_KPM, n_moments_KPM)
	
	energies = cond.energies
	
	# conductance should be normalized by the area covered by the vectors used to approximate the 
	# evaluation of the trace. Here, we use random vectors that cover the whole system, so we must 
	# multiply this area by the number of vectors.
	area_normalization = Nx * Ny
	
	plot_data = np.array([cond(e, temperature=0.).real / area_normalization for e in energies])
	
	fig = plt.figure(figsize=(6, 18), layout = "tight")
	plt.plot(energies, plot_data)
	#plt.ylim([-2, 2])
	
	plt.show()
	
	return


def plot_cond_DC_CI_V2(Nx, Ny, r, epsilon_1, epsilon_2, gamma, W, n_vec_KPM = 10, n_moments_KPM = 100):
	"""Plot conductance of disordered CI with PBC."""		
	cond = get_cond_DC_CI_V2_wraparound(Nx, Ny, r, epsilon_1, epsilon_2, gamma, W, n_vec_KPM, n_moments_KPM)
	
	energies = cond.energies
	
	# conductance should be normalized by the area covered by the vectors used to approximate the 
	# evaluation of the trace. Here, we use random vectors that cover the whole system, so we must 
	# multiply this area by the number of vectors.
	area_normalization = Nx * Ny
	
	plot_data = np.array([cond(e, temperature=0.).real / area_normalization for e in energies])
	
	fig = plt.figure(figsize=(6, 18), layout = "tight")
	plt.plot(energies, plot_data)
	#plt.ylim([-2, 2])
	plt.title(fr"$r = {r}, \epsilon_1 = {epsilon_1}, \epsilon_2 = {epsilon_2}, \gamma = {gamma}, W = {W}$, n_vec_KPM = {n_vec_KPM}$, n_moments_KPM = {n_moments_KPM}$")
	
	plt.show()
	
	return


def plot_cond_DC_CI(Nx, Ny,  r_1, r_2, gamma, W, phi = 0., n_vec_KPM = 10, n_moments_KPM = 100):
	"""Plot conductance of disordered CI with PBC."""		
	cond = get_cond_DC_CI_wraparound(Nx, Ny, r_1, r_2, gamma, W, phi, n_vec_KPM, n_moments_KPM)
	
	energies = cond.energies
	
	# conductance should be normalized by the area covered by the vectors used to approximate the 
	# evaluation of the trace. Here, we use random vectors that cover the whole system, so we must 
	# multiply this area by the number of vectors.
	area_normalization = Nx * Ny
	
	plot_data = np.array([cond(e, temperature=0.).real / area_normalization for e in energies])
	
	fig = plt.figure(figsize=(6, 18), layout = "tight")
	plt.plot(energies, plot_data)
	#plt.ylim([-2, 2])
	plt.title(fr"$r_1 = {r_1}, r_2 = {r_2}, \gamma = {gamma}, W = {W}, \phi = {phi}$")
	
	plt.show()
	
	return

def plot_cond_DC_CI_moments(Nx, Ny,  r_1, r_2, gamma, W, phi = 0., n_vec_KPM = [10], n_moments_KPM = [100]):
	"""Plot conductance of disordered CI with PBC."""		
	n_vals = len(n_vec_KPM)
	
	results = list()
	for j in range(n_vals):
		cond = get_cond_DC_CI_wraparound(Nx, Ny, r_1, r_2, gamma, W, phi, n_vec_KPM[j], n_moments_KPM[j])
		energies = cond.energies
		area_normalization = Nx * Ny
		plot_data_j = np.array([cond(e, temperature=0.).real / area_normalization for e in energies])
		del cond
		gc.collect()
		results.append([energies, plot_data_j])	
		print(f"Data point {j} calcuklated")
	
	fig = plt.figure(figsize=(6, 18), layout = "tight")
	
	for j in range(n_vals):
		energies_j = results[j][0]
		plot_data_j = results[j][1]
		plt.plot(energies_j, plot_data_j, label = rf"$n_v = {n_vec_KPM[j]}, n_m = {n_moments_KPM[j]}$")
	#plt.ylim([-2, 2])
	plt.title(fr"$r_1 = {r_1}, r_2 = {r_2}, \gamma = {gamma}, W = {W}, \phi = {phi}$")
	plt.legend()
	
	plt.show()
	
	return

	

def main():
	W_min = 2.
	W_max = 8

	E_min = -1.5
	E_max = 1.5
	
	n_W = 41
	n_E = 41
	n_inst = 1
		
	r_1 = 1
	r_2 = 1.	
	  
	Nx = 50
	Ny = 50
	
		
	W = 0.2
	
	n_vec_KPM = 50
	n_moments_KPM = 100
	
	
	r = 1
	
	#plot_cond_CI(Nx, Ny, r, W, n_vec_KPM, n_moments_KPM)

	r_1 = 1.9
	r_2 = 1
	gamma = 2
	phi = 0.00
	
	#plot_cond_DC_CI(Nx, Ny, r_1, r_2, gamma, W, phi, n_vec_KPM, n_moments_KPM)
	n_vec_KPM = [50, 50, 50]
	n_moments_KPM = [100, 125, 150]
	plot_cond_DC_CI_moments(Nx, Ny,  r_1, r_2, gamma, W, phi, n_vec_KPM, n_moments_KPM)
	
	r = 1
	epsilon_1 = 0.3
	epsilon_2 = 2
	gamma = 1.5
	
	#plot_cond_DC_CI_V2(Nx, Ny, r, epsilon_1, epsilon_2, gamma, W, n_vec_KPM, n_moments_KPM)
	

	
	
	
	
	
	
	
	
	
if __name__ == '__main__':
	main()




	
