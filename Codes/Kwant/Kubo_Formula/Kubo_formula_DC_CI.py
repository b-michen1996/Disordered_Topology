import os
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import kwant
import numpy as np
import gc
from matplotlib import pyplot as plt

def make_system_DC_CI(Nx, Ny, r_1, r_2, gamma, W, PBC = 1.):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
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

	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy					
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + ((r_1 + r_2)/2) * sigma_z
		
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = gamma * (1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z 
	syst[kwant.builder.HoppingKind((2, 0), lat, lat)] = - (1/2) * sigma_z
	
	# Hopping in y-direction
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = gamma * (1/(2j)) * sigma_y - (1/2) * sigma_z
	
	# Set boundary terms along x-direction
	syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = PBC * (gamma * (1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z)
	syst[kwant.builder.HoppingKind((-(Nx-2), 0), lat, lat)] = -PBC * (1/2) * sigma_z
	
	# Set boundary terms along y-direction
	syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = PBC * (gamma * (1/(2j)) * sigma_y - (1/2) * sigma_z)
	syst.eradicate_dangling()
	
	#kwant.plotter.plot(syst)
											 
	return syst.finalized(), lat


def make_system_DC_CI_disc(R, R_b, r_1, r_2, gamma, W, PBC = 1.):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
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
	
	def circle(pos):
		x, y = pos
		return x ** 2 + y ** 2 < r ** 2
	
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	
	lat = kwant.lattice.square(a, norbs = 2)
	syst = kwant.Builder()

	# initialize by setting zero on-site terms
	syst[(lat(jx,jy) for jx in range (Nx) for jy in range(Ny))] = 0 * sigma_0

	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy					
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + ((r_1 + r_2)/2) * sigma_z
		
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = gamma * (1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z 
	syst[kwant.builder.HoppingKind((2, 0), lat, lat)] = - (1/2) * sigma_z
	
	# Hopping in y-direction
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = gamma * (1/(2j)) * sigma_y - (1/2) * sigma_z
	
	# Set boundary terms along x-direction
	syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = PBC * (gamma * (1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z)
	syst[kwant.builder.HoppingKind((-(Nx-2), 0), lat, lat)] = -PBC * (1/2) * sigma_z
	
	# Set boundary terms along y-direction
	syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = PBC * (gamma * (1/(2j)) * sigma_y - (1/2) * sigma_z)
	
	#kwant.plotter.plot(syst)
											 
	return syst.finalized(), lat



def make_system_CI(Nx, Ny, r, W, PBC = 1.):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
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

	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy					
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + r * sigma_z
		
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = (1/(2j)) * sigma_x - (1/2) * sigma_z 
		
	# Hopping in y-direction
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = (1/(2j)) * sigma_y - (1/2) * sigma_z
	
	# Set boundary terms along x-direction
	syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = PBC * ((1/(2j)) * sigma_x - (1/2) * sigma_z)
	
	# Set boundary terms along y-direction
	syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = PBC * ((1/(2j)) * sigma_y - (1/2) * sigma_z)
	
	#kwant.plotter.plot(syst)
											 
	return syst.finalized(), lat


def get_conductivity(Nx, Ny, r_1, r_2, gamma, W, PBC = 1., n_vec_KPM = 10, n_moments_KPM = 100, edge_offset = 20):
	"""Get linear response conductivity tensor for all energies from KPM method."""
	system, lat = make_system_DC_CI(Nx, Ny, r_1, r_2, gamma, W, PBC)
	#system, lat = make_system_CI(Nx, Ny, r_1, W, PBC)
	
	conductivity_xy = kwant.kpm.conductivity(system, alpha='x', beta='y', num_vectors=n_vec_KPM, num_moments = n_moments_KPM)
	#function for sampling
	def	where(s):
		s_x = s.pos[0]
		s_y = s.pos[1]
		return (abs(s_x - Nx /2) > edge_offset) & (abs(s_y - Ny /2) > edge_offset)
	
	def where_2(s):
		return True
	
	# create random vectors
	rand_vecs_bulk = kwant.kpm.RandomVectors(system, where=where_2)
	#rand_vecs_bulk = kwant.kpm.LocalVectors(system, where)
	
	#conductivity_xy = kwant.kpm.conductivity(system, alpha='x', beta='y', vector_factory = rand_vecs_bulk, num_vectors=n_vec_KPM, num_moments = n_moments_KPM)
	
		
	
	return conductivity_xy, lat


def get_conductivity_disc(R, R_b, r_1, r_2, gamma, W, PBC = 1., n_vec_KPM = 10, n_moments_KPM = 100, edge_offset = 20):
	"""Get linear response conductivity tensor for all energies from KPM method."""
	system, lat = make_system_DC_CI_disc(R, R_b, r_1, r_2, gamma, W, PBC)
	#system, lat = make_system_CI(Nx, Ny, r_1, W, PBC)
	
	conductivity_xy = kwant.kpm.conductivity(system, alpha='x', beta='y', num_vectors=n_vec_KPM, num_moments = n_moments_KPM)
	#function for sampling
	def	where(s):
		s_x = s.pos[0]
		s_y = s.pos[1]
		return (abs(s_x - Nx /2) > edge_offset) & (abs(s_y - Ny /2) > edge_offset)
	
	def where_2(s):
		return True
	
	# create random vectors
	rand_vecs_bulk = kwant.kpm.RandomVectors(system, where=where_2)
	#rand_vecs_bulk = kwant.kpm.LocalVectors(system, where)
	
	#conductivity_xy = kwant.kpm.conductivity(system, alpha='x', beta='y', vector_factory = rand_vecs_bulk, num_vectors=n_vec_KPM, num_moments = n_moments_KPM)
	
		
	
	return conductivity_xy, lat



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
	
	# create random vectors
	rand_vecs_bulk = kwant.kpm.RandomVectors(system, where=where_2)
	#rand_vecs_bulk = kwant.kpm.LocalVectors(system, where)
	
	#conductivity_xy = kwant.kpm.conductivity(system, alpha='x', beta='y', vector_factory = rand_vecs_bulk, num_vectors=n_vec_KPM, num_moments = n_moments_KPM)
	
		
	
	return dos, lat
	

def plot_cond(Nx, Ny, r_1, r_2, gamma, W, PBC = 1., n_vec_KPM = 10, n_moments_KPM = 100):
	
	cond, lat = get_conductivity(Nx, Ny, r_1, r_2, gamma, W, PBC, n_vec_KPM, n_moments_KPM)
	
	energies = cond.energies
	
	# conductance should be normalized by the area covered by the vectors used to approximate the 
	# evaluation of the trace. Here, we use random vectors that cover the whole system, so we must 
	# multiply this area by the number of vectors.
	area_per_site = np.abs(np.cross(*lat.prim_vecs)) / len(lat.sublattices)
	area_normalization = Nx * Ny
	
	print(area_normalization)
	
	plot_data = np.array([cond(e, temperature=0.01).real / area_normalization for e in energies])
	
	fig = plt.figure(figsize=(6, 18), layout = "tight")
	plt.plot(energies, plot_data)
	#plt.ylim([-2, 2])
	
	plt.show()
	
	return


def plot_DOS(Nx, Ny, r_1, r_2, gamma, W, PBC = 1., n_vec_KPM = 10, n_moments_KPM = 100):
	
	dos, lat = get_DOS(Nx, Ny, r_1, r_2, gamma, W, PBC, n_vec_KPM, n_moments_KPM)
	
	energies = dos.energies
	
	# conductance should be normalized by the area covered by the vectors used to approximate the 
	# evaluation of the trace. Here, we use random vectors that cover the whole system, so we must 
	# multiply this area by the number of vectors.
	area_per_site = np.abs(np.cross(*lat.prim_vecs)) / len(lat.sublattices)
	area_normalization = Nx * Ny
	
	print(area_normalization)
	
	plot_data = np.array([dos(e) for e in energies])
	
	fig = plt.figure(figsize=(6, 18), layout = "tight")
	plt.plot(energies, plot_data)
	#plt.ylim([-2, 2])
	
	plt.show()
	
	return

	
	
def export_conductance_double_cone_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, N_lead_start, N_lead_stop, r_1, r_2, gamma, t_lead, run_nr, PBC = 1., direction = "x"):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "G_DC_CI_model_results/DC_CI_model_run_" + str(run_nr)

	print(f"Starting DC_CI run {run_nr}")
	
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
	
	f_parameter.write("r_1 " + str(r_1) + str("\n"))	
	f_parameter.write("r_2 " + str(r_2) + str("\n"))	
	f_parameter.write("gamma " + str(gamma) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
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
			syst = make_system_DC_CI(Nx, Ny, N_lead_start, N_lead_stop, r_1, r_2, gamma, W_curr, t_lead, PBC, direction)			
			
			# calculate transmission for all energy values
			data_j_inst = np.zeros(n_E)			
			for j_E in range(n_E):
				energy = E_vals[j_E]
				smatrix = kwant.smatrix(syst, energy)					
				data_j_inst[j_E] = smatrix.transmission(1, 0)
			
			# collect garbage
			del syst
			gc.collect()
			
			print(f"j_inst = {j_inst + 1} of {n_inst}, j_W = {j_W + 1} of {n_W}, W = {W_curr} ")	
						
			result[:, j_W] += (1 / n_inst) * data_j_inst 
			
			np.savetxt(foldername + "/result.txt", result, delimiter=' ')  		
								
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
	
	gamma = 2
	
	t_lead = 5
	

	PBC = 1.	
	  
	Nx = 100
	Ny = 100
	
	N_lead_start = 0
	N_lead_stop = 100
	
	direction = "y"
		
	run_nr = 58
	
	#export_conductance_double_cone_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, N_lead_start, N_lead_stop, r_1, r_2, gamma, t_lead, run_nr, PBC, direction)
	W = 0
	n_vec_KPM = 50
	n_moments_KPM = 200
	PBC = 1.
	r_1 = 1
	r_2 = 3
	plot_cond(Nx, Ny, r_1, r_2, gamma, W, PBC = PBC, n_vec_KPM = n_vec_KPM, n_moments_KPM = n_moments_KPM)
	#plot_DOS(Nx, Ny, r_1, r_2, gamma, W, PBC = PBC, n_vec_KPM = n_vec_KPM, n_moments_KPM = n_moments_KPM)
	
	#test_cond = get_conductivity(Nx, Ny, r_1, r_2, gamma, W)
	
	#print("sigma_xy = ", test_cond(0., temperature = 0.0).real / (Nx * Ny))
	r = 3
	#syst = make_system_CI(Nx, Ny, r, W, PBC)
	#kwant.plotter.current(syst)
	
	
	
	
	
	
	
if __name__ == '__main__':
	main()




	
