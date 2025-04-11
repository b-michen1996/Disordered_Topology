import os
default_n_threads = 128
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

import numpy as np
import scipy.sparse
import scipy.linalg
import kwant


sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])
sigma_0 = np.array([[1, 0],[0, 1]])	


def H_k_final(kx, ky, r, epsilon_1, epsilon_2, gamma, gamma_2):
	"""Bloch Hamiltonian that exhibits a dipole-like charge separation of topological charge"""
	
	damp = epsilon_1 + epsilon_2 * (1 - np.cos(kx))/2
	
	h_x = gamma * np.sin(kx)
	h_y = damp * np.sin(ky)
	h_z = -(damp + gamma_2) * np.cos(ky)

	return h_x * sigma_x + h_y * sigma_y + h_z * sigma_z


def bulk_energies_final(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2):
	"""Calculate bulk energies"""
	energies = np.zeros((Nx * Ny, 2))
	for jx in range(0, Nx):
		kx = jx * 2 * np.pi / Nx
		for jy in range(0, Ny):
			ky = jy * 2 * np.pi / Ny

			H_k_curr = H_k_final(kx, ky, r, epsilon_1, epsilon_2, gamma, gamma_2)

			j = jx * Ny + jy 
			energies[j, :] = np.linalg.eigvalsh(H_k_curr)
	energies = np.sort(energies.flatten())
	
	return energies


def Hamiltonian_final(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, PBC = 0.):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
	# values for lattice constant and number of orbitals
	a = 1
		
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
		
	lat = kwant.lattice.square(a, norbs = 3)
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
				
	H = syst.hamiltonian_submatrix()
												 
	return H


def operator_X_Y(Nx, Ny):
	"""Get matrix for X operator."""
	# value for lattice constant 
	a = 1.
	sigma_0 = np.array([[1, 0],
				[0, 1]])
		
	lat = kwant.lattice.square(a, norbs = 2)
	syst_X = kwant.Builder()	
	syst_Y = kwant.Builder()	
	
	def X_coord(site):
		x,y = site.tag
		return x * sigma_0
	
	def Y_coord(site):
		x,y = site.tag		
		return y * sigma_0
	
	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):					
					syst_X[lat(jx,jy)] = X_coord
					syst_Y[lat(jx,jy)] = Y_coord
				
	syst_X = syst_X.finalized()
	syst_Y = syst_Y.finalized()
	
	X = syst_X.hamiltonian_submatrix()
	Y = syst_Y.hamiltonian_submatrix()
													 
	return X,Y


def spec_loc(kappa, x, y, E, H, X, Y):
	"""Build spectral localizer matrix."""
	size = H.shape[0]
	Id_matrix = np.identity(size)
	L = np.block([[H - E * Id_matrix, kappa * (X - 1j * Y - (x - 1j * y) * Id_matrix)],
	[kappa * (X + 1j * Y - (x + 1j * y) * Id_matrix), -H + E * Id_matrix]])

	return L


def get_localizer_gap(kappa, E, H, X, Y, X_array, Y_array, n_pos, sparse = True):
	"""Calculate localizer gap and topological index Q if not sparse."""
	gaps = np.zeros((n_pos, n_pos))
	Q_vals = np.zeros((n_pos, n_pos))

	for jx in range(n_pos):
		for jy in range(n_pos):
			x = X_array[jx, jy]
			y = Y_array[jx, jy]
			
			# get localizer matrix
			localizer = spec_loc(kappa, x, y, E, H, X, Y)

			if sparse:
				# convert to sparse
				localizer_sparse = scipy.sparse.csr_matrix(localizer)
				try:
					localizer_ev = scipy.sparse.linalg.eigsh(localizer_sparse, k = 2, which = "SM", return_eigenvectors = False)   
				except:
					localizer_ev = [0,0]

				gaps[jx, jy] = min(np.abs(localizer_ev))
				
			else:
				eigenvalues = np.linalg.eigvalsh(localizer)
				
				# get localizer gap at this position
				gaps[jx, jy] = min(np.sort(np.abs(eigenvalues)))
				
				# get value of index Q at this position
				n_positive = sum(eigenvalues > 0)
				n_negative = sum(eigenvalues < 0)
				
				Q_vals[jx, jy] = (n_positive - n_negative)/2
				
	return np.min(gaps), np.mean(Q_vals)


def export_sl_gap_final_energy(kappa, Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, n_pos, E_min, E_max, n_E, run_nr = 1, sparse = True):
	"""Export the minimal spectral localizer gap of n_pos^2 reference positions in the bulk as a function of energy"""
	# create folder
	foldername = "s_l_final_energy_result/s_l_final_energy_run_" + str(run_nr)

	print(f"Starting sl_final run {run_nr}")
	
	try:
			os.makedirs(foldername)
	except:
			pass 

	# create arrays X, Y for reference positions in the center Wigner-Seitz cell
	x_vals = np.linspace(Nx/2 - 1, Nx/2, n_pos, endpoint=False)
	y_vals = np.linspace(Ny/2 - 1, Ny/2, n_pos, endpoint=False)

	if n_pos == 1:
			x_vals = (Nx - 1)/2
			y_vals = (Ny - 1)/2      

	X_array, Y_array = np.meshgrid(x_vals, y_vals)

	# create array E_vals of energies
	E_vals = np.linspace(E_min, E_max, n_E)

	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')           

	# export bulk energies of the clean system
	E_bulk = bulk_energies_final(100, 100, r, epsilon_1, epsilon_2, gamma, gamma_2)

	np.savetxt(foldername + "/bulk_energies.txt", E_bulk, delimiter=' ')           

	# Build operator matrices
	X, Y = operator_X_Y(Nx, Ny)

	H = Hamiltonian_final(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, 0)	
					
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("kappa " + str(kappa) + str("\n"))
	f_parameter.write("r " + str(r) + str("\n"))
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))
	f_parameter.write("gamma " + str(gamma) + str("\n"))
	f_parameter.write("gamma_2 " + str(gamma_2) + str("\n"))
	f_parameter.write("W " + str(W) + str("\n"))

	f_parameter.write("Nx " + str(Nx) + str("\n"))
	f_parameter.write("Ny " + str(Ny) + str("\n"))

	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	f_parameter.write("n_E " + str(n_E) + str("\n"))

	f_parameter.write("n_pos " + str(n_pos) + str("\n"))
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))
	f_parameter.write("sparse " + str(sparse) + str("\n"))
			
	f_parameter.close()
	
	# calculate and export minimum of localizer gap in the bulk at the given energies
	result_sl_gap = np.zeros(n_E)
	result_sl_Q_index = np.zeros((n_E, 2))
	
	for j_E in range(n_E):
		E = E_vals[j_E]
		gap_E, Q_E = get_localizer_gap(kappa, E, H, X, Y, X_array, Y_array, n_pos, sparse)
		
		if sparse:
			result_sl_gap[j_E] = gap_E
			np.savetxt(foldername + "/result_sl_gap.txt", result_sl_gap, delimiter=' ')    
		else:
			result_sl_gap[j_E] = gap_E
			result_sl_Q_index[j_E, 0] = E
			result_sl_Q_index[j_E, 1] = Q_E

			np.savetxt(foldername + "/result_sl_gap.txt", result_sl_gap, delimiter=' ')    
			np.savetxt(foldername + "/result_sl_Q_index.txt", result_sl_Q_index, delimiter=' ')    
															
	return

def export_sl_gap_final_energy_contour(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, n_pos, E_min, E_max, n_E, kappa_min, kappa_max, n_kappa, run_nr = 1, sparse = True):
	"""Export the minimal spectral localizer gap of n_pos^2 reference positions in the bulk as a function of energy"""
	# create folder
	foldername = "s_l_final_contour_result/s_l_final_contour_run_" + str(run_nr)

	print(f"Starting sl_final_contour run {run_nr}")
	
	try:
			os.makedirs(foldername)
	except:
			pass 

	# arrays for datapoint
	x_vals = np.linspace((Nx - 1)/3 , (Nx - 1)/4, n_pos)
	y_vals = np.linspace(Ny/2 - 1, Ny/2, n_pos)

	# create arrays X, Y for reference positions in the center Wigner-Seitz cell
	x_vals = np.linspace(Nx/2 - 1, Nx/2, n_pos, endpoint=False)
	y_vals = np.linspace(Ny/2 - 1, Ny/2, n_pos, endpoint=False)

	if n_pos == 1:
			x_vals = (Nx - 1)/2
			y_vals = (Ny - 1)/2      

	X_array, Y_array = np.meshgrid(x_vals, y_vals)

	# create array E_vals of energies
	E_vals = np.linspace(E_min, E_max, n_E)
	Kappa_vals = np.linspace(kappa_min, kappa_max, n_kappa)

	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')         
	np.savetxt(foldername + "/Kappa_vals.txt", Kappa_vals, delimiter=' ')           

	# export bulk energies of the clean system
	E_bulk = bulk_energies_final(100, 100, r, epsilon_1, epsilon_2, gamma, gamma_2)

	np.savetxt(foldername + "/bulk_energies.txt", E_bulk, delimiter=' ')           

	# Build operator matrices
	X, Y = operator_X_Y(Nx, Ny)

	H = Hamiltonian_final(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, 0)	
					
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("r " + str(r) + str("\n"))
	f_parameter.write("epsilon_1 " + str(epsilon_1) + str("\n"))
	f_parameter.write("epsilon_2 " + str(epsilon_2) + str("\n"))
	f_parameter.write("gamma " + str(gamma) + str("\n"))
	f_parameter.write("gamma_2 " + str(gamma_2) + str("\n"))
	f_parameter.write("W " + str(W) + str("\n"))

	f_parameter.write("Nx " + str(Nx) + str("\n"))
	f_parameter.write("Ny " + str(Ny) + str("\n"))

	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	f_parameter.write("kappa_min " + str(kappa_min) + str("\n"))
	f_parameter.write("kappa_max " + str(kappa_max) + str("\n"))
	f_parameter.write("n_E " + str(n_E) + str("\n"))
	f_parameter.write("n_kappa " + str(n_kappa) + str("\n"))

	f_parameter.write("n_pos " + str(n_pos) + str("\n"))
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))
	f_parameter.write("sparse " + str(sparse) + str("\n"))
			
	f_parameter.close()
	
	# calculate and export minimum of localizer gap in the bulk at the given energies
	result_sl_gap = np.zeros((n_E, n_kappa))
	result_sl_Q_index = np.zeros((n_E, n_kappa))
	
	for j_E in range(n_E):
		E = E_vals[j_E]
		for j_kappa in range(0, n_kappa):
			kappa = Kappa_vals[j_kappa]
			gap_E, Q_E = get_localizer_gap(kappa, E, H, X, Y, X_array, Y_array, n_pos, sparse)
			
			if sparse:
				result_sl_gap[j_E, j_kappa] = gap_E
				np.savetxt(foldername + "/result_sl_gap.txt", result_sl_gap, delimiter=' ')    
			else:
				result_sl_gap[j_E, j_kappa] = gap_E				
				result_sl_Q_index[j_E, j_kappa] = Q_E

				np.savetxt(foldername + "/result_sl_gap.txt", result_sl_gap, delimiter=' ')    
				np.savetxt(foldername + "/result_sl_Q_index.txt", result_sl_Q_index, delimiter=' ')    
											  
															
	return


def main():
	
	Nx = 15
	Ny = 15
		
	r = 0.4
	
	epsilon_1 = 0.5
	epsilon_2 = 5
	
	gamma = 4
	gamma_2 = 1
	
	W = 0
	
	kappa = 0.5
	
	E_min = -7
	E_max = 7
	kappa_min = 0
	kappa_max = 2
		
	n_E = 100
	n_kappa = 50
	
	n_pos = 5
	
	run_nr = 2
	
	#export_sl_gap_final_energy(kappa, Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, n_pos, E_min, E_max, n_E, run_nr, sparse = False)
	
	export_sl_gap_final_energy_contour(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, W, n_pos, E_min, E_max, n_E, kappa_min, kappa_max, n_kappa, run_nr, False)
	


if __name__ == '__main__':
	main()