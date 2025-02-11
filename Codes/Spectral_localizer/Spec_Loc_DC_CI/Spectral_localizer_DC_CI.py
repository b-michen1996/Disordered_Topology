import numpy as np
import scipy.sparse
import scipy.linalg
import os

sigma_0 = np.identity(2)
sigma_x = np.array([[0,1], [1,0]])
sigma_y = np.array([[0,-1j], [1j,0]])
sigma_z = np.array([[1,0], [0,-1]])

def coord_ind(jx, jy, orbital, Ny):
	"Return coordinates,  " 
	return 2 * (jx * Ny + jy) + orbital


def H_QSH_k(kx, ky, alpha, beta, gamma, m):
	"""Bloch Hamiltonian"""
	epsilon = 2 * gamma * (2 - np.cos(kx) - np.cos(ky))

	dx_m_idy = alpha * (np.sin(kx) + 1j * np.sin(ky))
	dz = m + 2 * beta * (2 - np.cos(kx) - np.cos(ky))                                   
	
	res = np.array([[epsilon + dz, dx_m_idy],
		[np.conj(dx_m_idy), epsilon - dz]])
	
	return res 


def H_DC_CI_k(kx, ky, r1, r2):
	"""Bloch Hamiltonian for the double cone CI."""
	r_k =  r1 * (1 + np.cos(kx)) / 2 + r2 * ((1 - np.cos(kx)) / 2) 
	h_x = np.sin(kx)
	h_y = np.sin(ky)
	h_z = (r_k - (np.cos(2 * kx) + np.cos(ky))) 
	
	return h_x * sigma_x + h_y * sigma_y + h_z* sigma_z
	

def bulk_energies_QSH(Nx, Ny, alpha, beta, gamma, m):
	"""Calculate bulk energies"""
	energies = np.zeros((Nx * Ny, 2))
	for jx in range(0, Nx):
		kx = jx * 2 * np.pi / Nx
		for jy in range(0, Ny):
			ky = jy * 2 * np.pi / Ny

			H_k_curr = H_QSH_k(kx, ky, alpha, beta, gamma, m)

			j = jx * Ny + jy 
			energies[j, :] = np.linalg.eigvalsh(H_k_curr)
	energies = np.sort(energies.flatten())
	
	return energies


def bulk_energies_DC_CI(Nx, Ny, r1, r2):
	"""Calculate bulk energies"""
	energies = np.zeros((Nx * Ny, 2))
	for jx in range(0, Nx):
		kx = jx * 2 * np.pi / Nx
		for jy in range(0, Ny):
			ky = jy * 2 * np.pi / Ny

			H_k_curr = H_DC_CI_k(kx, ky, r1, r2)

			j = jx * Ny + jy 
			energies[j, :] = np.linalg.eigvalsh(H_k_curr)
	energies = np.sort(energies.flatten())
	
	return energies


def Hamiltonian_DC_CI(Nx, Ny, r1, r2, PBC = 0.):
	"""Create tight binding matrix for double cone CI."""
	# lists with non-zero array elements
	data = list()
	row = list()
	col = list()

	# just loop over all elements, shouldn't be critical. Only set upper diagonal elements, 
	# fill in the rest by adding hermitian conjugate
	for jx in range(0, Nx):
		for jy in range(0, Ny):
			j_a = coord_ind(jx, jy, 0, Ny)
			j_b = coord_ind(jx, jy, 1, Ny)                        

			# set on-site terms (only half because we add the Hermitian conjugate later)
			row.append(j_a)
			col.append(j_a)
			data.append((r1 + r2)/4)

			row.append(j_b)
			col.append(j_b)
			data.append(-(r1 + r2)/ 4)

			# set delta_x hopping terms
			if (jx < Nx - 2):
				j_a_p_delta_x = coord_ind(jx + 1, jy, 0, Ny)                         
				j_b_p_delta_x = coord_ind(jx + 1, jy, 1, Ny)

				# set delta_x hoppings
				row.append(j_a)
				col.append(j_a_p_delta_x)
				data.append((r1 - r2)/ 4)

				row.append(j_a)
				col.append(j_b_p_delta_x)
				data.append(1 / (2 * 1j))

				row.append(j_b)
				col.append(j_a_p_delta_x)
				data.append(1 / (2 * 1j))

				row.append(j_b)
				col.append(j_b_p_delta_x)
				data.append(-(r1 - r2)/ 4)
				
				# set 2 delta_x hoppings
				j_a_p_2_delta_x = coord_ind(jx + 2, jy, 0, Ny)                         
				j_b_p_2_delta_x = coord_ind(jx + 2, jy, 1, Ny)
				
				row.append(j_a)
				col.append(j_a_p_2_delta_x)
				data.append(-1/2)
				
				row.append(j_b)
				col.append(j_b_p_2_delta_x)
				data.append(1/2)

			else:	
				if (jx < Nx - 1):
					j_a_p_delta_x = coord_ind(jx + 1, jy, 0, Ny)                         
					j_b_p_delta_x = coord_ind(jx + 1, jy, 1, Ny)

					row.append(j_a)
					col.append(j_a_p_delta_x)
					data.append((r1 - r2)/ 4)

					row.append(j_a)
					col.append(j_b_p_delta_x)
					data.append(1 / (2 * 1j))

					row.append(j_b)
					col.append(j_a_p_delta_x)
					data.append(1 / (2 * 1j))

					row.append(j_b)
					col.append(j_b_p_delta_x)
					data.append(-(r1 - r2)/ 4)
					
					# set 2 delta_x hoppings
					j_a_p_2_delta_x = coord_ind(0, jy, 0, Ny)                         
					j_b_p_2_delta_x = coord_ind(0, jy, 1, Ny)
					
					row.append(j_a)
					col.append(j_a_p_2_delta_x)
					data.append(-PBC * 1/2)
					
					row.append(j_b)
					col.append(j_b_p_2_delta_x)
					data.append(PBC * 1/2)
				else:
					j_a_p_delta_x = coord_ind(0, jy, 0, Ny)                         
					j_b_p_delta_x = coord_ind(0, jy, 1, Ny)

					row.append(j_a)
					col.append(j_a_p_delta_x)
					data.append(PBC * (r1 - r2)/ 4)

					row.append(j_a)
					col.append(j_b_p_delta_x)
					data.append(PBC * 1 / (2 * 1j))

					row.append(j_b)
					col.append(j_a_p_delta_x)
					data.append(PBC * 1 / (2 * 1j))

					row.append(j_b)
					col.append(j_b_p_delta_x)
					data.append(-PBC * (r1 - r2)/ 4)
					
					# set 2 delta_x hoppings
					j_a_p_2_delta_x = coord_ind(1, jy, 0, Ny)                         
					j_b_p_2_delta_x = coord_ind(1, jy, 1, Ny)
					
					row.append(j_a)
					col.append(j_a_p_2_delta_x)
					data.append(-PBC * 1/2)
					
					row.append(j_b)
					col.append(j_b_p_2_delta_x)
					data.append(PBC * 1/2)
					
			# set delta_y hopping terms
			if (jy < Ny - 1):
				j_a_p_delta_y = coord_ind(jx, jy + 1, 0, Ny)                    
				j_b_p_delta_y = coord_ind(jx, jy + 1, 1, Ny)

				row.append(j_a)
				col.append(j_a_p_delta_y)
				data.append(-1/2)

				row.append(j_a)
				col.append(j_b_p_delta_y)
				data.append(-1/2)

				row.append(j_b)
				col.append(j_a_p_delta_y)
				data.append(1/2)

				row.append(j_b)
				col.append(j_b_p_delta_y)
				data.append(1/2)
			else:
				j_a_p_delta_y = coord_ind(jx, 0, 0, Ny)                    
				j_b_p_delta_y = coord_ind(jx, 0, 1, Ny)

				row.append(j_a)
				col.append(j_a_p_delta_y)
				data.append(-PBC * 1/2)

				row.append(j_a)
				col.append(j_b_p_delta_y)
				data.append(-PBC * 1/2)

				row.append(j_b)
				col.append(j_a_p_delta_y)
				data.append(PBC * 1/2)

				row.append(j_b)
				col.append(j_b_p_delta_y)
				data.append(PBC * 1/2)
							
	H = scipy.sparse.csr_matrix((data, (row,col)), shape = (2 * Nx * Ny, 2 * Nx * Ny), dtype= complex)  

	H = H + H.conj().T   

	return H


def Hamiltonian_QSH(Nx, Ny, alpha, beta, gamma, m, PBC = 0.):
		"""Create tight binding matrix for spin up Block Hamiltonian of QSH."""
		# lists with non-zero array elements
		data = list()
		row = list()
		col = list()

		f = np.sqrt(2) * np.exp(-1j * np.pi / 4)

		# just loop over all elements, shouldn't be critical. Only set upper diagonal elements, 
		# fill in the rest by adding hermitian conjugate
		for jx in range(0, Nx):
				for jy in range(0, Ny):
						j_a = coord_ind(jx, jy, 0, Ny)
						j_b = coord_ind(jx, jy, 1, Ny)                        

						# set on-site terms (only half because we add the Hermitian conjugate later)
						row.append(j_a)
						col.append(j_a)
						data.append(2 * (gamma + beta + m/4))

						row.append(j_b)
						col.append(j_b)
						data.append(2 * (gamma - beta - m/4))

						# set delta_x hopping terms
						if (jx < Nx - 1):
								j_a_p_delta_x = coord_ind(jx + 1, jy, 0, Ny)                         
								j_b_p_delta_x = coord_ind(jx + 1, jy, 1, Ny)

								row.append(j_a)
								col.append(j_a_p_delta_x)
								data.append(-(gamma + beta))

								row.append(j_a)
								col.append(j_b_p_delta_x)
								data.append(-1j * alpha/2)

								row.append(j_b)
								col.append(j_a_p_delta_x)
								data.append(-1j * alpha/2)

								row.append(j_b)
								col.append(j_b_p_delta_x)
								data.append(-(gamma - beta))

						else:
								j_a_p_delta_x = coord_ind(0, jy, 0, Ny)                         
								j_b_p_delta_x = coord_ind(0, jy, 1, Ny)

								row.append(j_a)
								col.append(j_a_p_delta_x)
								data.append(-PBC * (gamma + beta))

								row.append(j_a)
								col.append(j_b_p_delta_x)
								data.append(-PBC *1j * alpha/2)

								row.append(j_b)
								col.append(j_a_p_delta_x)
								data.append(-PBC *1j * alpha/2)

								row.append(j_b)
								col.append(j_b_p_delta_x)
								data.append(-PBC *(gamma - beta))
						
						# set delta_y hopping terms
						if (jy < Ny - 1):
								j_a_p_delta_y = coord_ind(jx, jy + 1, 0, Ny)                    
								j_b_p_delta_y = coord_ind(jx, jy + 1, 1, Ny)

								row.append(j_a)
								col.append(j_a_p_delta_y)
								data.append(-(gamma + beta))

								row.append(j_a)
								col.append(j_b_p_delta_y)
								data.append(alpha/2)

								row.append(j_b)
								col.append(j_a_p_delta_y)
								data.append(-alpha/2)

								row.append(j_b)
								col.append(j_b_p_delta_y)
								data.append(-(gamma - beta))
						else:
								j_a_p_delta_y = coord_ind(jx, 0, 0, Ny)                    
								j_b_p_delta_y = coord_ind(jx, 0, 1, Ny)

								row.append(j_a)
								col.append(j_a_p_delta_y)
								data.append(-PBC *(gamma + beta))

								row.append(j_a)
								col.append(j_b_p_delta_y)
								data.append(PBC *alpha/2)

								row.append(j_b)
								col.append(j_a_p_delta_y)
								data.append(-PBC *alpha/2)

								row.append(j_b)
								col.append(j_b_p_delta_y)
								data.append(-PBC *(gamma - beta))
							   
		H = scipy.sparse.csr_matrix((data, (row,col)), shape = (2 * Nx * Ny, 2 * Nx * Ny), dtype= complex)  

		H = H + H.conj().T   

		return H



def operator_X(Nx, Ny):
		diagonal_entries_X = np.tensordot(np.arange(Nx), np.ones(2 * Ny), axes = 0).flatten()

		#print("X entries: \n", diagonal_entries_X)

		X = scipy.sparse.diags(diagonals = diagonal_entries_X)

		return X


def operator_Y(Nx, Ny):
		diagonal_entries_Y = np.tensordot(np.ones(Nx), np.tensordot(np.arange(Ny), np.ones(2), axes = 0), axes = 0).flatten()

		#print("Y entries: \n", diagonal_entries_Y)

		Y = scipy.sparse.diags(diagonals = diagonal_entries_Y)

		return Y


def operator_V_rand(Nx, Ny, W, W_2 = 0):
		random_array = np.random.rand(Nx * Ny)
		disorder_vals = W * (np.ones(2 * Nx * Ny) - 2 * np.tensordot(random_array, np.ones(2), axes = 0).flatten())

		disorder_vals_W2 = W_2 * (np.ones(2 * Nx * Ny) - 2 * np.random.rand(2 * Nx * Ny))

		V_rand = scipy.sparse.diags(diagonals = disorder_vals + disorder_vals_W2)

		return V_rand


def spec_loc(kappa, x, y, E, H, V_rand,  X, Y):
		"""Build spectral localizer matrix."""
		size = H.shape[0]
		Id_matrix = scipy.sparse.identity(size)

		L = scipy.sparse.block_array([[H + V_rand - E * Id_matrix, kappa * (X - 1j * Y - (x - 1j * y) * Id_matrix)],
		[kappa * (X + 1j * Y - (x + 1j * y) * Id_matrix), - (H + V_rand) + E * Id_matrix]], format = "csr")

		return L


def export_sl_gap_DC_CI_rs(kappa, Nx, Ny, r1, r2, W, E, x_min, x_max, y_min, y_max, n_pos, run_nr = 1):
		"""Export spectral localizer gap in real space at fixed energy."""

		# create folder
		foldername = "s_l_DC_CI_rs_gap_result/s_l_DC_CI_rs_gap_run_" + str(run_nr)
		
		try:
				os.makedirs(foldername)
		except:
				pass 

		# create arrays X, Y for reference positions
		x_vals = np.linspace(x_min, x_max, n_pos)
		y_vals = np.linspace(y_min, y_max, n_pos)

		X_array, Y_array = np.meshgrid(x_vals, y_vals)

		np.savetxt(foldername + "/X_array.txt", X_array, delimiter=' ')   
		np.savetxt(foldername + "/Y_array.txt", Y_array, delimiter=' ')   

		# Build operator matrices
		X = operator_X(Nx, Ny)
		Y = operator_Y(Nx, Ny)
		H = Hamiltonian_DC_CI(Nx, Ny, r1, r2, 0)
		disorder = operator_V_rand(Nx, Ny, W)
				
		# export parameters
		f_parameter = open(foldername + "/parameters.txt", "w")
		
		f_parameter.write("kappa " + str(kappa) + str("\n"))
		f_parameter.write("r1 " + str(r1) + str("\n"))
		f_parameter.write("r2  " + str(r2) + str("\n"))
		f_parameter.write("W " + str(W) + str("\n"))

		f_parameter.write("Nx " + str(Nx) + str("\n"))
		f_parameter.write("Ny " + str(Ny) + str("\n"))

		f_parameter.write("E " + str(E) + str("\n"))
		f_parameter.write("x_min " + str(x_min) + str("\n"))
		f_parameter.write("x_max " + str(x_max) + str("\n"))
		f_parameter.write("y_min " + str(y_min) + str("\n"))
		f_parameter.write("y_max " + str(y_max) + str("\n"))

		f_parameter.write("n_pos " + str(n_pos) + str("\n"))
		f_parameter.write("run_nr " + str(run_nr) + str("\n"))
				
		f_parameter.close()

		# calculate and export localizer gaps at reference positions
		result = np.zeros((n_pos, n_pos))
		for jx in range(n_pos):
				for jy in range(n_pos):
						x = X_array[jx, jy]
						y = Y_array[jx, jy]

						localizer = spec_loc(kappa, x, y, E, H, disorder, X, Y)

						sigma, ev = scipy.sparse.linalg.eigsh(localizer, k = 3, which = "SM")     
						gap = min(np.sort(np.abs(sigma)))

						result[jx,jy] = gap

						np.savetxt(foldername + "/result.txt", result, delimiter=' ')    
		
		return result


def export_sl_gap_DC_CI_rs_line(kappa, Nx, Ny, r1, r2, W, E, x_min, x_max, y, n_pos, run_nr = 1):
		"""Export spectral localizer gap along line in real space at fixed energy. """

		# create folder
		foldername = "s_l_DC_CI_rs_line_result/s_l_DC_CI_rs_line_run_" + str(run_nr)
		
		try:
				os.makedirs(foldername)
		except:
				pass 

		# create array for positions
		x_vals = np.linspace(x_min, x_max, n_pos)

		np.savetxt(foldername + "/x_vals.txt", x_vals, delimiter=' ')   
		 

		# Build operator matrices
		X = operator_X(Nx, Ny)
		Y = operator_Y(Nx, Ny)
		H = H = Hamiltonian_DC_CI(Nx, Ny, r1, r2, 0)
		disorder = operator_V_rand(Nx, Ny, W)
				
		# export parameters
		f_parameter = open(foldername + "/parameters.txt", "w")
		
		f_parameter.write("kappa " + str(kappa) + str("\n"))
		f_parameter.write("r1 " + str(r1) + str("\n"))
		f_parameter.write("r2 " + str(r2) + str("\n"))
		f_parameter.write("W " + str(W) + str("\n"))

		f_parameter.write("Nx " + str(Nx) + str("\n"))
		f_parameter.write("Ny " + str(Ny) + str("\n"))

		f_parameter.write("E " + str(E) + str("\n"))
		f_parameter.write("x_min " + str(x_min) + str("\n"))
		f_parameter.write("x_max " + str(x_max) + str("\n"))
		f_parameter.write("y " + str(y) + str("\n"))

		f_parameter.write("n_pos " + str(n_pos) + str("\n"))
		f_parameter.write("run_nr " + str(run_nr) + str("\n"))
				
		f_parameter.close()

		# calculate and export localizer gaps at reference positions
		result = np.zeros(n_pos)
		for jx in range(n_pos):
				
						x = x_vals[jx]
						
						localizer = spec_loc(kappa, x, y, E, H, disorder, X, Y)

						sigma, ev = scipy.sparse.linalg.eigsh(localizer, k = 3, which = "SM")     
						gap = min(np.sort(np.abs(sigma)))

						result[jx] = gap

						np.savetxt(foldername + "/result.txt", result, delimiter=' ')    
		
		return result


def export_index_Q_CD_CI(kappa, x,y, Nx, Ny, r1, r2, W, E_vals, run_nr = 1):
		"""Export the topological index Q = sgn(L(x,y,E)) at the center of the system 
		for the energies specified in E_vals."""

		# create folder
		foldername = "s_l_QSH_index_result/s_l_QSH_index_run_" + str(run_nr)
		
		try:
				os.makedirs(foldername)
		except:
				pass         

		# Build operator matrices
		X = operator_X(Nx, Ny)
		Y = operator_Y(Nx, Ny)
		H = Hamiltonian_DC_CI(Nx, Ny, r1, r2, 0)
		disorder = operator_V_rand(Nx, Ny, W)
				
		# export parameters
		f_parameter = open(foldername + "/parameters.txt", "w")
		
		f_parameter.write("kappa " + str(kappa) + str("\n"))
		f_parameter.write("r1 " + str(r1) + str("\n"))
		f_parameter.write("r2 " + str(r2) + str("\n"))
		f_parameter.write("W " + str(W) + str("\n"))		

		f_parameter.write("Nx " + str(Nx) + str("\n"))
		f_parameter.write("Ny " + str(Ny) + str("\n"))
		f_parameter.write("x " + str(x) + str("\n"))
		f_parameter.write("y " + str(y) + str("\n"))

		f_parameter.write("E_vals " + str(E_vals) + str("\n"))
		
		f_parameter.write("run_nr " + str(run_nr) + str("\n"))
				
		f_parameter.close()

		# calculate and export localizer gaps at reference positions
		n_E = len(E_vals)
		#x = (Nx - 1) / 2
		#y = (Ny - 1) / 2

		result = np.zeros((n_E, 2))

		for j_E in range(n_E):
						E = E_vals[j_E]                        

						localizer = spec_loc(kappa, x, y, E, H, disorder, X, Y)

						eigenvalues = np.linalg.eigvalsh(localizer.todense())     
												
						n_positive = sum(eigenvalues > 0)
						n_negative = sum(eigenvalues < 0)

						Q_E = (n_positive - n_negative)/2

						result[j_E,:] = E, Q_E

						np.savetxt(foldername + "/result.txt", result, delimiter=' ')    
		
		return result


def export_sl_gap_DC_CI_energy(kappa, Nx, Ny, r1, r2, W, n_pos, E_min, E_max, n_E, run_nr = 1, use_sparse = True):
		"""Export the minimal spectral localizer gap of n_pos^2 reference positions in the bulk as a function of energy"""
		# create folder
		foldername = "s_l_DC_CI_energy_result/s_l_DC_CI_energy_run_" + str(run_nr)
		
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

		np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')           

		# export bulk energies of the clean system
		E_bulk = bulk_energies_DC_CI(100, 100, r1, r2)

		np.savetxt(foldername + "/bulk_energies.txt", E_bulk, delimiter=' ')           

		# Build operator matrices
		X = operator_X(Nx, Ny)
		Y = operator_Y(Nx, Ny)
		H = H = Hamiltonian_DC_CI(Nx, Ny, r1, r2, 0)
		disorder = operator_V_rand(Nx, Ny, W)
						
		# export parameters
		f_parameter = open(foldername + "/parameters.txt", "w")
		
		f_parameter.write("kappa " + str(kappa) + str("\n"))
		f_parameter.write("r1 " + str(r1) + str("\n"))
		f_parameter.write("r2 " + str(r2) + str("\n"))
		f_parameter.write("W " + str(W) + str("\n"))

		f_parameter.write("Nx " + str(Nx) + str("\n"))
		f_parameter.write("Ny " + str(Ny) + str("\n"))

		f_parameter.write("E_min " + str(E_min) + str("\n"))
		f_parameter.write("E_max " + str(E_max) + str("\n"))
		f_parameter.write("n_E " + str(n_E) + str("\n"))

		f_parameter.write("n_pos " + str(n_pos) + str("\n"))
		f_parameter.write("run_nr " + str(run_nr) + str("\n"))
				
		f_parameter.close()
		
		# calculate and export minimum of localizer gap in the bulk at the given energies
		result = np.zeros(n_E)
		for j_E in range(n_E):
				E = E_vals[j_E]
				gaps = np.zeros((n_pos, n_pos))

				for jx in range(n_pos):
						for jy in range(n_pos):
								x = X_array[jx, jy]
								y = Y_array[jx, jy]
								
								localizer = spec_loc(kappa, x, y, E, H, disorder, X, Y)

								if use_sparse:
									sigma, ev = scipy.sparse.linalg.eigsh(localizer, k = 1, which = "SM")     
								else:
									sigma = np.linalg.eigvalsh(localizer.todense())
								gaps[jx, jy] = min(np.sort(np.abs(sigma)))
								

				result[j_E] = np.min(gaps)

				np.savetxt(foldername + "/result.txt", result, delimiter=' ')    
												
		return result


def main():        
	Nx = 15
	Ny = 15
	
	r1 = 1.7
	r2 = 1

	W = 0
	
	E = 0.
	kappa = 0.25

	x_min = -1
	x_max = Nx

	y_min = -1
	y_max = Ny

	E_min = -4
	E_max = 4

	n_pos = 5
	n_E = 100
	run_nr = 3

	E_vals = [-4,-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4]
	E_vals = np.linspace(-4, 4, 67)
	


	x = (Nx- 1) / 2
	y = (Ny- 1) / 2

	#export_sl_gap_DC_CI_rs(kappa, Nx, Ny, r1, r2, W, E, x_min, x_max, y_min, y_max, n_pos, run_nr = 1)
	#export_sl_gap_DC_CI_energy(kappa, Nx, Ny, r1, r2, W, n_pos, E_min, E_max, n_E, run_nr, False)
	export_index_Q_CD_CI(kappa, x,y, Nx, Ny, r1, r2, W, E_vals, run_nr)
	
	"""
	Nx = 20
	Ny = 20
	
	r1 = 1.
	r2 = 1.
	
	test_H = Hamiltonian_DC_CI(Nx, Ny, r1, r2,  1.)
	energies = scipy.linalg.eigvalsh(test_H.todense())     
	#print(energies)
	E_bulk = bulk_energies_DC_CI(Nx, Ny, r1, r2)

	diff = np.linalg.norm(energies - E_bulk)

	print("Difference Bloch energies to TB: ", diff)
	"""



if __name__ == '__main__':
	main()