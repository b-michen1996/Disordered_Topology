import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import kwant
from scipy.interpolate import CubicSpline as CS
from  scipy.integrate import nquad as q_int

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


def H_final(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2):
	"""Bloch Hamiltonian that exhibits a dipole-like charge separation of topological charge"""
	
	lambda_kx = epsilon_1 + epsilon_2 * (1 - np.cos(kx))/2
	
	h_x = gamma * np.sin(kx) + gamma_2 * np.cos(ky) 
	h_y = lambda_kx * np.sin(ky) 
	h_z = - lambda_kx * np.cos(ky) 

	return h_x * sigma_x + h_y * sigma_y + h_z * sigma_z


def H_CI(kx, ky, r):
	"""Bloch Hamiltonian that exhibits a dipole-like charge separation of topological charge"""
	
	
	h_x = np.sin(kx) 
	h_y = np.sin(ky) 
	h_z = r - np.cos(kx) - np.cos(ky) 

	return h_x * sigma_x + h_y * sigma_y + h_z * sigma_z

	
def Berry_curvature(N, H_bloch):
	"""Calculate berry curvature of H_bloch and return it as an array along with k_x, k_y values suitable for contour plot."""
	delta_k = 2 * np.pi / N 
	k_vals = np.linspace(-np.pi + delta_k/2, np.pi + delta_k/2, N, endpoint= False)
	
	kx_array, ky_array = np.meshgrid(k_vals,k_vals)
	
	n_band = H_bloch(0,0).shape[0]

	energies =  np.zeros((N,N,n_band))
	ES = 1j * np.zeros((N,N,n_band, n_band))

	Phi_vals =  np.zeros((N,N,n_band))
	
	# calculate all eigenstates
	for jx in range(N):        
		for jy in range(N):            
			kx = kx_array[jy,jx] - delta_k/2
			ky = ky_array[jy,jx] - delta_k/2

			H_k = H_bloch(kx, ky)

			E_k, EV_k = np.linalg.eigh(H_k)

			energies[jy, jx, :] = E_k
			ES[jy, jx, :, :] = EV_k     


	# calculate Berry curvature
	for jx in range(N):
		jx_p1 = (jx + 1) % N
		for jy in range(N):
			jy_p1 = (jy + 1) % N

			p_vals = [(jy,jx), (jy, jx_p1), (jy_p1, jx_p1), (jy_p1, jx)]
			
			# generate array of products for each band
			prod_arrray = np.einsum("jk,jk->k", np.conj(ES[p_vals[0]]), ES[p_vals[1]])

			for l in range(1, 4):
				l_p1 = (l+1)%4
				prod_arrray = prod_arrray * np.einsum("jk,jk->k", np.conj(ES[p_vals[l]]), ES[p_vals[l_p1]])

			# calculate flux through plaquette
			Phi_vals[jy, jx, :] = -np.angle(prod_arrray)
	
	# calculate chern number for each band
	C_n = (1 / (2 * np.pi)) * np.sum(Phi_vals, axis = (0,1)) 

	# give back results and arrays of k_vals
	return ((N / (2 * np.pi)) ** 2) * Phi_vals, C_n, kx_array, ky_array, energies

	
def Berry_curvature_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2):
	"""Calculate berry curvature of H_bloch and return it as an array along with k_x, k_y values suitable for contour plot."""
	delta_k = 2 * np.pi / N 
	k_vals = np.linspace(-np.pi + delta_k/2, np.pi + delta_k/2, N, endpoint= False)
	
	kx_array, ky_array = np.meshgrid(k_vals,k_vals)

	Phi_vals =  np.zeros((N,N))
	
	energies = np.zeros((N,N))
	
	# calculate Berry curvature
	for jx in range(N):
		for jy in range(N):
			kx = kx_array[jy, jx]
			ky = ky_array[jy, jx]
			# calculate flux through plaquette
			Omega, energy = Omega_func(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2, band = -1)
			Phi_vals[jy, jx] = Omega
			energies[jy, jx] = energy						

	H_bloch = lambda kx, ky: H_final(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2)
	
	for jx in range(N):        
		for jy in range(N):            
			kx = kx_array[jy,jx] 
			ky = ky_array[jy,jx] 

			H_k = H_bloch(kx, ky)

			E_k = np.linalg.eigvalsh(H_k)

			energies[jy, jx] = E_k[0]
		
	# calculate chern number for each band
	C_n = (2 * np.pi / (N ** 2)) * np.sum(Phi_vals) 

	# give back results and arrays of k_vals
	return Phi_vals, C_n, kx_array, ky_array, energies

	

def Berry_curvature_analytically_V3(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3 = 0.):
	"""Calculate berry curvature of H_bloch and return it as an array along with k_x, k_y values suitable for contour plot."""
	delta_k = 2 * np.pi / N 
	k_vals = np.linspace(-np.pi + delta_k/2, np.pi + delta_k/2, N, endpoint= False)
	
	kx_array, ky_array = np.meshgrid(k_vals,k_vals)

	Phi_vals =  np.zeros((N,N))
	
	energies = np.zeros((N,N,2))
	
	# calculate Berry curvature
	for jx in range(N):
		for jy in range(N):
			kx = kx_array[jy, jx]
			ky = ky_array[jy, jx]
			# calculate flux through plaquette
			Omega, energy_m, energy_p = Omega_func_V3(kx, ky, r, epsilon_1, epsilon_2, gamma, gamma_2, band = -1, gamma_3 = gamma_3)
			Phi_vals[jy, jx] = Omega
			energies[jy, jx, 0] = energy_m
			energies[jy, jx, 1] = energy_p
		
	# calculate chern number for each band
	C_n = (2 * np.pi / (N ** 2)) * np.sum(Phi_vals) 

	# give back results and arrays of k_vals
	return Phi_vals, C_n, kx_array, ky_array, energies


def Integrate_Berry_analytically(epsilon_1, epsilon_2, gamma, gamma_2, E_F, band = -1): 
	"""Calculate itegral of Berry Curvature up to energy E."""
	Integrand = lambda kx, ky : Omega_func_EF(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2, E_F, band)
	
	res, err = q_int(Integrand, [[-np.pi, np.pi], [-np.pi, np.pi]], opts = {"limit" : 100})
	
	return res


def Omega_func_EF(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2, E_F, band = -1):
	"""calculate Berry curvature at kx, ky multiplied by Theta(E-E_F)"""
	s_x = np.sin(kx)
	c_x = np.cos(kx)
	c_y = np.cos(ky)	
	lambda_kx = (epsilon_1 + epsilon_2 * (1 - c_x)/2)
				
	E = band * np.sqrt(lambda_kx**2 + (gamma * s_x + gamma_2 * c_y)**2)
	
	res = lambda_kx * (epsilon_2 * gamma * s_x ** 2 / 2 + epsilon_2 * gamma_2 * s_x * c_y / 2 - gamma * lambda_kx * c_x)
	
	if E > E_F:
		return 0
	else:
		res = - band * res / (4 * np.pi * (E ** 3))
		return res
	

def Omega_func(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2, band = -1):
	"""calculate Berry curvature at kx, ky"""
	s_x = np.sin(kx)
	c_x = np.cos(kx)
	c_y = np.cos(ky)	
	lambda_kx = (epsilon_1 + epsilon_2 * (1 - c_x)/2)
				
	E = band * np.sqrt(lambda_kx**2 + (gamma * s_x + gamma_2 * c_y)**2)
	
	res = lambda_kx * (epsilon_2 * gamma * s_x ** 2 / 2 + epsilon_2 * gamma_2 * s_x * c_y / 2 - gamma * lambda_kx * c_x)
	
	res = - band * res / (2 * (E ** 3))
	
	return res, E



def Omega_func_V3(kx, ky, r, epsilon_1, epsilon_2, gamma, gamma_2, band = -1, gamma_3 = 0):
	"""calculate Berry curvature at kx, ky"""
	s_x = np.sin(kx)
	s_2x = np.sin(2 * kx)
	s_y = np.sin(ky)
	c_x = np.cos(kx)
	c_2x = np.cos(2 * kx)
	c_y = np.cos(ky)	
	lambda_kx = (epsilon_1 + epsilon_2 * (1 - c_x)/2)
				
	E = np.sqrt((gamma * s_x) ** 2 + (lambda_kx * s_y)**2 + (gamma_2 * (r - c_2x) - lambda_kx * c_y) ** 2)
	
	res = lambda_kx * (epsilon_2 * gamma * s_x ** 2 / 2 + 2 * gamma * gamma_2 * s_x * s_2x * c_y 
					+ gamma * gamma_2 * (r - c_2x) * c_x * c_y - gamma * lambda_kx * c_x)
	
	res = - band * res / (2 * (E ** 3))
	
	d0 = gamma_3 * np.cos(ky)
	
	return res, d0 - E, d0 + E

def syst_final(Nx, Ny, epsilon_1, epsilon_2, gamma, gamma_2, PBC = 0., wrap_dir = "x"):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
	# values for lattice constant and number of orbitals
	a = 1
				
	lat = kwant.lattice.square(a, norbs = 3)

	# set translation symmetry
	sym = kwant.TranslationalSymmetry((a, 0))
	if wrap_dir == "y":
		sym = kwant.TranslationalSymmetry((0, a))
		
	if wrap_dir == "xy":
		sym = kwant.TranslationalSymmetry([a, 0], [0, a])
		
	syst = kwant.Builder(sym)

	mat_os = 0
	hop_mat_dx = gamma/(2j) * sigma_x 
	hop_mat_dy = ((epsilon_1 + epsilon_2/2)/(2j)) * sigma_y - ((epsilon_1 + epsilon_2/2) / 2) * sigma_z + (gamma_2 /2) * sigma_x 
	hop_mat_dx_dy = -(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z
	hop_mat_dx_mdy = +(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z
	
	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy					
					syst[lat(jx,jy)] = mat_os
	
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_mat_dx
	# Hopping in y-direction
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hop_mat_dy

	# Hopping in (x+y)-direction
	syst[kwant.builder.HoppingKind((1, 1), lat, lat)] = hop_mat_dx_dy
	
	# Hopping in (x-y)-direction
	syst[kwant.builder.HoppingKind((1, -1), lat, lat)] = hop_mat_dx_mdy
	
	if wrap_dir == "x":
		# Set boundary terms along y-direction
		syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = PBC * hop_mat_dy
		
		# Set boundary terms along (x+y)-direction
		syst[kwant.builder.HoppingKind((1, -(Ny-1)), lat, lat)] = PBC * hop_mat_dx_dy
		
		# Set boundary terms along (x-y)-direction
		syst[kwant.builder.HoppingKind((1, (Ny-1)), lat, lat)] = PBC * hop_mat_dx_mdy
		
	elif wrap_dir == "y":
		# Set boundary terms along x-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = PBC * hop_mat_dx
		
		# Set boundary terms along (x+y)-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), 1), lat, lat)] = PBC * hop_mat_dx_dy
		
		# Set boundary terms along (x-y)-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), -1), lat, lat)] = PBC * hop_mat_dx_mdy

	# wrap system
	if wrap_dir == "xy":		
		syst = kwant.wraparound.wraparound(syst, coordinate_names=('1', '2'))
	else:
		syst = kwant.wraparound.wraparound(syst, coordinate_names='1')
	
	
	return syst.finalized()


def Berry_curvature_integral_final(N, epsilon_1, epsilon_2, gamma, gamma_2 = 1):
	"""Integral of Berry curvature up to Fermi energy"""
	H_bloch = lambda kx, ky: H_final(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2)
	Phi_vals, C_n, k_x_vals, k_y_vals, energies = Berry_curvature(N, H_bloch)
	
	print("Min / Max energies")
	print(np.min(energies))
	print(np.max(energies))
	
	n_band = H_bloch(0,0).shape[0]
	
	data = np.zeros((N ** 2, 2, n_band))
	
	# Flatten data for Berry curvature and energy and multiply by prefactor  delta_k^2 / (2 pi)
	prefactor = (2 * np.pi / N) ** 2
	
	for j_band in range(n_band):
		# get list of Berry curvatures values and energies in each cell
		Berry_curvature_list = prefactor * Phi_vals[:,:,j_band].flatten()
		Energy_list = energies[:,:,j_band].flatten()
		
		# get index to sort by energy
		idx = np.argsort(Energy_list)
		
		# save to data
		data[:,0,j_band] = Berry_curvature_list[idx]
		data[:,1,j_band] = Energy_list[idx]
		
	# Integrate
	for j in range(N**2):
		try:
			data[j, 0, :] += data[j-1, 0, :]
		except:
			pass
	
	
	data[:,0,:] = (1 / (2 * np.pi)) * data[:,0,:]
	
	return data#, energies



def Berry_curvature_integral_CI(N, r):
	"""Integral of Berry curvature up to Fermi energy"""
	H_bloch = lambda kx, ky: H_CI(kx, ky, r)
	Phi_vals, C_n, k_x_vals, k_y_vals, energies = Berry_curvature(N, H_bloch)
	
	print("Min / Max energies")
	print(np.min(energies))
	print(np.max(energies))
	
	n_band = H_bloch(0,0).shape[0]
	
	data = np.zeros((N ** 2, 2, n_band))
	
	# Flatten data for Berry curvature and energy and multiply by prefactor  delta_k^2 / (2 pi)
	prefactor = (2 * np.pi / N) ** 2
	
	for j_band in range(n_band):
		# get list of Berry curvatures values and energies in each cell
		Berry_curvature_list = prefactor * Phi_vals[:,:,j_band].flatten()
		Energy_list = energies[:,:,j_band].flatten()
		
		# get index to sort by energy
		idx = np.argsort(Energy_list)
		
		# save to data
		data[:,0,j_band] = Berry_curvature_list[idx]
		data[:,1,j_band] = Energy_list[idx]
		
	# Integrate
	for j in range(N**2):
		try:
			data[j, 0, :] += data[j-1, 0, :]
		except:
			pass
	
	
	data[:,0,:] = (1 / (2 * np.pi)) * data[:,0,:]
	
	return data#, energies


def sigma_xy_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2):
	"""Integral of Berry curvature up to Fermi energy, obtained from analytical Berry curvature discretized on N x N lattice."""
	Omega_vals_lower_band, C_n, k_x_vals, k_y_vals, energies_lower_band = Berry_curvature_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2)

	print("Min / Max energies")
	print(np.min(energies_lower_band))
	print(np.max(energies_lower_band))
	
	print ("Chern numbers ", C_n)	
		

	# Flatten data for Berry curvature and energy and multiply by prefactor  delta_k^2 / (2 pi)
	prefactor = 2 * np.pi / (N ** 2)
	 
	Integration_data_lower = prefactor * Omega_vals_lower_band.flatten()
	energy_data_lower = energies_lower_band.flatten()
	Integration_data_upper = - Integration_data_lower
	energy_data_upper = - energy_data_lower
	
	# get index to sort by energy
	idx_lower = np.argsort(energy_data_lower)
	idx_upper = np.argsort(energy_data_upper)
	
	Integration_data_lower = Integration_data_lower[idx_lower]
	energy_data_lower = energy_data_lower[idx_lower]
	Integration_data_upper = Integration_data_upper[idx_upper]
	energy_data_upper = energy_data_upper[idx_upper]
	
	# Integrate
	for j in range(1, N**2):
		Integration_data_lower[j] += Integration_data_lower[j-1]
		Integration_data_upper[j] += Integration_data_upper[j-1]
			
	# Some values of E occur multiple times, throw them out
	final_integral_lower = [Integration_data_lower[0]]
	final_energies_lower = [energy_data_lower[0]]
	
	for j in range(1, N**2):
		if (energy_data_lower[j] -  final_energies_lower[-1]) < 10 ** (-8):
			final_integral_lower[-1] = Integration_data_lower[j]
			final_energies_lower[-1] = energy_data_lower[j]
		else:
			final_integral_lower.append(Integration_data_lower[j])
			final_energies_lower.append(energy_data_lower[j])
			
	final_integral_upper = [Integration_data_upper[0]]
	final_energies_upper = [energy_data_upper[0]]
	
	for j in range(1, N**2):
		if (energy_data_upper[j] -  final_energies_upper[-1]) < 10 ** (-8):
			final_integral_upper[-1] = Integration_data_upper[j]
			final_energies_upper[-1] = energy_data_upper[j]
		else:
			final_integral_upper.append(Integration_data_upper[j])
			final_energies_upper.append(energy_data_upper[j])
	
		
	return final_integral_lower, final_integral_upper, final_energies_lower, final_energies_upper, energies_lower_band



def sigma_xy_V3_analytically(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3 = 0):
	"""Integral of Berry curvature up to Fermi energy, obtained from analytical Berry curvature discretized on N x N lattice."""
	Omega_vals_lower_band, C_n, k_x_vals, k_y_vals, energies = Berry_curvature_analytically_V3(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3)

	print("Min / Max energies lower band")
	print(np.min(energies[:,:,0]))
	print(np.max(energies[:,:,0]))
	
	print("Min / Max energies upper band")
	print(np.min(energies[:,:,1]))
	print(np.max(energies[:,:,1]))
	
	print ("Chern numbers ", C_n)	
		

	# Flatten data for Berry curvature and energy and multiply by prefactor  delta_k^2 / (2 pi)
	prefactor = 2 * np.pi / (N ** 2)
	 
	Integration_data_lower = prefactor * Omega_vals_lower_band.flatten()
	energy_data_lower = energies[:,:,0].flatten()
	Integration_data_upper = - Integration_data_lower
	energy_data_upper = energies[:,:,1].flatten()
	
	# get index to sort by energy
	idx_lower = np.argsort(energy_data_lower)
	idx_upper = np.argsort(energy_data_upper)
	
	Integration_data_lower = Integration_data_lower[idx_lower]
	energy_data_lower = energy_data_lower[idx_lower]
	Integration_data_upper = Integration_data_upper[idx_upper]
	energy_data_upper = energy_data_upper[idx_upper]
	
	# Integrate
	for j in range(1, N**2):
		Integration_data_lower[j] += Integration_data_lower[j-1]
		Integration_data_upper[j] += Integration_data_upper[j-1]
			
	# Some values of E occur multiple times, throw them out
	final_integral_lower = [Integration_data_lower[0]]
	final_energies_lower = [energy_data_lower[0]]
	
	for j in range(1, N**2):
		if (energy_data_lower[j] -  final_energies_lower[-1]) < 10 ** (-8):
			final_integral_lower[-1] = Integration_data_lower[j]
			final_energies_lower[-1] = energy_data_lower[j]
		else:
			final_integral_lower.append(Integration_data_lower[j])
			final_energies_lower.append(energy_data_lower[j])
			
	final_integral_upper = [Integration_data_upper[0]]
	final_energies_upper = [energy_data_upper[0]]
	
	for j in range(1, N**2):
		if (energy_data_upper[j] -  final_energies_upper[-1]) < 10 ** (-8):
			final_integral_upper[-1] = Integration_data_upper[j]
			final_energies_upper[-1] = energy_data_upper[j]
		else:
			final_integral_upper.append(Integration_data_upper[j])
			final_energies_upper.append(energy_data_upper[j])
	
		
	return final_integral_lower, final_integral_upper, final_energies_lower, final_energies_upper, energies[:,:,0], energies[:,:,1]

	
