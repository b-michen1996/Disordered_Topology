import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors


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
	
	res = lambda_kx * (epsilon_2 * gamma * s_x ** 2 / 2 - 2 * gamma * gamma_2 * s_x * s_2x * c_y 
					+ gamma * gamma_2 * (r - c_2x) * c_x * c_y - gamma * lambda_kx * c_x )
	
	res = - band * res / (2 * (E ** 3))
	
	d0 = gamma_3 * np.cos(ky)
	
	return res, d0 - E, d0 + E


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

	
