import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import kwant
from scipy.interpolate import CubicSpline as CS
from  scipy.integrate import nquad as q_int

import Berry_codes

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


def contour_plot_berry_curvature_numerical(N, epsilon_1, epsilon_2, gamma, gamma_2 = 1, n = 0):
	"""Do contour plot for Berry curvature"""
	
	H_bloch = lambda kx, ky: Berry_codes.H_final(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2)
	
	Phi_vals, C_n, k_x_vals, k_y_vals, energies = Berry_codes.Berry_curvature(N, H_bloch)

	print("Chern numbers: ", C_n)

	a_1_title = rf"Berry curvature for band {n}, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma = {gamma}$, $\gamma_2 = {gamma_2}$"
	a_2_title = rf"Energy of band {n}"

	fig = plt.figure(figsize=(6, 6), layout = "tight")
	a1 =  fig.add_subplot(1,2,1)
	
	a1.set_title(a_1_title , fontsize=30)
	a1.set_xlabel(r"$k_x$", fontsize=30)
	a1.set_ylabel(r"$k_y$", fontsize=30)
	a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   

	a2 =  fig.add_subplot(1,2,2)
	a2.set_title(a_2_title , fontsize=30)
	a2.set_xlabel(r"$k_x$", fontsize=30)
	a2.set_ylabel(r"$k_y$", fontsize=30)
	a2.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)

	levels = np.linspace(-3600, 3600, 500)
	contour_plot_curvature = a1.contourf(k_x_vals, k_y_vals, (2 * N * np.pi)**2 * Phi_vals[:,:, n], levels = levels, extend = "both", cmap=plt.cm.magma_r)
	contour_plot_energy = a2.contourf(k_x_vals, k_y_vals, energies[:,:, n], levels = 100, extend = "both", cmap=plt.cm.viridis)

	cbar_1 = plt.colorbar(contour_plot_curvature, ax = a1, label =r"$\Omega$")    
	cbar_2 = plt.colorbar(contour_plot_energy, ax = a2, label = r"$E(k_x, k_y)$")

	
	plt.show()

	return


def contour_plot_berry_curvature_analytical(N, epsilon_1, epsilon_2, gamma, gamma_2 = 1, n = 0):
	"""Do contour plot for Berry curvature"""
	
	H_bloch = lambda kx, ky: Berry_codes.H_final(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2)
	
	Phi_vals, C_n, k_x_vals, k_y_vals = Berry_codes.Berry_curvature_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2)

	print("Chern numbers: ", C_n)

	a_1_title = rf"Berry curvature for band {n}, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma = {gamma}$, $\gamma_2 = {gamma_2}$"
	a_2_title = rf"Energy of band {n}"

	fig = plt.figure(figsize=(6, 6), layout = "tight")
	a1 =  fig.add_subplot(1,1,1)
	
	a1.set_title(a_1_title , fontsize=30)
	a1.set_xlabel(r"$k_x$", fontsize=30)
	a1.set_ylabel(r"$k_y$", fontsize=30)
	a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   

	levels = np.linspace(-3600, 3600, 500)
	contour_plot_curvature = a1.contourf(k_x_vals, k_y_vals, Phi_vals[:,:, n], levels = levels, extend = "both", cmap=plt.cm.magma_r)

	cbar_1 = plt.colorbar(contour_plot_curvature, ax = a1, label =r"$\Omega$")    

	
	plt.show()

	return



def main():
	Nx= 200
	Ny= 200

	r_1 = 1.7
	r_2 = 1.
	
	gamma = 2
	v = 0.
	
	PBC = 0

	wrap_dir = "x"
	
	
	Nx= 100
	Ny= 100
	N = 100
	
	r = 1.5
	epsilon_1 = -0.3
	epsilon_2 = 4

	gamma = 2
	gamma_2 = 0.2
	gamma_3 = 0

	PBC = 0

	wrap_dir = "x"
	
	N_E = 10
	E_F = -1
	#test = Integrate_Berry_analytically(epsilon_1, epsilon_2, gamma, gamma_2, E_F, band = -1)	
	#print(test)
	#plot_integral_Berry_analytically(N_E, epsilon_1, epsilon_2, gamma, gamma_2)
	
	epsilon_1 = 0.3
	epsilon_2 = 2

	gamma = 2
	gamma_2 = 0.2
	

	#contour_plot_berry_curvature_numerical(N, epsilon_1, epsilon_2, gamma, gamma_2, n = 0)
	#contour_plot_berry_curvature_analytical(N, epsilon_1, epsilon_2, gamma, gamma_2, n = 0)
	
	H_bloch = lambda kx, ky: Berry_codes.H_final(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2)
	Phi_vals_test, C_n, k_x_vals, k_y_vals, energies = Berry_codes.Berry_curvature(N, H_bloch)
	Phi_vals, C_n, k_x_vals, k_y_vals = Berry_codes.Berry_curvature_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2)
	
	for jx in range(N):
		for jy in range(N):
			print(Phi_vals_test[jy,jx], Phi_vals[jy,jx])
		
	

	
	
	
	
	
if __name__ == '__main__':
	main()
