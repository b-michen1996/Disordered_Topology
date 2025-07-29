import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import kwant
from scipy.interpolate import CubicSpline as CS
from  scipy.integrate import nquad as q_int
from matplotlib.patches import PathPatch

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


def plot_spec_cylinder_final_edge_states(N, epsilon_1, epsilon_2, gamma, gamma_2, PBC = 0., wrap_dir = "x", weight = 0.9):
	"""Plot bandstructure for cylinder. Direction is set by wrap_dir. """
	# generate system
	syst = Berry_codes.syst_final(N, N, epsilon_1, epsilon_2, gamma, gamma_2, PBC, wrap_dir)
	
	k_vals = np.linspace(-np.pi, np.pi, N)
	energies = np.zeros((2 * N, N))
	k_vals_edge = list()
	energies_edge = list()
	
	# get momentum eigentstaes for k. filter by proximity to edge
	for j in range(N):
		k = k_vals[j]
		H_k = syst.hamiltonian_submatrix(params = {"k_1":k})
		energies_k, EV_k = np.linalg.eigh(H_k)
		energies[:, j] = energies_k
		
		# check if states are localized at edge
		for l in range(2 * N):
			localization = np.linalg.norm(EV_k[:N, l]) 
			
			if localization > weight:
				k_vals_edge.append(k)
				energies_edge.append(energies_k[l])
				
	# momentum values
	if wrap_dir == "x":
		x_title = r"$k_x$"
	else:
		x_title = r"$k_y$"
			
	title = rf"Cylinder bands for $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma = {gamma}$, $\gamma_2 = {gamma_2}$, $N = {N}$, PBC = {PBC}"

	fig = plt.figure(figsize=(24, 12), layout = "tight")
	a1 =  fig.add_subplot(1,1,1)
	a1.set_title(title , fontsize=30)
	a1.set_xlabel(x_title, fontsize=30)
	a1.set_ylabel("E", fontsize=30)
		
	#kwant.plotter.spectrum(syst, x = ("k_1", k_vals), ax = a1)
	
	for l in range(2 * N):
		a1.plot(k_vals, energies[l,:], color = "b")
	
	a1.scatter(k_vals_edge, energies_edge, color = "r", marker = "o")
	

	plt.show()
		

def plot_energy_berry_curvature_H_final(N, N_colorsteps, epsilon_1, epsilon_2, gamma, gamma_2):
	"""Plot energy of all bands over k_x with a color-code representing the integral
	of the Berry curvature up to the given energy. """
	print("Startting sigma_xy calculation")	
	I_low, I_up, E_low, E_up, energies_band_low  = Berry_codes.sigma_xy_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2)
	
	# get lower and upper bounds for bulk spectrum along kx in both bands
	energy_bounds = np.zeros((N, 2, 2))
	
	for jx in range(N):        		
		E_min_jx_n_0 = np.min(energies_band_low[:, jx])
		E_max_jx_n_0 = np.max(energies_band_low[:, jx])
		E_min_jx_n_1 = -E_max_jx_n_0
		E_max_jx_n_1 = -E_min_jx_n_0
		
		energy_bounds[jx, 0, 0] = E_min_jx_n_0
		energy_bounds[jx, 1, 0] = E_max_jx_n_0
		energy_bounds[jx, 0, 1] = E_min_jx_n_1
		energy_bounds[jx, 1, 1] = E_max_jx_n_1

	
	fig = plt.figure(figsize=(24, 12), layout = "tight")
	a1 =  fig.add_subplot(1,2,1)
	a1.set_xlabel(r"$k_x$", fontsize=30)
	a1.set_ylabel("E", fontsize=30)
	
	# first print rectangle with colormap coming from integral of Berry curvature.
	# We have to interpolate our Berry curvature integral data to even energy spacing to be able to show it with 
	# imshow
	
	interpolation_sigma_xy_low = CS(E_low, I_low)
	interpolation_sigma_xy_up = CS(E_up, I_up)
	
	E_min_low = np.min(E_low)
	E_max_low = np.max(E_low)
	E_min_up = np.min(E_up)
	E_max_up = np.max(E_up)
	
	E_vals_low = np.linspace(E_min_low, E_max_low, N_colorsteps)
	E_vals_up = np.linspace(E_min_up, E_max_up, N_colorsteps)
	
	color_data_low = np.zeros((N_colorsteps,1))
	color_data_up = np.zeros((N_colorsteps, 1))
	
	for j in range(N_colorsteps):
		color_data_low[j] = interpolation_sigma_xy_low(E_vals_low[j])
		color_data_up[j] = interpolation_sigma_xy_up(E_vals_up[j])
		
	print("Starting colorcode")	
	# Show the data as contourplot in imshow
	im_low = a1.imshow(color_data_low,
			   aspect='auto',
			   origin='lower',
			   extent=[-np.pi, np.pi, E_min_low, E_max_low],
			   vmin = 0,
			   vmax = 1,
			   cmap=plt.cm.viridis
			  )
	
	im_up = a1.imshow(color_data_up,
			   aspect='auto',
			   origin='lower',
			   extent=[-np.pi, np.pi, E_min_up, E_max_up],
			   vmin = 0,
			   vmax = 1,
			   cmap=plt.cm.magma_r
			  )
	
	# clip by band boundaries
	kx_vals = np.linspace(-np.pi, np.pi, N, endpoint= False)
	paths_low = a1.fill_between(kx_vals, energy_bounds[:, 0, 0], energy_bounds[:, 1, 0], facecolor = "none", lw = 0)
	paths_up = a1.fill_between(kx_vals, energy_bounds[:, 0, 1], energy_bounds[:, 1, 1], facecolor = "none", lw = 0)

	patch_low = PathPatch(paths_low._paths[0], visible=False)
	a1.add_artist(patch_low)
	im_low.set_clip_path(patch_low)
	
	patch_up = PathPatch(paths_up._paths[0], visible=False)
	a1.add_artist(patch_up)
	im_up.set_clip_path(patch_up)

	cbar_title = r"$\sigma_{xy}[e^2/\hbar]$"
	cbar = plt.colorbar(im_low)
	cbar.ax.set_title(cbar_title, fontsize = 20) 
	
	plt.show()


def plot_berry_curvature_final_integral(N, epsilon_1, epsilon_2, gamma, gamma_2 = 1):
	"""Integral of Berry curvature up to energy. """
	
	data = Berry_codes.Berry_curvature_integral_final(N, epsilon_1, epsilon_2, gamma, gamma_2)

	# add together phi_vals for opposing k_y as they have the same energy. A grid spaced with an even number 
	# of sites ensures that there is a pair for all ky 

	n_bands = len(data[0,0,:])

	title = rf"Berry curvature contained in Fermi sphere for $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma = {gamma}$, $\gamma_2 = {gamma_2}$"

	fig = plt.figure(figsize=(24, 12), layout = "tight")
	a1 =  fig.add_subplot(1,1,1)
	a1.set_title(title , fontsize=30)
	a1.set_xlabel(r"$E_F$", fontsize=30)
	a1.set_ylabel(r"$\int \Omega$", fontsize=30)
	
	plt.axhline(y=0, color='black', linestyle='-')
	plt.axhline(y=1, color='black', linestyle='--')

	for j in range(n_bands):
		a1.plot(data[:,1,j], data[:,0,j], marker = "x", label = rf"Band {j+1}")
		
	a1.legend(loc="upper left")
	
	#fig.tight_layout()
	#fig.savefig(f"Curvature_energy_DC_CI_r1_{r1}_r_2_{r2}.png",  bbox_inches='tight', dpi=300)

	plt.show()
	

def plot_berry_curvature_final_V3_integral(N, r, epsilon_1, epsilon_2, gamma, gamma_2 = 1):
	"""Integral of Berry curvature up to energy. """
	
	I_low, I_up, E_low, E_up, energies_band_low, energies_band_up  = Berry_codes.sigma_xy_V3_analytically(N, r, epsilon_1, epsilon_2, gamma, gamma_2)
	data = Berry_codes.Berry_curvature_integral_final_V3(N, r, epsilon_1, epsilon_2, gamma, gamma_2)
	
	# add together phi_vals for opposing k_y as they have the same energy. A grid spaced with an even number 
	# of sites ensures that there is a pair for all ky 

	
	title = rf"Berry curvature contained in Fermi sphere for $r = {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma = {gamma}$, $\gamma_2 = {gamma_2}$"

	fig = plt.figure(figsize=(24, 12), layout = "tight")
	a1 =  fig.add_subplot(1,1,1)
	a1.set_title(title , fontsize=30)
	a1.set_xlabel(r"$E_F$", fontsize=30)
	a1.set_ylabel(r"$\int \Omega$", fontsize=30)
	
	plt.axhline(y=0, color='black', linestyle='-')
	plt.axhline(y=1, color='black', linestyle='--')

	a1.plot(E_low, I_low, color = "blue", label = "analytical")
	a1.plot(E_up, I_up, color = "blue")
	
	a1.plot(data[:,1,0], data[:,0,0], label = "numerical", color = "red")
	a1.plot(data[:,1,1], data[:,0,1], color = "red")
		
	a1.legend(loc="upper left")
	
	#fig.tight_layout()
	#fig.savefig(f"Curvature_energy_DC_CI_r1_{r1}_r_2_{r2}.png",  bbox_inches='tight', dpi=300)

	plt.show()
	


def plot_berry_curvature_CI_integral(N, r):
	"""Integral of Berry curvature up to energy. """
	
	data = Berry_codes.Berry_curvature_integral_CI(N, r)

	# add together phi_vals for opposing k_y as they have the same energy. A grid spaced with an even number 
	# of sites ensures that there is a pair for all ky 

	n_bands = len(data[0,0,:])

	title = rf"Berry curvature contained in Fermi sphere for $N$ = {N}, $r$ = {r}"

	fig = plt.figure(figsize=(24, 12), layout = "tight")
	a1 =  fig.add_subplot(1,1,1)
	a1.set_title(title , fontsize=30)
	a1.set_xlabel(r"$E_F$", fontsize=30)
	a1.set_ylabel(r"$\int \Omega$", fontsize=30)
	
	plt.axhline(y=0, color='black', linestyle='-')
	plt.axhline(y=1, color='black', linestyle='--')

	for j in range(n_bands):
		a1.plot(data[:,1,j], data[:,0,j], marker = "x", label = rf"Band {j+1}")
		
	a1.legend(loc="upper left")
	
	#fig.tight_layout()
	#fig.savefig(f"Curvature_energy_DC_CI_r1_{r1}_r_2_{r2}.png",  bbox_inches='tight', dpi=300)

	plt.show()
	

def plot_integral_Berry_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2, test_integral = False, test_Evals = None):
	"""Plot Integral of Berry curvature up to E_F"""

	final_integral_lower, final_integral_upper, final_energies_lower, final_energies_upper, energies_band_low = Berry_codes.sigma_xy_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2)
		
	fig = plt.figure(figsize=(24, 12), layout = "tight")
	a1 =  fig.add_subplot(1,2,1)
	a1.set_xlabel(r"$E_F$", fontsize=30)
	a1.set_ylabel(r"$\sigma_{xy}$", fontsize=30)
	
	a1.plot(final_energies_lower, final_integral_lower)
	a1.plot(final_energies_upper, final_integral_upper)

	if test_integral:
		n_vals = len(test_Evals)
		test_data = np.zeros(n_vals)

		for j in range(n_vals):
			E_F = test_Evals[j]
			test_data[j] = Berry_codes.Integrate_Berry_analytically(epsilon_1, epsilon_2, gamma, gamma_2, E_F, band = -1)
			
		a1.scatter(test_Evals, test_data, marker = "x")		
	
	plt.show()
	

def main():
	Nx= 200
	Ny= 200

	r_1 = 1.7
	r_2 = 1.
	
	gamma = 2
	v = 0.
	
	PBC = 0

	wrap_dir = "y"
	
	#plot_spec_cylinder_DC_CI_3B(Nx, Ny, r_1, r_2, gamma, v, PBC, wrap_dir)
	
	Nx= 100
	Ny= 100
	
	N = 100
	N_colorsteps = 100
	
	
	r = 1.5
	epsilon_1 = -0.3
	epsilon_2 = 2

	gamma = 2
	gamma_2 = 0.5
	gamma_3 = 0

	PBC = 0

	wrap_dir = "x"

	#plot_spec_cylinder_final_edge_states(N, epsilon_1, epsilon_2, gamma, gamma_2, PBC, wrap_dir)
	#plot_energy_berry_curvature_H_final(N, N_colorsteps, epsilon_1, epsilon_2, gamma, gamma_2)

	
	
	N_E = 100
	E_F = -1
	#test = Integrate_Berry_analytically(epsilon_1, epsilon_2, gamma, gamma_2, E_F, band = -1)	
	#print(test)
	
	N = 100
	#plot_integral_Berry_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2, test_integral = True, test_Evals = [-2, -1, 1, 2])
	#plot_integral_Berry_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2)
	#plot_berry_curvature_final_integral(N, epsilon_1, epsilon_2, gamma, gamma_2)	
	
	kx = np.pi
	ky = 0 
	
	#test = Berry_codes.Omega_func(kx, ky, epsilon_1, epsilon_2, gamma, gamma_2, band = -1)
	
	#print(test)
	
	r = 1
	N = 100
	
	plot_berry_curvature_CI_integral(N, r)
	

	N = 200
	r = 1.5
	epsilon_1 = 0.3
	epsilon_2 = 2

	gamma = 2
	gamma_2 = 0.3
	
	#plot_berry_curvature_final_V3_integral(N, r, epsilon_1, epsilon_2, gamma, gamma_2)

	
	
	
	
	
if __name__ == '__main__':
	main()
