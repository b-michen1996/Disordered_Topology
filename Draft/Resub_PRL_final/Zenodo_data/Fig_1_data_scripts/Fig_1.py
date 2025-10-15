import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline as CS
from matplotlib.patches import PathPatch
from matplotlib.colors import ListedColormap


# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "times"
})


def sigma_xy_clean(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3 = 0):
	"""Integral of Berry curvature up to Fermi energy, obtained from analytical Berry curvature discretized on N x N lattice."""
	Omega_vals_lower_band, C_n, k_x_vals, k_y_vals, energies = Berry_curvature(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3)
	
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


def Berry_curvature(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3 = 0.):
	"""Calculate berry curvature of H_bloch on a discrete grid of k_x, k_y values. """
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
			Phi, energy_m, energy_p = Omega(kx, ky, r, epsilon_1, epsilon_2, gamma, gamma_2, band = -1, gamma_3 = gamma_3)
			Phi_vals[jy, jx] = Phi
			energies[jy, jx, 0] = energy_m
			energies[jy, jx, 1] = energy_p
		
	# calculate chern number for each band
	C_n = (2 * np.pi / (N ** 2)) * np.sum(Phi_vals) 

	# give back results and arrays of k_vals
	return Phi_vals, C_n, kx_array, ky_array, energies


def Omega(kx, ky, r, epsilon_1, epsilon_2, gamma, gamma_2, band = -1, gamma_3 = 0):
	"""Calculate Berry curvature at kx, ky"""
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


def Fig_1(N, N_colorsteps, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3, data_location, data_location_inset):
	"""Plot energy of all bands over k_x with a color-code representing the integral
	of the Berry curvature up to the given energy. """
	print("Startting sigma_xy calculation")
	I_low, I_up, E_low, E_up, energies_band_low, energies_band_up  = sigma_xy_clean(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3)
	
	# get lower and upper bounds for bulk spectrum along kx in both bands
	energy_bounds = np.zeros((N, 2, 2))
	
	for jx in range(N):        		
		E_min_jx_n_0 = np.min(energies_band_low[:, jx])
		E_max_jx_n_0 = np.max(energies_band_low[:, jx])
		E_min_jx_n_1 = np.min(energies_band_up[:, jx])
		E_max_jx_n_1 = np.max(energies_band_up[:, jx])
		
		energy_bounds[jx, 0, 0] = E_min_jx_n_0
		energy_bounds[jx, 1, 0] = E_max_jx_n_0
		energy_bounds[jx, 0, 1] = E_min_jx_n_1
		energy_bounds[jx, 1, 1] = E_max_jx_n_1
	
	E_min_low = np.min(E_low)
	E_max_low = np.max(E_low)
	E_min_up = np.min(E_up)
	E_max_up = np.max(E_up)
	
	label_size = 25
	label_scale_inset = 0.75
	
	fig, axs = plt.subplots(1, 2, figsize =(12, 6))
	fig.subplots_adjust(wspace=0.1)
	a1 = axs[0]
	a2 = axs[1]
		
	# first print rectangle with colormap coming from integral of Berry curvature.
	# We have to interpolate our Berry curvature integral data to even energy spacing to be able to show it with 
	# imshow
	
	interpolation_sigma_xy_low = CS(E_low, I_low)
	interpolation_sigma_xy_up = CS(E_up, I_up)
	
	E_vals_low = np.linspace(E_min_low, E_max_low, N_colorsteps)
	E_vals_up = np.linspace(E_min_up, E_max_up, N_colorsteps)
	
	color_data_low = np.zeros((N_colorsteps,1))
	color_data_up = np.zeros((N_colorsteps, 1))
		
	for j in range(N_colorsteps):
		color_data_low[j] = interpolation_sigma_xy_low(E_vals_low[j])
		color_data_up[j] = interpolation_sigma_xy_up(E_vals_up[j])

	color_max = max(max(color_data_low), max(color_data_up))
			
	print("Starting colorcode")	
	magmga_map= plt.get_cmap("magma_r", 1000) 
	cmap_magma_reduced = ListedColormap(magmga_map(np.linspace(0, 0.5 * color_max, 500)))
	
	# Show the data as contourplot in imshow
	im_low = a1.imshow(color_data_low,
			   aspect='auto',
			   origin='lower',
			   extent=[-np.pi, np.pi, E_min_low, E_max_low],
			   vmin = 0, vmax = color_max,
			   cmap=cmap_magma_reduced
			  )
	
	im_up = a1.imshow(color_data_up,
			   aspect='auto',
			   origin='lower',
			   extent=[-np.pi, np.pi, E_min_up, E_max_up],
			   vmin = 0, vmax = color_max,
			   cmap=cmap_magma_reduced
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

	cbar_title = r"$\sigma_{xy}[e^2/h]$"
	cbar = fig.colorbar(im_low, location='left', shrink=0.6, ticks=[0, 0.5, 0.9])
	cbar.ax.tick_params(labelsize=label_size) 
	cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25) 
		
	a2.axvline(x=1, color='black', linestyle='--')
	a2.axvline(x=0, color='black', linestyle='--')
	a2.axvline(x=color_max, color='gray', linestyle='dotted')
	
	a2.text(0.65, 0.05, r"$\sigma_{xy}$ = " + rf"{np.round(color_max[0], decimals = 2)}", 
		 fontsize = label_scale_inset * label_size, transform=a2.transAxes, color = "gray")
	
	a2.plot(I_low, E_low, color = "black", label = r"$W = 0$")
	a2.plot(I_up, E_up, color = "black")
		
	# add numerical data
	E_vals_data = np.genfromtxt(data_location + "/E_vals.txt", dtype= float, delimiter=' ') 
	data_sigma_xy = np.genfromtxt(data_location + "/result_sigma_xy.txt", dtype= float, delimiter=' ') 
	data_sigma_xy_mean = np.mean(data_sigma_xy, axis = 1)
	data_sigma_xy_std = np.std(data_sigma_xy, axis = 1) / np.sqrt(len(data_sigma_xy[0,:]))
	
	a2.plot(data_sigma_xy_mean, E_vals_data, color = "r", label = r"$W = 1.5$")
	
	# indicate standard deviation
	E_l_plateu_1 = -1.7
	E_r_plateu_1 = -0.5
	E_l_plateu_2 = 0.5
	E_r_plateu_2 = 1.7
	
	E_l_plateu_1 = -4
	E_r_plateu_1 = 4
	
	mask_plateau_1 = (E_vals_data > E_l_plateu_1) &  (E_vals_data < E_r_plateu_1)
	mask_plateau_2 = (E_vals_data > E_l_plateu_2) &  (E_vals_data < E_r_plateu_2)

	a2.fill_betweenx(E_vals_data[mask_plateau_1], data_sigma_xy_mean[mask_plateau_1] - data_sigma_xy_std[mask_plateau_1], data_sigma_xy_mean[mask_plateau_1] + data_sigma_xy_std[mask_plateau_1], facecolor = "red", alpha = 0.2, lw = 0)
	
	# add inset
	n_vals_inset = 50
	x_data_inset = np.genfromtxt(data_location_inset + "/Nx_vals.txt", dtype= float, delimiter=' ') 
	data_inset = np.genfromtxt(data_location_inset + "/result_sigma_xy.txt", dtype= float, delimiter=' ') 
	data_inset_mean = np.mean(data_inset[:, :n_vals_inset], axis = 1)
	data_inset_std = np.std(data_inset[:, :n_vals_inset], axis = 1) / np.sqrt(n_vals_inset)

	x_0 = 0.19
	y_0 = 0.64
	
	inset_width = 0.4
	inset_height = 0.15
		
	a2_inset = a2.inset_axes(
	[x_0, y_0, inset_width, inset_height],
	xlim=(400, 2000), ylim=(0.5, 1))
	
	a2_inset.fill_between(x_data_inset, data_inset_mean - data_inset_std, data_inset_mean + data_inset_std, facecolor = "red", alpha = 0.2,  lw = 0)
	a2_inset.plot(x_data_inset, data_inset_mean, color = "red")
	a2_inset.axhline(y=1, color='black', linestyle='--')
			
	# set labels and fonts	
	a1.set_xlabel(r"$k_x$", fontsize=label_size)
	#a1.set_ylabel("E", fontsize=30)
	a1.set_xlim([-1.05 * np.pi, 1.05 * np.pi])
	a1.set_ylim([1.05 * E_min_low, 1.05 * E_max_up])
	
	a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
	
	a1.set_xticks([-np.pi, 0, np.pi])
	a1.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
	
	a1.xaxis.set_label_coords(0.75, -0.04)
	
	a1.set_yticks([])
	
	a2.set_xlim([-0.1,1.1])
	
	a2.set_ylim([1.05 * E_min_low, 1.05 * E_max_up])
		
	a2.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
	a2.set_xticks([0, 0.5, 1])
	a2.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
	
	a2.set_yticks([-3, -2, -1, 0., 1, 2, 3])
	a2.set_yticklabels([r"$-3$", "", "", r"$0$", "", "", r"$3$"])
	a2.yaxis.set_ticks_position('right')

	a2.set_xlabel(r"$\sigma_{xy}$", fontsize=label_size)	
	a2.set_ylabel(r"$E_\mathrm{F}$", fontsize=label_size, rotation='horizontal')
		
	a2.xaxis.set_label_coords(0.7, -0.04)
	a2.yaxis.set_label_coords(1.05, 0.75)
	
	a2.legend(loc="upper left", bbox_to_anchor=(0., 0.46), fontsize = label_size)

	a2_inset.set_xlabel(r"$N_{x}$", fontsize = label_scale_inset * label_size)    
	a2_inset.set_ylabel(r"$\sigma_{xy}$", fontsize = label_scale_inset * label_size, rotation='horizontal')
	a2_inset.xaxis.set_label_coords(0.8, -0.17)
	a2_inset.yaxis.set_label_coords(-0.13, 0.14)

	a2_inset.tick_params(direction='out', length=4, width=2,  labelsize = label_scale_inset * label_size, pad = 3)
	a2_inset.set_xticks([500, 1000, 1500, 2000, 2500, 3000])
	#a2_inset.set_xticklabels(["", r"$1$", "", r"$2$","", r"$3$"])
	a2_inset.set_xticklabels(["", r"$1\mathrm{e}3$", "", r"$2\mathrm{e}3$", "", r"$3\mathrm{e}3$"])
	
	a2_inset.set_yticks([0.8, 1, 1.2])
	a2_inset.set_yticklabels([r"$0.8$", r"$1$", r"$1.2$"])
	
	a2_inset.set_ylim([0.8, 1.2])
	
	a1.text(0.5,0.94, r"(a)", fontsize = label_size, transform=a1.transAxes)
	a2.text(0.5, 0.94, r"(b)", fontsize = label_size, transform=a2.transAxes)
	a2_inset.text(0.07, 0.72, r"$E_\mathrm{F} = 1.2, W = 1.5$", fontsize = label_scale_inset * label_size, transform=a2_inset.transAxes)

	fig.tight_layout()
	fig.savefig("Fig_1_publication.png",  bbox_inches='tight', dpi = 600)

	plt.show()
	

def main():

	N = 400
	N_colorsteps = 500
	
	r = 1.5
	epsilon_1 = 0.3
	epsilon_2 = 2

	gamma = 2
	gamma_2 = 0.3
	gamma_3 = 0.

	data_location_sigma_xy= "Data_Fig_1_Hall_line"
	data_location_inset = "Data_Fig_1_Size_Scaling"
	
	Fig_1(N, N_colorsteps, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3, data_location_sigma_xy, data_location_inset)
	
	
if __name__ == '__main__':
	main()
