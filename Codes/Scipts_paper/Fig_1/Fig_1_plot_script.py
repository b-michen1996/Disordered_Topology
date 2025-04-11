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
		

def Fig_1(N, N_colorsteps, epsilon_1, epsilon_2, gamma, gamma_2, data_location):
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
	
	E_min_low = np.min(E_low)
	E_max_low = np.max(E_low)
	E_min_up = np.min(E_up)
	E_max_up = np.max(E_up)
	
	label_size = 25
	
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
			   cmap=plt.cm.viridis
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
	cbar = plt.colorbar(im_low, location='left', shrink=0.6, ticks=[0, 0.5, 1])
	cbar.ax.tick_params(labelsize=label_size) 
	cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25) 
	
	# plot clean and dirty sigma_xy
	final_integral_lower, final_integral_upper, final_energies_lower, final_energies_upper, energies_band_low = Berry_codes.sigma_xy_analytically(N, epsilon_1, epsilon_2, gamma, gamma_2)
	
	plt.axvline(x=1, color='black', linestyle='--')
	
	a2.plot(final_integral_lower, final_energies_lower, color = "b", label = r"$W = 0$")
	a2.plot(final_integral_upper, final_energies_upper, color = "b")
	
	# add numerical data
	n_vals = 18
	E_vals_data = np.genfromtxt(data_location + "/E_vals.txt", dtype= float, delimiter=' ') 
	data_sigma_xy = np.genfromtxt(data_location + "/result_sigma_xy.txt", dtype= float, delimiter=' ') 
	data_sigma_xy_mean = np.mean(data_sigma_xy[:, :n_vals], axis = 1)
	
	a2.plot(data_sigma_xy_mean, E_vals_data, color = "r", label = r"$W = 1.2$")
	
	# set labels and fonts
	
	a1.set_xlabel(r"$k_x$", fontsize=label_size)
	#a1.set_ylabel("E", fontsize=30)
	a1.set_xlim([-1.05 * np.pi, 1.05 * np.pi])
	a1.set_ylim([1.05 * E_min_low, 1.05 * E_max_up])
	
	a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
	
	a1.set_xticks([-np.pi, 0, np.pi])
	a1.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
	
	a1.xaxis.set_label_coords(0.75, -0.025)
	
	a1.set_yticks([])
	
	a2.set_xlim([0.,1.1])
	
	a2.set_ylim([1.05 * E_min_low, 1.05 * E_max_up])
		
	a2.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
	a2.set_xticks([0, 0.5, 1])
	a2.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
	
	a2.set_yticks([-3, 0., 3])
	#a2.yaxis.tick_right()
	a2.yaxis.set_ticks_position('right')

	
	a2.set_xlabel(r"$\sigma_{xy}$", fontsize=label_size)	
	a2.set_ylabel(r"$E_\mathrm{F}$", fontsize=label_size, rotation='horizontal')
		
	a2.xaxis.set_label_coords(0.75, -0.025)
	a2.yaxis.set_label_coords(1.05, 0.75)
	
	a2.legend(loc="upper left", bbox_to_anchor=(0.,0.85), fontsize = label_size)

	
	a1.text(0.5,0.94, r"a)", fontsize = label_size, transform=a1.transAxes)
	a2.text(0.5, 0.94, r"b)", fontsize = label_size, transform=a2.transAxes)

	fig.tight_layout()
	fig.savefig("Fig_1_publication.png",  bbox_inches='tight')

	plt.show()
	

def main():

	N = 400
	N_colorsteps = 400
	
	epsilon_1 = 0.3
	epsilon_2 = 2

	gamma = 2
	gamma_2 = 0.2

	data_location = "Kubo_final_line_results/Kubo_final_line_run_27"
	
	Fig_1(N, N_colorsteps, epsilon_1, epsilon_2, gamma, gamma_2, data_location)

	
	
	
	
	
if __name__ == '__main__':
	main()
