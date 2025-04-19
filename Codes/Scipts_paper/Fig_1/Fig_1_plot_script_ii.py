import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors
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
		

def gen_dictionary(path):
	"""Generate dictionary with parameters from file."""
	parameters = {}
	
	with open(path) as f:        
		for line in f:            
			current_line = line.split()            
			try: 
				parameters[current_line[0]] = np.array(current_line[1:], dtype = float)
			except:
				try: 
					parameters[current_line[0]] = np.array(current_line[1:], dtype = str)                
				except:
					pass
	return parameters


def Fig_1_ii(N, N_colorsteps, epsilon_1, epsilon_2, gamma, gamma_2, data_location, data_location_inset):
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
		
	print("Starting colorcode")	
	# Show the data as contourplot in imshow
	im_low = a1.imshow(color_data_low,
			   aspect='auto',
			   origin='lower',
			   extent=[-np.pi, np.pi, E_min_low, E_max_low],
			   vmin = 0,
			   vmax = 1,
			   cmap=plt.cm.magma
			  )
	
	im_up = a1.imshow(color_data_up,
			   aspect='auto',
			   origin='lower',
			   extent=[-np.pi, np.pi, E_min_up, E_max_up],
			   vmin = 0,
			   vmax = 1,
			   cmap=plt.cm.magma
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
	cbar = plt.colorbar(im_low, location='left', shrink=0.6, ticks=[0, 0.5, 1])
	cbar.ax.tick_params(labelsize=label_size) 
	cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25) 
		
	I_low, I_up, E_low, E_up, energies_band_low 

	a2.axvline(x=1, color='black', linestyle='--')
	
	a2.plot(I_low, E_low, color = "b", label = r"$W = 0$")
	a2.plot(I_up, E_up, color = "b")
	
	# add numerical data
	n_vals = 7
	E_vals_data = np.genfromtxt(data_location + "/E_vals.txt", dtype= float, delimiter=' ') 
	data_sigma_xy = np.genfromtxt(data_location + "/result_sigma_xy.txt", dtype= float, delimiter=' ') 
	#data_sigma_yx = np.genfromtxt(data_location + "/result_sigma_yx.txt", dtype= float, delimiter=' ') 
	#data_mean = 0.5 * (data_sigma_xy + data_sigma_yx)
	data_sigma_xy_mean = np.mean(data_sigma_xy[:, :n_vals], axis = 1)
	data_sigma_xy_std = np.std(data_sigma_xy[:, :n_vals], axis = 1)
	
	a2.plot(data_sigma_xy_mean, E_vals_data, color = "r", label = r"$W = 1$")
	
	# indicate standard deviation
	E_l_plateu_1 = -1.7
	E_r_plateu_1 = -0.3
	E_l_plateu_2 = 1
	E_r_plateu_2 = 2
	
	mask_plateau_1 = (E_vals_data > E_l_plateu_1) &  (E_vals_data < E_r_plateu_1)
	mask_plateau_2 = (E_vals_data > E_l_plateu_2) &  (E_vals_data < E_r_plateu_2)

	a2.fill_betweenx(E_vals_data[mask_plateau_1], data_sigma_xy_mean[mask_plateau_1] - data_sigma_xy_std[mask_plateau_1], data_sigma_xy_mean[mask_plateau_1] + data_sigma_xy_std[mask_plateau_1], facecolor = "gray", lw = 0)
	a2.fill_betweenx(E_vals_data[mask_plateau_2], data_sigma_xy_mean[mask_plateau_2] - data_sigma_xy_std[mask_plateau_2], data_sigma_xy_mean[mask_plateau_2] + data_sigma_xy_std[mask_plateau_2], facecolor = "gray", lw = 0)
	
	# add inset
	x_data_inset = np.genfromtxt(data_location_inset + "/Nx_vals.txt", dtype= float, delimiter=' ') 
	data_inset = np.genfromtxt(data_location_inset + "/result_sigma_xx.txt", dtype= float, delimiter=' ') 
	data_inset_mean = np.mean(data_inset, axis = 1)
	data_inset_std = np.std(data_inset, axis = 1)

	x_0 = 0.1
	y_0 = 0.2
	inset_width = 0.4
	inset_height = 0.2
		
	a2_inset = a2.inset_axes(
	[x_0, y_0, inset_width, inset_height],
	xlim=(0, 2000), ylim=(0, 2))
	
	a2_inset.fill_between(x_data_inset, data_inset_mean - data_inset_std, data_inset_mean + data_inset_std, facecolor = "gray", lw = 0)
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
	
	a2.legend(loc="upper left", bbox_to_anchor=(0., 0.85), fontsize = label_size)

	a2_inset.set_xlabel(r"$N_{x}$", fontsize = label_scale_inset * label_size)	
	a2_inset.set_ylabel(r"$\sigma_{xy}$", fontsize = label_scale_inset * label_size, rotation='horizontal')
	a2_inset.xaxis.set_label_coords(0.75, -0.1)
	a2_inset.yaxis.set_label_coords(-0.1, 0.55)

	a2_inset.tick_params(direction='out', length=4, width=2,  labelsize = label_scale_inset * label_size, pad = 5)
	a2_inset.set_xticks([0, 1000, 2000])
	a2_inset.set_xticklabels([r"$0$", r"$1000$", r"$2000$"])
	
	a2_inset.set_yticks([0, 0.5, 1])
	a2_inset.set_yticklabels([r"$0$", r"$0.5$", r"$1$"])
	
	a1.text(0.5,0.94, r"a)", fontsize = label_size, transform=a1.transAxes)
	a2.text(0.5, 0.94, r"b)", fontsize = label_size, transform=a2.transAxes)
	a2_inset.text(0.2, 0.25, r"$E_\mathrm{F} = -1$", fontsize = label_scale_inset * label_size, transform=a2_inset.transAxes)

	fig.tight_layout()
	fig.savefig("Fig_1_publication.png",  bbox_inches='tight')

	plt.show()
	

def main():

	N = 100
	N_colorsteps = 100
	
	epsilon_1 = 0.3
	epsilon_2 = 2

	gamma = 1.5
	gamma_2 = 0.1

	data_location = "G_final_V3_Hall_line_run_12"
	data_location_inset = "G_final_Hall_size_scaling_run_1"
	
	Fig_1_ii(N, N_colorsteps, epsilon_1, epsilon_2, gamma, gamma_2, data_location, data_location_inset)

	
	
	
	
	
if __name__ == '__main__':
	main()
