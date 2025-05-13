import numpy as np
import matplotlib as mpl

from matplotlib import pyplot as plt

import kwant

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
		

def syst_final_V3(Nx, Ny, r, epsilon_1, epsilon_2, gamma, gamma_2, PBC = 0., wrap_dir = "x"):
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

	mat_os = gamma_2 * r * sigma_z 
	hop_mat_dx = gamma/(2j) * sigma_x 
	hop_mat_2dx = - gamma_2 *(1/2) * sigma_z 
	hop_mat_dy = ((epsilon_1 + epsilon_2/2)/(2j)) * sigma_y - ((epsilon_1 + epsilon_2/2) / 2) * sigma_z 
	hop_mat_dx_dy = -(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z 
	hop_mat_dx_mdy = +(epsilon_2/(8j)) * sigma_y + (epsilon_2 / 8) * sigma_z 
	
	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy					
					syst[lat(jx,jy)] = mat_os
	
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hop_mat_dx
	syst[kwant.builder.HoppingKind((2, 0), lat, lat)] = hop_mat_2dx
	
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
		syst[kwant.builder.HoppingKind((-(Nx-2), 0), lat, lat)] = PBC * hop_mat_2dx
		
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


def Fig_1_App(N, N_k,  r, epsilon_1, epsilon_2, gamma, gamma_2):
	"""Plot Cylinder spectrum with edge states indicated for a cut in y and x direction."""
	PBC = 0.
	weight = 0.9
	# generate system
	syst_x_wrap = syst_final_V3(N, N, r, epsilon_1, epsilon_2, gamma, gamma_2, PBC, "x")
	syst_y_wrap = syst_final_V3(N, N, r, epsilon_1, epsilon_2, gamma, gamma_2, PBC, "y")
	
	k_vals = np.linspace(-np.pi, np.pi, N_k)
	energies_x_wrap = np.zeros((2 * N, N_k))
	energies_y_wrap = np.zeros((2 * N, N_k))
	k_vals_edge_x_wrap = [list(), list()]
	k_vals_edge_y_wrap = [list(), list()]
	energies_edge_x_wrap = [list(), list()]
	energies_edge_y_wrap = [list(), list()]
	
	N_edge = int(0.5 * N)
	
	# get momentum eigentstaes for k. filter by proximity to edge
	for j in range(N_k):
		k = k_vals[j]
		
		H_k_x_wrap = syst_x_wrap.hamiltonian_submatrix(params = {"k_1":k})
		energies_k_x_wrap, EV_k_x_wrap = np.linalg.eigh(H_k_x_wrap)
		energies_x_wrap[:, j] = energies_k_x_wrap
		
		H_k_y_wrap = syst_y_wrap.hamiltonian_submatrix(params = {"k_1":k})
		energies_k_y_wrap, EV_k_y_wrap = np.linalg.eigh(H_k_y_wrap)
		energies_y_wrap[:, j] = energies_k_y_wrap
		
		# check if states are localized at edge
		for l in range(2 * N):
			localization_x_wrap_R = np.linalg.norm(EV_k_x_wrap[:N_edge, l]) 
			localization_x_wrap_L = np.linalg.norm(EV_k_x_wrap[-N_edge:, l]) 
			localization_y_wrap_L = np.linalg.norm(EV_k_y_wrap[:N_edge, l]) 
			localization_y_wrap_R = np.linalg.norm(EV_k_y_wrap[-N_edge:, l]) 
			
			if localization_x_wrap_L > weight:
				k_vals_edge_x_wrap[0].append(k)
				energies_edge_x_wrap[0].append(energies_k_x_wrap[l])
			
			if localization_x_wrap_R > weight:
				k_vals_edge_x_wrap[1].append(k)
				energies_edge_x_wrap[1].append(energies_k_x_wrap[l])
				
			if localization_y_wrap_L > weight:
				k_vals_edge_y_wrap[0].append(k)
				energies_edge_y_wrap[0].append(energies_k_y_wrap[l])				
				
			if localization_y_wrap_R > weight:
				k_vals_edge_y_wrap[1].append(k)
				energies_edge_y_wrap[1].append(energies_k_y_wrap[l])				

	fig, axs = plt.subplots(1, 2, figsize =(24, 12))
	fig.subplots_adjust(wspace=0.1)	
	a1 = axs[0]
	a2 = axs[1]
	
	label_size = 40	
	
	a1.set_xlabel(r"$k_x$", fontsize=label_size)
	a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
		
	a1.set_xticks([-np.pi, 0, np.pi])
	a1.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
	a1.set_yticks([])
	
	a1.xaxis.set_label_coords(0.75, -0.04)
	
	for l in range(2 * N):
		a1.plot(k_vals, energies_x_wrap[l,:], color = "black", zorder=0)
	
	a1.plot(k_vals_edge_x_wrap[0], energies_edge_x_wrap[0], color = "r", zorder=1, label = r"Left edge", lw = 2)#, marker = "o", s = 2)
	a1.plot(k_vals_edge_x_wrap[1], energies_edge_x_wrap[1], color = "b", zorder=2, label = r"Right edge", lw = 2)
	
	a2.set_xlabel(r"$k_y$", fontsize=label_size)
	a2.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
		
	a2.set_xticks([-np.pi, 0, np.pi])
	a2.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
	a2.set_yticks([])
	
	a2.xaxis.set_label_coords(0.75, -0.04)
	
	for l in range(2 * N):
		a2.plot(k_vals, energies_y_wrap[l,:], color = "black", zorder=0)
	
	a2.plot(k_vals_edge_y_wrap[0], energies_edge_y_wrap[0], color = "r", zorder=1, lw = 2)
	a2.plot(k_vals_edge_y_wrap[1], energies_edge_y_wrap[1], color = "b", zorder=2, lw = 2)
	
	a1.text(0.5,0.94, r"(a)", fontsize = label_size, transform=a1.transAxes)
	a2.text(0.5, 0.94, r"(b)", fontsize = label_size, transform=a2.transAxes)
	
	#a1.legend(loc="upper left", bbox_to_anchor=(0.5, 0.2), fontsize = label_size)
	a1.legend(loc="lower center", fontsize = 0.75 * label_size)
	
	fig.tight_layout()
	fig.savefig("Fig_1_App_publication.png",  bbox_inches='tight', dpi = 200)

	plt.show()

def main():
	N = 100
	N_k = 500	

	r = 1.5
	epsilon_1 = 0.3
	epsilon_2 = 2

	gamma = 2
	gamma_2 = 0.3
	
	
	Fig_1_App(N, N_k,  r, epsilon_1, epsilon_2, gamma, gamma_2)

	
	
	
	
	
if __name__ == '__main__':
	main()
