import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib
import json

# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "times"
})


sigma_0 = np.identity(2)
sigma_x = np.array([[0,1], [1,0]])
sigma_y = np.array([[0,-1j], [1j,0]])
sigma_z = np.array([[1,0], [0,-1]])



def H_lin(kx, ky, m):
	"""Bloch Hamiltonian for Chern insulator"""
	h_x = m*2 * kx 
	h_y = m ** 2 * ky
	h_z = m + (kx**2 + ky**2)/2
	
	return h_x * sigma_x + h_y * sigma_y + h_z * sigma_z


def H_dipole(kx, ky, r1, r2):
	"""Bloch Hamiltonian that exhibits a dipole-like charge separation of topological charge"""
	r_k =  r1 * (1 + np.cos(kx)) / 2 + r2 * ((1 - np.cos(kx)) / 2) 
	h_x = np.sin(kx)
	h_y = np.sin(ky)
	h_z = (r_k - (np.cos(2 * kx) + np.cos(ky))) 
	
	return h_x * sigma_x + h_y * sigma_y + h_z* sigma_z


def H_dipole_2(kx, ky, r1, r2):
	"""Bloch Hamiltonian that exhibits a dipole-like charge separation of topological charge"""
	
	h_x = r1 ** 2 * np.sin(kx)  
	h_y = r1 ** 2 * np.sin(ky)
	h_z_r1 = (r1 - np.cos(kx) - np.cos(ky)) 
	h_z_r2 = -0 * (r1 - np.cos(kx) - np.cos(ky)) * (1 + np.cos(kx)) * (1 + np.cos(ky))/4    # dominant at k = pi

	return h_x * sigma_x + h_y * sigma_y + (h_z_r1 + h_z_r2) * sigma_z
	

def Berry_curvature(N, H_bloch):
	"""Calculate berry curvature of H_bloch and return it as an array along with k_x, k_y values suitable for contour plot."""
	delta_k = 1 / N ** 2
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
	return Phi_vals, C_n, kx_array, ky_array, energies


def contour_plot_berry_curvature(N, n = 0, r1 = -2, r2 = -1):
	"""Do contour plot for Berry curvature"""
	
	H_bloch = lambda kx, ky: H_dipole(kx, ky, r1, r2)
	
	Phi_vals, C_n, k_x_vals, k_y_vals, energies = Berry_curvature(N, H_bloch)

	print("Chern numbers: ", C_n)

	a_1_title = rf"Berry curvature for band {n}, $r_1 = {r1}$, $r_2 = {r2}$"
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

	levels = np.linspace(-100, 100, 500)
	contour_plot_curvature = a1.contourf(k_x_vals, k_y_vals, N**2* Phi_vals[:,:, n], levels = levels, extend = "both", cmap=plt.cm.magma_r)
	contour_plot_energy = a2.contourf(k_x_vals, k_y_vals, energies[:,:, n], levels = 100, extend = "both", cmap=plt.cm.viridis)

	cbar_1 = plt.colorbar(contour_plot_curvature, ax = a1, label =r"$\Omega$")    
	cbar_2 = plt.colorbar(contour_plot_energy, ax = a2, label = r"$E(k_x, k_y)$")

	
	plt.show()

	return


def plot_energy_berry_curvature(N, r1, r2):
	"""Plot energy of all bands over k_x with a color-code representing the Berry curvature. """

	H_bloch = lambda kx, ky: H_dipole(kx, ky, r1, r2)
	
	Phi_vals, C_n, k_x_vals, k_y_vals, energies = Berry_curvature(2 * N, H_bloch)

	# add together phi_vals for opposing k_y as they have the same energy. A grid spaced with an even number 
	# of sites ensures that there is a pair for all ky 

	n_bands = len(C_n)

	color_data = np.zeros((N, 2 * N, n_bands))

	for j_y in range(0, N):
		j_y_op =  2 * N - 1 - j_y
		color_data[j_y, :, :] = (2*N) ** 2 * (Phi_vals[j_y, :, :] + Phi_vals[j_y_op, :, :])
	
	print("Chern numbers: ", C_n)

	title = rf"Bulk bands and Berry curvature for $r_1= {r1}$, $r_2 = {r2}$"

	fig = plt.figure(figsize=(24, 12), layout = "tight")
	a1 =  fig.add_subplot(1,1,1)
	a1.set_title(title , fontsize=30)
	a1.set_xlabel(r"$k_x$", fontsize=30)
	a1.set_ylabel("E", fontsize=30)

	levels = np.linspace(-10, 10, 100)

	v_min = -1000
	v_max = 1000


	# plot bands with colorcode representing curvature 
	for l in range(0, n_bands):
		scatterplot = a1.scatter(k_x_vals[:N, :], energies[:N, :, l], marker = "o", c = color_data[:,:,l], 
								 label = r"$E_{n}(k_x)$", cmap=plt.cm.magma_r, vmin= v_min, vmax = v_max)
		
	fig.colorbar(scatterplot)    
	
	#fig.tight_layout()
	#fig.savefig(f"Curvature_energy_DC_CI_r1_{r1}_r_2_{r2}.png",  bbox_inches='tight', dpi=300)

	plt.show()



			  
def main():
	N = 100
	n = 0
	m = 0

	r1 = 1.7
	r2 = 1
	#print(H_dipole(0, 0, r1, r2))

	#contour_plot_berry_curvature(N, n, r1, r2)
	plot_energy_berry_curvature(N, r1, r2)

	





if __name__ == '__main__':
	main()
