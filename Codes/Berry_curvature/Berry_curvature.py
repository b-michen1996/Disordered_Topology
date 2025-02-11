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


def H_k_double_chern(r, kx, ky, E_sep):
	"""Bloch Hamiltonian backfolding 1"""    
	
	gamma_1 = np.kron(sigma_x, sigma_0)
	gamma_2_a = np.kron(sigma_y, sigma_x)
	gamma_2_b = np.kron(sigma_y, sigma_y)
	gamma_3 = np.kron(sigma_z, sigma_0)
	
	gamma_4 = np.kron(sigma_0, sigma_z)
	
	separator = np.kron(sigma_0, sigma_z)
	
	
	d_1 = np.sin(kx)
	d_2_a = np.cos(ky/2) * np.sin(ky/2)
	d_2_b = np.sin(ky/2) * np.sin(ky/2)
	d_3 = r - np.cos(kx) - np.cos(ky)
	
	d_4 = (1 - np.cos(ky))/4
	d_4 = 0.1
	
	

	res = d_1 * gamma_1 + d_2_a * gamma_2_a + d_2_b * gamma_2_b + d_3 * gamma_3  + d_4 * gamma_4 # + E_sep * separator
	#res = d_1 * gamma_1 + d_4 * gamma_4 + d_3 * gamma_3 #+ E_sep * separator
	
	return res


def H_lin(kx, ky, m):
	"""Bloch Hamiltonian for Chern insulator"""
	h_11 = 100*  (m + (kx**2 + ky**2)/2)

	h_12 = kx - 1j * ky

	res = np.array([[h_11, h_12],
					[np.conj(h_12), -h_11]])
	return res


def H_dipole(kx, ky, r1, r2):
	"""Bloch Hamiltonian that exhibits a dipole-like charge separation of topological charge"""
	h_x = np.sin(kx) 
	h_y = np.sin(ky)
	h_z_r1 = (r1 - np.cos(kx) - np.cos(2 * ky)) # * (1 + np.cos(ky)) # dominant at k = 0
	h_z_r2 = (r2 - np.cos(kx) - np.cos(ky)) * (1 - np.cos(ky))   # dominant at k = pi

	return h_x * sigma_x + h_y * sigma_y + (h_z_r1 + h_z_r2) * sigma_z
	

def H_CI(kx, ky, r = 1):
	"""Bloch Hamiltonian for Chern insulator"""
	h_11 = r - np.cos(2 * kx) - np.cos(ky)


	h_12 = np.sin(kx) + 1j * np.sin(ky)

	res = np.array([[h_11, h_12],
					[np.conj(h_12), -h_11]])
	return np.exp(kx ** 2 + ky ** 2)  * res


def H_QSH(kx, ky, m = 0.1, alpha = 0.3645, beta = 0.686, gamma = 0.512):
		"""Bloch Hamiltonian"""
		epsilon = 2 * gamma * (2 - np.cos(kx) - np.cos(ky))

		dx_m_idy = alpha * (np.sin(kx) + 1j * np.sin(ky))
		dz = m + 2 * beta * (2 - np.cos(kx) - np.cos(ky))                                   
		
		res = np.array([[epsilon + dz, dx_m_idy],
				  [np.conj(dx_m_idy), epsilon - dz]])
		
		return res 


def H_Fulga_Bergholtz(kx, ky, v = 0.5):
	"""Bloch Hamiltonian for Chern insulator"""
	h_11 = 2 * (np.cos(kx) - np.cos(ky))
	
	h_12 = np.sqrt(2) * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx)
			+ np.exp(1j * ky) +  1j * np.exp(1j * (kx + ky)) + 1) 
	
	h_12 = np.sqrt(2) * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx)
			+ np.exp(-1j * ky) +  1j * (np.exp(1j * (kx - ky)) + 1)) 
	
	res = np.array([[h_11, h_12, v],
				[np.conj(h_12), -h_11, 0],
				[v, 0, 0]])
	
	return res


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


def contour_plot_berry_curvature(N, n = 0):
	"""Do contour plot for Berry curvature"""

	r = 1.99

	H_bloch = lambda kx, ky: H_CI(kx, ky, r)

	v = 5.5

	H_bloch = lambda kx, ky: H_Fulga_Bergholtz(kx, ky, v)    

	m = 0.0005
	#H_bloch = lambda kx, ky: H_QSH(kx, ky, m)

	m = -0.01
	#H_bloch = lambda kx, ky: H_lin(kx, ky, m)

	r1 = 1.99
	r2 = 1.9
	#H_bloch = lambda kx, ky: H_dipole(kx, ky, r1, r2)
	
	Phi_vals, C_n, k_x_vals, k_y_vals, energies = Berry_curvature(N, H_bloch)

	print("Chern numbers: ", C_n)

	title = rf"Berry curvature for band {n}"

	fig = plt.figure(figsize=(6, 6), layout = "tight")
	a1 =  fig.add_subplot(1,1,1)
	a1.set_title(title , fontsize=30)
	a1.set_xlabel("x", fontsize=30)
	a1.set_ylabel("y", fontsize=30)
	a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)

	levels = np.linspace(-10, 10, 100)
	contour_plot = a1.contourf(k_x_vals, k_y_vals, N**2* Phi_vals[:,:, n], levels = levels, extend = "both", cmap=plt.cm.magma_r)
	#contour_plot = a1.contourf(k_x_vals, k_y_vals, energies[:,:, n], cmap=plt.cm.magma_r)

	fig.colorbar(contour_plot)

	plt.show()

	return


def plot_energy_berry_curvature(N):
	"""Plot energy of all bands over k_x with a color-code representing the Berry curvature. """

	r = 1.99

	H_bloch = lambda kx, ky: H_CI(kx, ky, r)

	v = 4.5

	H_bloch = lambda kx, ky: H_Fulga_Bergholtz(kx, ky, v)

	r1 = 1.9
	r2 = 1
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

	title = rf"Bulk bands and Berry curvature for $v = {v}$"

	fig = plt.figure(figsize=(24, 12), layout = "tight")
	a1 =  fig.add_subplot(1,1,1)
	a1.set_title(title , fontsize=30)
	a1.set_xlabel(r"$k_x$", fontsize=30)
	a1.set_ylabel("E", fontsize=30)

	levels = np.linspace(-10, 10, 100)

	v_min = -100
	v_max = 100


	# plot bands with colorcode representing curvature 
	for l in range(0, n_bands):
		scatterplot = a1.scatter(k_x_vals[:N, :], energies[:N, :, l], marker = "o", c = color_data[:,:,l], 
								 label = r"$E_{n}(k_x)$", cmap=plt.cm.magma_r, vmin= v_min, vmax = v_max)
		
	fig.colorbar(scatterplot)  

	#fig.tight_layout()
	#fig.savefig("Curvature_energy.png",  bbox_inches='tight', dpi=300)

	plt.show()



			  
def main():
	t = 1
	N = 100

	# plot_backfolding_bands(N, t)

	Nx = 100
	Ny = 100
	r = 0.5

	gamma = 1
	E_sep = 4


	N = 100
	n = 1

	#contour_plot_berry_curvature(N, n)
	plot_energy_berry_curvature(N)





if __name__ == '__main__':
	main()
