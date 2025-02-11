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


def H_cylinder(kx, Ny, r1, r2, PBC = 0.):
	"""H(kx) for cylinder with width N_y. """
	res = 1j * np.zeros((2 * Ny,2 * Ny))
	
	# even/odd indices correspond to sublattice a/b
	for jy in range(0, Ny):
		jy_a = 2 * jy
		jy_b = 2 * jy + 1
		
		jy_a_p1 = 2 * (jy + 1)
		jy_b_p1 = 2 * (jy + 1) + 1
		
		res[jy_a, jy_a] = (r1 + r2)/2 + np.cos(kx) * (r1 - r2)/2 - np.cos(2 * kx) 
		res[jy_b, jy_b] = -((r1 + r2)/2 + np.cos(kx) * (r1 - r2)/2 - np.cos(2 * kx))
		
		res[jy_a, jy_b] = np.sin(kx)
		res[jy_b, jy_a] = np.sin(kx)
		
		if jy < (Ny - 1):
			res[jy_a, jy_b_p1] = -1/2
			res[jy_b, jy_a_p1] = 1/2
			
			res[jy_b_p1, jy_a] = -1/2
			res[jy_a_p1, jy_b] = 1/2
			
			res[jy_a, jy_a_p1] = -1/2
			res[jy_b, jy_b_p1] = 1/2
			
			res[jy_a_p1, jy_a] = -1/2
			res[jy_b_p1, jy_b] = 1/2
		else:
			jy_a_p1 = 0
			jy_b_p1 = 1
		
			res[jy_a, jy_b_p1] = -PBC * 1/2 
			res[jy_b, jy_a_p1] = PBC * 1/2
			
			res[jy_b_p1, jy_a] = -PBC * 1/2
			res[jy_a_p1, jy_b] = PBC * 1/2
			
			res[jy_a, jy_a_p1] = -PBC * 1/2
			res[jy_b, jy_b_p1] = PBC * 1/2
			
			res[jy_a_p1, jy_a] = -PBC * 1/2
			res[jy_b_p1, jy_b] = PBC * 1/2
	
	return res
			
			
def plot_spec_cylinder(Nx, Ny, r1, r2, PBC = 0.):
	"""Plot bandstructure for cylinder (i.e. PBC along x). PBC can be set optionally."""
	kx_vals = np.linspace(-np.pi, np.pi, Nx, endpoint=False)
	plot_data = np.zeros((Nx, 2 * Ny))
	
	for jx in range(0, Nx):
		kx = kx_vals[jx]		
		H_kx = H_cylinder(kx, Ny, r1, r2, PBC)
		
		energies_kx = np.linalg.eigvalsh(H_kx)
		plot_data[jx, :] = energies_kx

	title = rf"Cylinder bands for $r_1= {r1}$, $r_2 = {r2}$, $N_y = {Ny}$, PBC = {PBC} "

	fig = plt.figure(figsize=(24, 12), layout = "tight")
	a1 =  fig.add_subplot(1,1,1)
	a1.set_title(title , fontsize=30)
	a1.set_xlabel(r"$k_x$", fontsize=30)
	a1.set_ylabel("E", fontsize=30)
	
	# plot bands with colorcode representing curvature 
	for jy in range(0, 2 * Ny):
		scatterplot = a1.scatter(kx_vals, plot_data[:, jy], marker = "o")
		
	plt.show()
	



			  
def main():
	Nx = 200
	Ny= 60

	r1 = 1.9
	r2 = 1
	
	PBC = 0.
	
	plot_spec_cylinder(Nx, Ny, r1, r2, PBC)
	
if __name__ == '__main__':
	main()
