import kwant
import kwant.wraparound
import numpy as np
import os
import gc
from matplotlib import pyplot as plt


def plot_energy_double_cone_CI(Nx, Ny, r_1, r_2, PBC = 1.):
	"""Creates system with the given parameters and then adds random impurities."""
	# values for lattice constant and number of orbitals
	a = 1
	
	# matrices acting in orbital space to build the system
	sigma_x = np.array([[0, 1],
				[1, 0]])
	
	sigma_y = np.array([[0, -1j],
				[1j, 0]])
	
	sigma_z = np.array([[1, 0],
				[0, -1]])
	
	sigma_0 = np.array([[1, 0],
				[0, 1]])	
		
	lat = kwant.lattice.square(a, norbs = 2)
	sym_syst = kwant.TranslationalSymmetry((-a, 0))
	syst = kwant.Builder(sym_syst)
	
	# initialize by setting zero on-site terms
	syst[(lat(jx,jy) for jx in range (Nx) for jy in range(Ny))] = 0 * sigma_0

	# hoppings and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy

					# set disorder
					syst[lat(jx,jy)] = ((r_1 + r_2)/2) * sigma_z

					# set hoppings 
					if jx < (Nx-1):                
						syst[lat(jx,jy), lat(jx+1,jy)] = (1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z
					
					if jx < (Nx-2):                
						#syst[lat(jx,jy), lat(jx+2,jy)] = - (1/2) * sigma_z
						pass
						
						  							                          							 
					if jy < (Ny-1):    
						syst[lat(jx,jy), lat(jx,jy+1)] = (1/(2j)) * sigma_y - (1/2) * sigma_z
					else:
						syst[lat(jx,Ny-1), lat(jx,0)] = PBC * ((1/(2j)) * sigma_y - (1/2) * sigma_z)
	

	
	syst = syst.finalized()
	kwant.plotter.bands(syst, show=False)
	plt.xlabel("kx")
	plt.ylabel("E")
	plt.title(f"Bands for Nx = {Nx}, Ny = {Ny}, r1 = {r_1}, r2 = {r_2}")
	plt.show()											 
	return syst

	

def plot_system_double_cone_CI(Nx, Ny, r_1, r_2, t_lead, PBC = 1.):
	"""Creates system with the given parameters and then adds random impurities."""
	# values for lattice constant and number of orbitals
	a = 1
	
	# matrices acting in orbital space to build the system
	sigma_x = np.array([[0, 1],
				[1, 0]])
	
	sigma_y = np.array([[0, -1j],
				[1j, 0]])
	
	sigma_z = np.array([[1, 0],
				[0, -1]])
	
	sigma_0 = np.array([[1, 0],
				[0, 1]])	
		
	lat = kwant.lattice.square(a)
	syst = kwant.Builder()

	# initialize by setting zero on-site terms
	syst[(lat(jx,jy) for jx in range (Nx) for jy in range(Ny))] = 0 * sigma_0

	# hoppings and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy

					# set disorder
					syst[lat(jx,jy)] = ((r_1 + r_2)/2) * sigma_z

					# set hoppings 
					if jx < (Nx-1):                
						syst[lat(jx,jy), lat(jx+1,jy)] = (1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z 
					
					if jx < (Nx-2):                
						syst[lat(jx,jy), lat(jx+2,jy)] = - (1/2) * sigma_z
						  							                          							 
					if jy < (Ny-1):    
						syst[lat(jx,jy), lat(jx,jy+1)] = (1/(2j)) * sigma_y - (1/2) * sigma_z
					else:
						syst[lat(jx,Ny-1), lat(jx,0)] = PBC * ((1/(2j)) * sigma_y - (1/2) * sigma_z)
	
	# construct left lead	
	lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
	
	#initialize with on-site terms to avoid error 
	lead[(lat(0,jy) for jy in range(Ny))] = t_lead * sigma_x
	
	# set kinetic terms in lead to obtain 2 t_lead (cos(kx) + cos(ky)) dispersion
	for jy in range(Ny):     
		lead[lat(0,jy), lat(1,jy)] = t_lead * (sigma_x - 1j * sigma_y)/2
			
		if jy < Ny-1:                                
			lead[lat(0,jy), lat(0,jy+1)] =  t_lead * sigma_0
		else:
			lead[lat(0,Ny-1), lat(0,0)] =  PBC * t_lead * sigma_0
	
	# attach left lead 
	#syst.attach_lead(lead)
	
	# reverse left lead and attach it on the right
	#syst.attach_lead(lead.reversed())
	
	lead = lead.finalized()
	kwant.plotter.bands(lead, show=False)
	plt.xlabel("kx")
	plt.ylabel("E")
	plt.title(f"Bands for Nx = {Nx}, Ny = {Ny}, r1 = {r_1}, r2 = {r_2}")
	plt.show()
	
	#kwant.plot(syst)
											 
	return syst


def main():
	Nx = 10
	Ny = 2
		
	r_1 = 1.9
	r_2 = 1.9
	
	PBC = 0
	
	#plot_energy_double_cone_CI(Nx, Ny, r_1, r_2, PBC)
	
	t_lead = 1
	
	plot_system_double_cone_CI(Nx, Ny, r_1, r_2, t_lead, 0)
	
	
if __name__ == '__main__':
	main()




	