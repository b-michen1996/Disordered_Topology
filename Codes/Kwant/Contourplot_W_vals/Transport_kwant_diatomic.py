import kwant
import numpy as np
import os
import gc
import sys


def make_system_double_cone_CI_diatomic(Nx, Ny, r_1, r_2, W, t_lead, PBC = 1.):
	"""Creates system with the given parameters and then adds random impurities."""
	# values for lattice constant and number of orbitals
	a = 1
		
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
		
	primitive_vectors = [(1, 0), (0, 1)]
	lat_a = kwant.lattice.Monatomic(primitive_vectors, offset=(0, 0), norbs=1)
	lat_b = kwant.lattice.Monatomic(primitive_vectors, offset=(0, 0.5), norbs=1)
	
	syst = kwant.Builder()
	
	# initialize by setting zero on-site terms
	syst[(lat_a(jx,jy) for jx in range (Nx) for jy in range(Ny))] = 0 
	syst[(lat_b(jx,jy) for jx in range (Nx) for jy in range(Ny))] = 0 
	
	"""
	# hoppings and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy

					# set disorder
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + ((r_1 + r_2)/2) * sigma_z

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
	#lead[(lat(0,jy) for jy in range(Ny))] = t_lead * sigma_x
	lead[(lat(0,jy) for jy in range(Ny))] = 0 * sigma_0
	
	# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
	for jy in range(Ny):     
		lead[lat(0,jy), lat(1,jy)] = t_lead * sigma_0
					
	#initialize with on-site terms to avoid error 
	lead[(lat(0,jy) for jy in range(Ny))] = 0 * sigma_0
	
	# set kinetic terms in lead to obtain 2 t_lead (cos(kx) + cos(ky)) dispersion
	for jy in range(Ny):     
		lead[lat(0,jy), lat(1,jy)] = t_lead * sigma_0
			
		if jy < Ny-1:                                
			lead[lat(0,jy), lat(0,jy+1)] =  t_lead * sigma_0
		else:
			lead[lat(0,Ny-1), lat(0,0)] =  PBC * t_lead * sigma_0
	"""
	"""

	# attach left lead 
	syst.attach_lead(lead)
	
	# reverse left lead and attach it on the right
	syst.attach_lead(lead.reversed())
	"""
	kwant.plotter.plot(syst)
											 
	return syst


def make_system_CI(Nx, Ny, r, W, t_lead, PBC = 1.):
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
	
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	
	lat = kwant.lattice.square(a, norbs = 2)
	syst = kwant.Builder()

	# initialize by setting zero on-site terms
	syst[(lat(jx,jy) for jx in range (Nx) for jy in range(Ny))] = 0 * sigma_0

	# hoppings and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy

					# set disorder
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + r * sigma_z

					# set hoppings 
					if jx < (Nx-1):                
						syst[lat(jx,jy), lat(jx+1,jy)] = (1/(2j)) * sigma_x - (1/2) * sigma_z
															  													  							 
					if jy < (Ny-1):    
						syst[lat(jx,jy), lat(jx,jy+1)] = (1/(2j)) * sigma_y - (1/2) * sigma_z
					else:
						syst[lat(jx,Ny-1), lat(jx,0)] = PBC * ((1/(2j)) * sigma_y - (1/2) * sigma_z)
	
	# construct left lead
	lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
		
	#initialize with on-site terms to avoid error 
	#lead[(lat(0,jy) for jy in range(Ny))] = t_lead * sigma_x
	lead[(lat(0,jy) for jy in range(Ny))] = 0 * sigma_0
	
	# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
	for jy in range(Ny):     
		lead[lat(0,jy), lat(1,jy)] = t_lead * sigma_0
					
	"""
	#initialize with on-site terms to avoid error 
	lead[(lat(0,jy) for jy in range(Ny))] = 0 * sigma_0
	
	# set kinetic terms in lead to obtain 2 t_lead (cos(kx) + cos(ky)) dispersion
	for jy in range(Ny):     
		lead[lat(0,jy), lat(1,jy)] = t_lead * sigma_0
			
		if jy < Ny-1:                                
			lead[lat(0,jy), lat(0,jy+1)] =  t_lead * sigma_0
		else:
			lead[lat(0,Ny-1), lat(0,0)] =  PBC * t_lead * sigma_0
	"""
	
	# attach left lead 
	syst.attach_lead(lead)
	
	# reverse left lead and attach it on the right
	syst.attach_lead(lead.reversed())
	
	#kwant.plotter.plot(syst)
											 
	return syst


def export_conductance_double_cone_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r_1, r_2, t_lead, run_nr, PBC = 1.):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "G_DC_CI_diatomic_results/DC_CI_diatomic_run_" + str(run_nr)
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	# generate and save va
	E_vals = np.linspace(E_min, E_max, n_E)
	W_vals = np.linspace(W_min, W_max, n_W)
	
	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')  
	np.savetxt(foldername + "/W_vals.txt", W_vals, delimiter=' ')  
	
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("W_min " + str(W_min) + str("\n"))
	f_parameter.write("W_max " + str(W_max) + str("\n"))
	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	
	f_parameter.write("n_W " + str(n_W) + str("\n"))
	f_parameter.write("n_E " + str(n_E) + str("\n"))
	f_parameter.write("n_inst " + str(n_inst) + str("\n"))
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))	
	
	f_parameter.write("r_1 " + str(r_1) + str("\n"))	
	f_parameter.write("r_2 " + str(r_2) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	
	f_parameter.write("PBC " + str(PBC) + str("\n"))
			
	f_parameter.close()
					
	result = np.zeros((n_E, n_W))
		
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):
			# generate System
			syst = make_system_double_cone_CI_diatomic(Nx, Ny, r_1, r_2, W_curr, t_lead, PBC)
			syst = syst.finalized()
			
			# calculate transmission for all energy values
			data_j_inst = np.zeros(n_E)			
			for j_E in range(n_E):
				energy = E_vals[j_E]
				smatrix = kwant.smatrix(syst, energy)					
				data_j_inst[j_E] = smatrix.transmission(1, 0)
				sys.stdout.write(f"j_inst = {j_inst + 1} of {n_inst}, j_E = {j_E + 1} of {n_E} \n")
				sys.stdout.flush()
			
			# collect garbage
			del syst
			gc.collect()
			
						
			result[:, j_W] += (1 / n_inst) * data_j_inst 
		
		np.savetxt(foldername + "/result.txt", result, delimiter=' ')  
								
	return 


def export_conductance_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r, t_lead, run_nr, PBC = 1.):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "G_regular_CI_results/DC_regular_CI_run_" + str(run_nr)
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	# generate and save va
	E_vals = np.linspace(E_min, E_max, n_E)
	W_vals = np.linspace(W_min, W_max, n_W)
	
	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')  
	np.savetxt(foldername + "/W_vals.txt", W_vals, delimiter=' ')  
	
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("W_min " + str(W_min) + str("\n"))
	f_parameter.write("W_max " + str(W_max) + str("\n"))
	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	
	f_parameter.write("n_W " + str(n_W) + str("\n"))
	f_parameter.write("n_E " + str(n_E) + str("\n"))
	f_parameter.write("n_inst " + str(n_inst) + str("\n"))
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))	
	
	f_parameter.write("r " + str(r) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	
	f_parameter.write("PBC " + str(PBC) + str("\n"))
			
	f_parameter.close()
					
	result = np.zeros((n_E, n_W))
		
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):
			# generate System
			syst = make_system_CI(Nx, Ny, r, W_curr, t_lead, PBC)
			syst = syst.finalized()
			
			# calculate transmission for all energy values
			data_j_inst = np.zeros(n_E)			
			for j_E in range(n_E):
				energy = E_vals[j_E]
				smatrix = kwant.smatrix(syst, energy)					
				data_j_inst[j_E] = smatrix.transmission(1, 0)
				sys.stdout.write(f"j_inst = {j_inst + 1} of {n_inst}, j_E = {j_E + 1} of {n_E} \n")
				sys.stdout.flush()
			
			# collect garbage
			del syst
			gc.collect()
									
			result[:, j_W] += (1 / n_inst) * data_j_inst 
		
		np.savetxt(foldername + "/result.txt", result, delimiter=' ')  
								
	return 

	

def main():
	W_min = 0
	W_max = 4

	E_min = -2.5
	E_max = 2.5
	
	n_W = 32
	n_E = 50
	n_inst = 3

	Nx = 60
	Ny = 60
		
	r_1 = 1.9
	r_2 = 1
	
	t_lead = 5
	
	PBC = 0.	
	
	run_nr = 21
	
	#export_conductance_double_cone_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r_1, r_2, t_lead, run_nr, PBC)
	
	r1 = 1.9
	r2 = 1.9
	t_lead = 1
	W = 0
	
	#make_system_double_cone_CI_diatomic(Nx, Ny, r_1, r_2, W, t_lead, 0)
	
	r = 2.3
	export_conductance_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r, t_lead, run_nr, 0.)
	
	
if __name__ == '__main__':
	main()




	