import kwant
import numpy as np
import os
import gc
import sys


def make_system_Fulga_Bergholtz(Nx, Ny, v, W, t_lead, PBC = 1.):
	"""Creates system with the given parameters and then adds random impurities."""
	# values for lattice constant and number of orbitals
	a = 1

	f = np.sqrt(2) * np.exp(-1j * np.pi / 4)
	# matrices acting in orbital space to build the system
	M_OS = np.array([[0, 1j * f, v],
				[np.conj(1j * f), 0, 0],
				[v, 0, 0]])
	
	M_DX = np.array([[1, f, 0],
				[0, -1, 0],
				[0, 0, 0]])
	
	M_DY = np.array([[-1, 0, 0],
				[1j * f, 1, 0],
				[0, 0, 0]])
	
	M_DXmDY = np.array([[0, 1j * f, 0],
				[0, 0, 0],
				[0, 0, 0]])
	
	ID_3 = np.identity(3)
	
	# array containing values of random potential
	disorder = W * np.ones((Nx * Ny, 3)) - 2 * W * np.random.random_sample((Nx * Ny, 3))
		
	lat = kwant.lattice.square(a, name = "a")
	syst = kwant.Builder()

	# initialize by setting zero on-site terms
	syst[(lat(jx,jy) for jx in range (Nx) for jy in range(Ny))] = 0 * ID_3

	# hoppings and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy

					# set disorder
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + M_OS

					# set hoppings 
					if jx < (Nx-1):                
						syst[lat(jx,jy), lat(jx+1,jy)] = M_DX
						if jy > 0:    
							syst[lat(jx,jy), lat(jx+1,jy - 1)] = M_DXmDY   
						else:
							syst[lat(jx,jy), lat(jx+1, Ny - 1)] = PBC * M_DXmDY   
													  
							 
					if jy < (Ny-1):    
						syst[lat(jx,jy), lat(jx,jy+1)] = M_DY
					else:
						syst[lat(jx,Ny-1), lat(jx,0)] = PBC * M_DY
	
	# construct left lead
	lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
	
	#initialize with on-site terms to avoid error 
	lead[(lat(0,jy) for jy in range(Ny))] = 0 * ID_3
	
	# set kinetic terms in lead to obtain 2 t_lead (cos(kx) + cos(ky)) dispersion
	for jy in range(Ny):     
		lead[lat(0,jy), lat(1,jy)] = t_lead * ID_3
			
		if jy < Ny-1:                                
			lead[lat(0,jy), lat(0,jy+1)] =  t_lead * ID_3
		else:
			lead[lat(0,Ny-1), lat(0,0)] =  PBC * t_lead * ID_3
	
	# attach left lead 
	syst.attach_lead(lead)
	
	# reverse left lead and attach it on the right
	syst.attach_lead(lead.reversed())
											 
	return syst


def make_system_double_cone_CI(Nx, Ny, r_1, r_2, W, t_lead, PBC = 1.):
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



def export_conductance_CI_single_W(W, E_min, E_max, n_E, n_inst, Nx, Ny, r, t_lead, run_nr, PBC = 1.):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "G_regular_CI_single_W_results/DC_regular_CI_single_W_run_" + str(run_nr)
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	# generate and save va
	E_vals = np.linspace(E_min, E_max, n_E)
	
	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')  
	
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("W " + str(W) + str("\n"))
	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	
	f_parameter.write("n_E " + str(n_E) + str("\n"))
	f_parameter.write("n_inst " + str(n_inst) + str("\n"))
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))	
	
	f_parameter.write("r " + str(r) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	
	f_parameter.write("PBC " + str(PBC) + str("\n"))
			
	f_parameter.close()
					
	result = np.zeros(n_E)
		
	
	for j_inst in range(0, n_inst):
		# generate System
		syst = make_system_CI(Nx, Ny, r, W, t_lead, PBC)
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
							
	result = (1 / n_inst) * data_j_inst 
	
	np.savetxt(foldername + "/result.txt", result, delimiter=' ')  
								
	return 


def export_conductance_Fulga_Bergholtz_single_W(W, E_min, E_max, n_E, n_inst, Nx, Ny, v, t_lead, run_nr, PBC = 1.):
	"""Compute conductance for disorder realizations of the Fulga-Bergholtz model and export"""
	# create folder
	foldername = "G_F_B_model_single_W_results/F_B_model_single_W_run_" + str(run_nr)
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	# generate and save va
	E_vals = np.linspace(E_min, E_max, n_E)
		
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("W " + str(W) + str("\n"))
	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	
	f_parameter.write("n_E " + str(n_E) + str("\n"))
	f_parameter.write("n_inst " + str(n_inst) + str("\n"))
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))	
	
	f_parameter.write("v " + str(v) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	
	f_parameter.write("PBC " + str(PBC) + str("\n"))
			
	f_parameter.close()
					
	result = np.zeros(n_E)
				
	for j_inst in range(0, n_inst):
		# generate System
		syst = make_system_Fulga_Bergholtz(Nx, Ny, v, W, t_lead, PBC)
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
							
		result += (1 / n_inst) * data_j_inst 
	
	np.savetxt(foldername + "/result.txt", result, delimiter=' ')  
								
	return 



def export_conductance_double_cone_CI_single_W(W, E_min, E_max, n_E, n_inst, Nx, Ny, r_1, r_2, t_lead, run_nr, PBC = 1.):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "G_DC_CI_model_single_W_results/DC_CI_model_single_W_run_" + str(run_nr)
	
	try:
		os.makedirs(foldername)
	except:
		pass 
	
	# generate and save va
	E_vals = np.linspace(E_min, E_max, n_E)
	
	np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')  

	
	# export parameters
	f_parameter = open(foldername + "/parameters.txt", "w")
	
	f_parameter.write("W " + str(W) + str("\n"))
	f_parameter.write("E_min " + str(E_min) + str("\n"))
	f_parameter.write("E_max " + str(E_max) + str("\n"))
	
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
					
	result = np.zeros(n_E)
		
		
	for j_inst in range(0, n_inst):
		# generate System
		syst = make_system_double_cone_CI(Nx, Ny, r_1, r_2, W, t_lead, PBC)
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
							
		result += (1 / n_inst) * data_j_inst 
	
	np.savetxt(foldername + "/result.txt", result, delimiter=' ')  
								
	return 
	

def main():
	W = 0.5

	E_min = -0.2
	E_max = 0.2
	
	n_E = 31
	n_inst = 10

	Nx = 180
	Ny = 60
		
	r_1 = 1.99
	r_2 = 1
	
	t_lead = 5
	
	PBC = 0.	
	
	run_nr = 2
	
	export_conductance_double_cone_CI_single_W(W, E_min, E_max, n_E, n_inst, Nx, Ny, r_1, r_2, t_lead, run_nr, PBC)
	
	
	
	
if __name__ == '__main__':
	main()




	