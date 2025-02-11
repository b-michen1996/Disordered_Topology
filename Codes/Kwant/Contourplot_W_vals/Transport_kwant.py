import kwant
import numpy as np
import os
import gc


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


def make_system_DC_CI(Nx, Ny, r_1, r_2, W, t_lead, PBC = 1., direction = "x"):
	"""Creates system with the given parameters and then adds random impurities. The argument 
	direction sets the direction of transport."""
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

	# Onsite terms and impurities
	for jx in range(Nx):
			for jy in range(Ny):
					j = jx * Ny + jy					
					syst[lat(jx,jy)] = np.diag(disorder[j,:]) + ((r_1 + r_2)/2) * sigma_z
		
	# Hopping in x-direction
	syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = (1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z 
	syst[kwant.builder.HoppingKind((2, 0), lat, lat)] = - (1/2) * sigma_z
	
	# Hopping in y-direction
	syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = (1/(2j)) * sigma_y - (1/2) * sigma_z


	if direction == "x":
		# Set boundary terms along y-direction
		syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = PBC * ((1/(2j)) * sigma_y - (1/2) * sigma_z)
		
		# construct left lead
		lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
			
		#initialize with on-site terms to avoid error 
		lead[(lat(0,jy) for jy in range(Ny))] = 0 * sigma_0

		# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
		lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = t_lead * sigma_0
							
		# attach left lead 
		syst.attach_lead(lead)

		# reverse left lead and attach it on the right
		syst.attach_lead(lead.reversed())

	elif direction == "y":
		# PBC with flux along x-direction
		syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = PBC * ((1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z)
		syst[kwant.builder.HoppingKind((-(Nx-2), 0), lat, lat)] = -PBC * (1/2) * sigma_z
		
		# construct bottom lead
		lead = kwant.Builder(kwant.TranslationalSymmetry((0, -a)))
			
		#initialize with on-site terms to avoid error 
		lead[(lat(jx,0) for jx in range(Nx))] = 0 * sigma_0

		# set kinetic terms in lead to obtain mono-atomic lattice with simple NN-hoping along x and no hopping along y
		lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = t_lead * sigma_0
							
		# attach left lead 
		syst.attach_lead(lead)

		# reverse left lead and attach it on the right
		syst.attach_lead(lead.reversed())
				
	#kwant.plotter.plot(syst)
											 
	return syst.finalized()


def export_conductance_Fulga_Bergholtz(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, v, t_lead, run_nr, PBC = 1.):
	"""Compute conductance for disorder realizations of the Fulga-Bergholtz model and export"""
	# create folder
	foldername = "G_F_B_model_results/F_B_model_run_" + str(run_nr)
	
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
	
	f_parameter.write("v " + str(v) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	
	f_parameter.write("PBC " + str(PBC) + str("\n"))
			
	f_parameter.close()
					
	result = np.zeros((n_E, n_W))
		
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):
			# generate System
			syst = make_system_Fulga_Bergholtz(Nx, Ny, v, W_curr, t_lead, PBC)
			syst = syst.finalized()
			
			# calculate transmission for all energy values
			data_j_inst = np.zeros(n_E)			
			for j_E in range(n_E):
				energy = E_vals[j_E]
				smatrix = kwant.smatrix(syst, energy)					
				data_j_inst[j_E] = smatrix.transmission(1, 0)
			
			# collect garbage
			del syst
			gc.collect()
			
			print(f"j_inst = {j_inst + 1} of {n_inst}, j_W = {j_W + 1} of {n_W}, W = {W_curr} ")	
						
			result[:, j_W] += (1 / n_inst) * data_j_inst 
		
		np.savetxt(foldername + "/result.txt", result, delimiter=' ')  
								
	return 


def export_conductance_double_cone_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r_1, r_2, t_lead, run_nr, PBC = 1., direction = "x"):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "G_DC_CI_model_results/DC_CI_model_run_" + str(run_nr)

	print(f"Starting DC_CI run {run_nr}")
	
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
	f_parameter.write("direction " + str(direction) + str("\n"))
			
	f_parameter.close()
					
	result = np.zeros((n_E, n_W))
		
	for j_W in range(0, n_W):
		W_curr = W_vals[j_W]		
		for j_inst in range(0, n_inst):
			# generate System
			syst = make_system_DC_CI(Nx, Ny, r_1, r_2, W_curr, t_lead, PBC, direction)			
			
			# calculate transmission for all energy values
			data_j_inst = np.zeros(n_E)			
			for j_E in range(n_E):
				energy = E_vals[j_E]
				smatrix = kwant.smatrix(syst, energy)					
				data_j_inst[j_E] = smatrix.transmission(1, 0)
			
			# collect garbage
			del syst
			gc.collect()
			
			print(f"j_inst = {j_inst + 1} of {n_inst}, j_W = {j_W + 1} of {n_W}, W = {W_curr} ")	
						
			result[:, j_W] += (1 / n_inst) * data_j_inst 
			
			np.savetxt(foldername + "/result.txt", result, delimiter=' ')  		
								
	return 
	

def main():
	W_min = 0.1
	W_max = 4

	E_min = -1.5
	E_max = 1.5
	
	n_W = 39
	n_E = 51
	n_inst = 2
		
	r_1 = 1.9
	r_2 = 1
	
	t_lead = 5
	
	PBC = 1.	
	
	Nx = 50
	Ny = 50
	
	direction = "y"
		
	run_nr = 41

	export_conductance_double_cone_CI(W_min, W_max, E_min, E_max, n_W, n_E, n_inst, Nx, Ny, r_1, r_2, t_lead, run_nr, PBC, direction)
	
	
	
if __name__ == '__main__':
	main()




	
