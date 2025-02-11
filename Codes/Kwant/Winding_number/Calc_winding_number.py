import kwant
import numpy as np
import os
import gc
import sys


def make_system_DC_CI_flux(Nx, Ny, r_1, r_2, disorder, t_lead, direction = "x"):
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
	#disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
	
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
	
	# Flux hopping around boundary defined as function to make it variable
	def y_hop_phi(site1, site2, phi):
		res = np.exp(1j * phi) * ((1/(2j)) * sigma_y - (1/2) * sigma_z)
		return res 
	
	def x_hop_phi_NN(site1, site2, phi):
		res = np.exp(1j * phi) * ((1/(2j)) * sigma_x + ((r_1 - r_2)/4) * sigma_z)
		return res 
	
	def x_hop_phi_NNN(site1, site2, phi):
		res = -np.exp(1j * phi) * (1/2) * sigma_z
		return res 
	
	if direction == "x":
		# PBC with flux along y-direction
		syst[kwant.builder.HoppingKind((0, -(Ny-1)), lat, lat)] = y_hop_phi
		
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
		syst[kwant.builder.HoppingKind((-(Nx-1), 0), lat, lat)] = x_hop_phi_NN
		syst[kwant.builder.HoppingKind((-(Nx-2), 0), lat, lat)] = x_hop_phi_NNN
		
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


def winding_number(syst, E, n_phi):
	"""Calculate winding number for syst at energy E."""
	phi_vals = np.linspace(0, 2 * np.pi, n_phi)
	phase_vals = np.zeros(n_phi)
	
	# dictionary with all parameters that are needed for the functions appearing in syst, here only one is needed
	p= {"phi" : phi_vals[0]}
	
	# calculate first datapoint
	r_matrix_0 = kwant.smatrix(syst, E, params=p).submatrix(0, 0)
	phase_r_matrix_0 = np.angle(np.linalg.det(r_matrix_0))
	phase_vals[0] = phase_r_matrix_0

	sum_phases =  0
	
	for j in range(1, n_phi):	
		# Update value in dictionary
		p["phi"] = phi_vals[j]
		
		# Get reflection matrix 
		r_matrix = kwant.smatrix(syst, E, params=p).submatrix(0, 0)
		
		# get complex angle of determinant 
		phase_r_matrix = np.angle(np.linalg.det(r_matrix))
		phase_vals[j] = phase_r_matrix
		
		d_phase = phase_vals[j] - phase_vals[j-1]
		
		# Calculate integration step, take care that angle may jump
		if d_phase > 1.5:
			d_phase = phase_vals[j] - phase_vals[j-1] - 2 * np.pi
		if d_phase < -1.5:
			d_phase = phase_vals[j] - phase_vals[j-1] + 2 * np.pi
		
		sum_phases +=d_phase
	
	return sum_phases / (2 * np.pi)
		

def export_winding_number_DC_CI(W, E_min, E_max, n_E, n_inst, n_phi, Nx, Ny, r_1, r_2, t_lead, run_nr, direction = "x"):
	"""Compute conductance for disorder realizations of the double cone CI model and export"""
	# create folder
	foldername = "Winding_number_DC_CI/Winding_number_DC_CI_run_" + str(run_nr)
	
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
	f_parameter.write("n_phi " + str(n_phi) + str("\n"))
	
	f_parameter.write("Nx " + str(Nx) + str("\n"))	
	f_parameter.write("Ny " + str(Ny) + str("\n"))	
	
	f_parameter.write("r_1 " + str(r_1) + str("\n"))	
	f_parameter.write("r_2 " + str(r_2) + str("\n"))	
	
	f_parameter.write("t_lead " + str(t_lead) + str("\n"))	
	f_parameter.write("run_nr " + str(run_nr) + str("\n"))	
	f_parameter.write("direction " + str(direction) + str("\n"))	
				
	f_parameter.close()
					
	result = np.zeros((n_E, n_inst))	
		
	for j_inst in range(0, n_inst):
		# generate system
		disorder = W * np.ones((Nx * Ny, 2)) - 2 * W * np.random.random_sample((Nx * Ny, 2))
		syst = make_system_DC_CI_flux(Nx, Ny, r_1, r_2, disorder, t_lead, direction)
				
		# calculate winidng number for all energy values	
		for j_E in range(n_E):
			energy = E_vals[j_E]
			w_num = winding_number(syst, energy, n_phi)			
			result[j_E, j_inst] = w_num
			np.savetxt(foldername + "/result.txt", result, delimiter=' ')  
			
			sys.stdout.write(f"j_inst = {j_inst + 1} of {n_inst}, j_E = {j_E + 1} of {n_E} \n")
			sys.stdout.flush()
					
		# collect garbage
		del syst
		gc.collect()
													
	return 

	

def main():
	W = 1.5

	E_min = 0
	E_max = 1.5
	
	n_E = 31
	n_inst = 2

	Nx = 200
	Ny = 200
		
	r_1 = 1.9
	r_2 = 1
	
	t_lead = 5
	
	n_phi = 30	
	
	run_nr = 14
	
	direction = "y"
		
	export_winding_number_DC_CI(W, E_min, E_max, n_E, n_inst, n_phi, Nx, Ny, r_1, r_2, t_lead, run_nr, direction)

	
if __name__ == '__main__':
	main()




	