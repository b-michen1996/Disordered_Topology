import numpy as np
from matplotlib import pyplot as plt


# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams.update({
    "text.usetex": True,
    #"font.family": "cm"
    "font.family": "times"
})


def sigma_xy_clean(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3 = 0):
	"""Integral of Berry curvature up to Fermi energy, obtained from analytical Berry curvature discretized on N x N lattice."""
	Omega_vals_lower_band, C_n, k_x_vals, k_y_vals, energies = Berry_curvature(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3)
			
	# Flatten data for Berry curvature and energy and multiply by prefactor  delta_k^2 / (2 pi)
	prefactor = 2 * np.pi / (N ** 2)
	 
	Integration_data_lower = prefactor * Omega_vals_lower_band.flatten()
	energy_data_lower = energies[:,:,0].flatten()
	Integration_data_upper = - Integration_data_lower
	energy_data_upper = energies[:,:,1].flatten()
	
	# get index to sort by energy
	idx_lower = np.argsort(energy_data_lower)
	idx_upper = np.argsort(energy_data_upper)
	
	Integration_data_lower = Integration_data_lower[idx_lower]
	energy_data_lower = energy_data_lower[idx_lower]
	Integration_data_upper = Integration_data_upper[idx_upper]
	energy_data_upper = energy_data_upper[idx_upper]
	
	# Integrate
	for j in range(1, N**2):
		Integration_data_lower[j] += Integration_data_lower[j-1]
		Integration_data_upper[j] += Integration_data_upper[j-1]
			
	# Some values of E occur multiple times, throw them out
	final_integral_lower = [Integration_data_lower[0]]
	final_energies_lower = [energy_data_lower[0]]
	
	for j in range(1, N**2):
		if (energy_data_lower[j] -  final_energies_lower[-1]) < 10 ** (-8):
			final_integral_lower[-1] = Integration_data_lower[j]
			final_energies_lower[-1] = energy_data_lower[j]
		else:
			final_integral_lower.append(Integration_data_lower[j])
			final_energies_lower.append(energy_data_lower[j])
			
	final_integral_upper = [Integration_data_upper[0]]
	final_energies_upper = [energy_data_upper[0]]
	
	for j in range(1, N**2):
		if (energy_data_upper[j] -  final_energies_upper[-1]) < 10 ** (-8):
			final_integral_upper[-1] = Integration_data_upper[j]
			final_energies_upper[-1] = energy_data_upper[j]
		else:
			final_integral_upper.append(Integration_data_upper[j])
			final_energies_upper.append(energy_data_upper[j])
	
		
	return final_integral_lower, final_integral_upper, final_energies_lower, final_energies_upper, energies[:,:,0], energies[:,:,1]


def Berry_curvature(N, r, epsilon_1, epsilon_2, gamma, gamma_2, gamma_3 = 0.):
	"""Calculate berry curvature of H_bloch on a discrete grid of k_x, k_y values. """
	delta_k = 2 * np.pi / N 
	k_vals = np.linspace(-np.pi + delta_k/2, np.pi + delta_k/2, N, endpoint= False)
	
	kx_array, ky_array = np.meshgrid(k_vals,k_vals)

	Phi_vals =  np.zeros((N,N))
	
	energies = np.zeros((N,N,2))
	
	# calculate Berry curvature
	for jx in range(N):
		for jy in range(N):
			kx = kx_array[jy, jx]
			ky = ky_array[jy, jx]
			# calculate flux through plaquette
			Phi, energy_m, energy_p = Omega(kx, ky, r, epsilon_1, epsilon_2, gamma, gamma_2, band = -1, gamma_3 = gamma_3)
			Phi_vals[jy, jx] = Phi
			energies[jy, jx, 0] = energy_m
			energies[jy, jx, 1] = energy_p
		
	# calculate chern number for each band
	C_n = (2 * np.pi / (N ** 2)) * np.sum(Phi_vals) 

	# give back results and arrays of k_vals
	return Phi_vals, C_n, kx_array, ky_array, energies


def Omega(kx, ky, r, epsilon_1, epsilon_2, gamma, gamma_2, band = -1, gamma_3 = 0):
	"""Calculate Berry curvature at kx, ky"""
	s_x = np.sin(kx)
	s_2x = np.sin(2 * kx)
	s_y = np.sin(ky)
	c_x = np.cos(kx)
	c_2x = np.cos(2 * kx)
	c_y = np.cos(ky)	
	lambda_kx = (epsilon_1 + epsilon_2 * (1 - c_x)/2)
				
	E = np.sqrt((gamma * s_x) ** 2 + (lambda_kx * s_y)**2 + (gamma_2 * (r - c_2x) - lambda_kx * c_y) ** 2)
	
	res = lambda_kx * (epsilon_2 * gamma * s_x ** 2 / 2 - 2 * gamma * gamma_2 * s_x * s_2x * c_y 
					+ gamma * gamma_2 * (r - c_2x) * c_x * c_y - gamma * lambda_kx * c_x )
	
	res = - band * res / (2 * (E ** 3))
	
	d0 = gamma_3 * np.cos(ky)
	
	return res, d0 - E, d0 + E


def Fig_4(location, r, gamma, gamma_2, epsilon_1, epsilon_2, N = 200, add_Q = True):
    """Plot gap and topological index for the spectral localizer."""    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')      
    bulk_energies = np.genfromtxt(location + "/bulk_energies.txt", dtype= float, delimiter=' ')      
    data_sl_gap = np.genfromtxt(location + "/result_sl_gap.txt", dtype= float, delimiter=' ')      
                
    try: 
        data_sl_Q_index = np.genfromtxt(location + "/result_sl_Q_index.txt", dtype= float, delimiter=' ')      
        L_data_gap = len(data_sl_gap)
        
        for l in range(L_data_gap):
            int_dev = np.abs(data_sl_Q_index[l, 1] - np.round(data_sl_Q_index[l, 1]))
            
            if int_dev > 0.01:
                data_sl_gap[l] = 0
    except:
        pass
    
    fig, a1 = plt.subplots(1, 1, figsize =(12, 6))
        
    # indicate sigma_xy = 0.5
    I_low, I_up, E_low, E_up, energies_band_low, energies_band_up  = sigma_xy_clean(N, r, epsilon_1, epsilon_2, gamma, gamma_2, 0)
    
    I_low = np.array(I_low)
    I_up = np.array(I_up)  
    E_low = np.array(E_low)
    E_up = np.array(E_up)  
    
    mask_I_low = I_low > 0.5
    mask_I_up = I_up > 0.5
    
    N_low = len(E_low[mask_I_low])
    N_up = len(E_low[mask_I_up])
        
    a1.fill_betweenx(E_low[mask_I_low], np.zeros(N_low), np.ones(N_low), facecolor = "gray", lw = 0, alpha = 0.2)
    a1.fill_betweenx(E_up[mask_I_up], np.zeros(N_up), np.ones(N_up), facecolor = "gray", lw = 0, alpha = 0.2, label = r"$\sigma_{xy}^{W = 0} > 0.5 e^2 / h$")
           
    label_size = 25
        
    a1.set_xlabel(r"$g_L$", fontsize=label_size)
    a1.set_ylabel(r"$E$", fontsize=label_size, rotation='horizontal')
    
    a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
    
    a1.set_xticks([0, 0.025, 0.05])
    a1.set_xticklabels([r"$0$", r"$0.025$", r"$0.05$"])
    a1.set_yticks([-3, -2, -1, 0., 1, 2, 3])
    a1.set_yticklabels([r"$-3$", "","", r"$0$","","", r"$3$"])
    
    a1.xaxis.set_label_coords(0.87, -0.03)
    a1.yaxis.set_label_coords(-0.03, 0.86)

    x_min_1 = 0

    y_min = np.min(E_vals)
    y_max = np.max(E_vals)

    a1.set_xlim([0, 0.05])
    a1.set_ylim([y_min, y_max])
    
    a1.plot(data_sl_gap, E_vals, linestyle = "-", color = "black", label = r"$g_L$")

    a1.scatter(x_min_1 * np.ones(len(bulk_energies)), bulk_energies, marker = "o", color = "red", label = r"Bulk energies")
    
    a1.legend(loc="upper left", bbox_to_anchor=(0.58, 0.9), fontsize = label_size, facecolor='white', framealpha=1)
    
    if add_Q:
        a1.text(0.05, 0.05, r"$Q = 0$", color = "Black", fontsize = 0.75 * label_size, transform=a1.transAxes, va = "center_baseline")
        a1.text(0.05, 0.5, r"$Q = 0$", color = "Black", fontsize = 0.75 * label_size, transform=a1.transAxes, va = "center_baseline")
        a1.text(0.05, 0.95, r"$Q = 0$", color = "Black", fontsize = 0.75 * label_size, transform=a1.transAxes, va = "center_baseline")
        
        a1.text(0.05, 0.34, r"$Q = -1$", color = "Black", fontsize = 0.75 * label_size, transform=a1.transAxes, va = "center_baseline")
        a1.text(0.05, 0.67, r"$Q = -1$", color = "Black", fontsize = 0.75 * label_size, transform=a1.transAxes, va = "center_baseline")
    
    fig.savefig("Fig_4_publication.png",  bbox_inches='tight', dpi = 600)
            
    plt.show()


def main():                
    location = "Data_Spectral_localizer"    
    
    r = 1.5
    epsilon_1 = 0.3
    epsilon_2 = 2

    gamma = 2
    gamma_2 = 0.3
    
    Fig_4(location, r, gamma, gamma_2, epsilon_1, epsilon_2)

if __name__ == '__main__':
    main()





    