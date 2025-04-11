import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors


# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})


def gen_dictionary(path):
    """Generate dictionary with parameters from file."""
    parameters = {}
    
    with open(path) as f:        
        for line in f:            
            current_line = line.split()            
            try: 
                parameters[current_line[0]] = np.array(current_line[1:], dtype = float)
            except:
                try: 
                    parameters[current_line[0]] = np.array(current_line[1:], dtype = str)                
                except:
                    pass
    return parameters
           


def title_generator_contour_plot(parameters):
    """Generate title from parameters"""
    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    x_l_1= int(parameters["x_lead_start"][0])
    x_l_2= int(parameters["x_lead_stop"][0])
    y_l_1= int(parameters["y_lead_start"][0])
    y_l_2= int(parameters["y_lead_stop"][0])
    
    try:
        r = parameters["r"][0]
        epsilon_1 = parameters["epsilon_1"][0]
        epsilon_2 = parameters["epsilon_2"][0]
        gamma = parameters["gamma"][0]
        gamma_2 = parameters["gamma_2"][0]
                
        title = title = rf"$r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $\gamma_2$ = {gamma_2}, $N_x = {Nx}$, $N_y = {Ny}$, $x_l = ({x_l_1}, {x_l_2})$, $y_l = ({y_l_1}, {y_l_2})$"
        
    except:
   
        r = parameters["r"][0]
        epsilon_1 = parameters["epsilon_1"][0]
        epsilon_2 = parameters["epsilon_2"][0]
        gamma = parameters["gamma"][0]
                
        title = title = rf"$r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$,$\gamma$ = {gamma}, $N_x = {Nx}$, $N_y = {Ny}$, $x_l = ({x_l_1}, {x_l_2})$, $y_l = ({y_l_1}, {y_l_2})$"
   
                
    return title


def get_conductance(cond_mat):
	"""Get elements of the conductance tensor from 4 lead conduction matrix for current between lead 0 and 1 and lead 2 and 3. Lead 0 to 1. 
	Starting counterclockwise from the left edge of a square sample, the leads are assumed to be attached in the order 0,1,2,3."""
	
	# Set V_3 = 0 and solve I = G V for I_0 = - I_1 = 1 and I_2 = 0, which implies I_3 = 0. This yields
	# V_0, V_1, and V_2, from which the conductance elements are calculated as as sigma_xy = I_0 / (V_3 - V_2) = -1 / V_2 
	# and sigma_xx = I_0 / (V_0 - V_1)
	cond_mat_truc_i = cond_mat[:-1, :-1]
	V_i = np.linalg.solve(cond_mat_truc_i, [1,-1, 0])   
	sigma_xx = 1 / (V_i[0] - V_i[1])
	sigma_xy = -1 / V_i[2]
	
	
	# Set V_0 = 0 and solve I = G V for I_2 = - I_3 = 1 and I_1 = 0, which implies I_0 = 0. This yields
	# V_1, V_2, and V_3, from which the transverse conductance is calculated as as sigma_yx = I_2 / (V_1 - V_0) = -1 / V_1 and
	# sigma_yy = I_2 / (V_2 - V_3)
	cond_mat_truc_ii = cond_mat[1:, 1:]
	V_ii = np.linalg.solve(cond_mat_truc_ii, [0, 1,-1])
	sigma_yy = 1 / (V_ii[1] - V_ii[2])
	sigma_yx = -1 / V_ii[0]
	
	return sigma_xx, sigma_xy, sigma_yx, sigma_yy


def contour_plot_Hall_from_cond_matrix(location, dots = False):
    """Contour plot for Hall conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    
    data_result = np.load(location + "/result.npy")
        
    n_E = len(E_vals)
    n_W = len(W_vals)
    
    data_sigma_xy = np.zeros((n_E, n_W))
    data_sigma_yx = np.zeros((n_E, n_W))
            
    # get transverse resitance
    for j_E in range(n_E):
        for j_W in range(n_W):
            cond_mat_curr = data_result[j_E, :, :, j_W]
            try:
                sigma_xx, sigma_xy, sigma_yx, sigma_yy = get_conductance(cond_mat_curr)
                data_sigma_xy[j_E, j_W] = abs(sigma_xy)
                data_sigma_yx[j_E, j_W] = abs(sigma_yx)
            except:
                print("Couldn't solve!")
                    
    data_sets = [data_sigma_xy, data_sigma_yx]     
    
    W_array, E_array = np.meshgrid(W_vals, E_vals)
    
    title = title_generator_contour_plot(parameters)
            
    fig = plt.figure(figsize=(12, 12), layout = "tight")
    fig.suptitle(title, fontsize = 20)
    
    n_levels = 1000
    vmin = 0.
    vmax = 2
    
    colormap = plt.cm.magma_r
    #colormap = plt.cm.hot.reversed()
    lead_names = ["x", "y"] 
    cond_entries = ["xy", "yx"] 
    for l in range(0, 2):
        lead_nr = l
        lead_name = lead_names[lead_nr] 
        al =  fig.add_subplot(1, 2, l + 1)
        al.set_xlabel("W", fontsize=20)
        al.set_ylabel("E", fontsize=20)
        al.set_title(f"Current in {lead_name}-direction", fontsize=20)
        al.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   
        
        data_curr = data_sets[l]

        if dots:
            scatterplot = al.scatter(W_array, E_array, marker = "o", c = data_curr,                                      
                                         cmap=colormap, vmin= vmin, vmax = vmax)
            cbar_title = r"$\sigma_{" + cond_entries[l] + "} [\hbar / e^2]$"
            cbar = plt.colorbar(scatterplot)
            cbar.ax.set_title(cbar_title, fontsize = 20)
            
        else:
            levels = np.linspace(vmin, vmax, n_levels)
            contour_plot = al.contourf(W_array, E_array, data_curr, levels = levels, cmap=colormap, extend = "both")
            
            cbar_title = r"$\sigma_{" + cond_entries[l] + "} [\hbar / e^2]$"
            cbar = plt.colorbar(contour_plot)
            cbar.ax.set_title(cbar_title, fontsize = 20) 
        
    plt.show()


def cond_plot_from_files(location, dots = False):
    """Contour plot for Hall conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    
    data_result = np.load(location + "/result.npy")
        
    n_E = len(E_vals)
    n_W = len(W_vals)
    
    sigma_indices = ["xx", "xy", "yx", "yy"]
    
    data_sets = list()
    
    for sigma_index in sigma_indices:
        data_curr = np.genfromtxt(location + "/result_sigma_" + sigma_index + ".txt", dtype= float, delimiter=' ')          
        #data_sets.append(data_curr)  
        data_sets.append(np.abs(data_curr))  
    
    W_array, E_array = np.meshgrid(W_vals, E_vals)
    title = title_generator_contour_plot(parameters)
            
    fig = plt.figure(figsize=(24, 24), layout = "tight")
    fig.suptitle(title, fontsize = 20)
    
    n_levels = 1000
    vmin = 0.
    vmax = 2
    
    for l in range(0, 4):
        al =  fig.add_subplot(2, 2, l + 1)
        al.set_xlabel("W", fontsize=30)
        al.set_ylabel("E", fontsize=30)
        al.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   
        
        data_curr = data_sets[l]
        
        sigma_index = sigma_indices[l]
        
        if dots:
            scatterplot = al.scatter(W_array, E_array, marker = "o", c = data_curr, 
                                         cmap=plt.cm.magma_r, vmin= vmin, vmax = vmax)
            
            cbar_title = r"$\sigma_{" + sigma_index + "}[e^2/\hbar]$"
            cbar = plt.colorbar(scatterplot)
            cbar.ax.set_title(cbar_title, fontsize = 20)
        else:
            levels = np.linspace(vmin, vmax, n_levels)
            contour_plot = al.contourf(W_array, E_array, data_curr, levels = levels, cmap=plt.cm.magma_r, extend = "both")
            
            cbar_title = r"$\sigma_{" + sigma_index + "}[e^2/\hbar]$"
            cbar = plt.colorbar(contour_plot)
            cbar.ax.set_title(cbar_title, fontsize = 20)  
        
    plt.show()
    
    return
    
    
def main():                
    location_Hall_cond_DC_CI_V2 = "G_DC_CI_V2_Hall_cond_results/G_DC_CI_V2_Hall_cond_run_10"    
    location_Hall_cond_DC_CI_V3 = "G_DC_CI_V3_Hall_cond_results/G_DC_CI_V3_Hall_cond_run_2"    
    dots = False

    #contour_plot_Hall_from_cond_matrix(location_Hall_cond, dots)
    cond_plot_from_files(location_Hall_cond_DC_CI_V2, dots)
    
    

if __name__ == '__main__':
    main()




    