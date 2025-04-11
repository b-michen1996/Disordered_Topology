import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib
import json

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
                    parameters[current_line[0]] = np.array(current_line[1:], dtype = bool)                
                except:
                    pass
    return parameters
    

def energy_plot_spec_loc_dense(location):
    """Plot gap and topological index for the spectral localizer."""
    parameters = gen_dictionary(location + "/parameters.txt")

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    W = parameters["W"][0]
    kappa = parameters["kappa"][0]
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')      
    bulk_energies = np.genfromtxt(location + "/bulk_energies.txt", dtype= float, delimiter=' ')      
    data_sl_gap = np.genfromtxt(location + "/result_sl_gap.txt", dtype= float, delimiter=' ')      
    data_sl_Q_index = np.genfromtxt(location + "/result_sl_Q_index.txt", dtype= float, delimiter=' ')      
    
    r = parameters["r"][0]
    epsilon_1 = parameters["epsilon_1"][0]
    epsilon_2 = parameters["epsilon_2"][0]
    gamma = parameters["gamma"][0]
    gamma_2 = parameters["gamma_2"][0]
    
    title = rf"$g_\lambda(E)$ for $W = {W}$, $r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $\gamma_2$ = {gamma_2}, $\kappa = {kappa}$, $N_x = {Nx}$, $N_y = {Ny}$"
    
    fig = plt.figure(figsize=(24, 12), layout = "tight")
    fig.suptitle(title, fontsize = 20)
    
    a1 =  fig.add_subplot(1,2,1)
    a1.set_xlabel(r"$g_\lambda(E)$", fontsize=30)
    a1.set_ylabel("E", fontsize=30)

    x_min_1 = 0
    x_max_1 = np.max(data_sl_gap)

    y_min = np.min(E_vals)
    y_max = np.max(E_vals)

    a1.set_xlim([x_min_1, x_max_1])
    a1.set_ylim([y_min, y_max])
    
    a1.plot(data_sl_gap, E_vals, linestyle = "--", color = "black")

    a1.scatter(x_min_1 * np.ones(len(bulk_energies)), bulk_energies, marker = "o", color = "red")
    
    a2 =  fig.add_subplot(1,2,2)
    a2.set_xlabel(r"$Q$", fontsize=30)
    a2.set_ylabel("E", fontsize=30)
    
    a2.axvline(0, color = "black")

    x_min_2 = 1.2 * np.min(data_sl_Q_index[:,1])
    x_max_2 = 1.2 * np.max(data_sl_Q_index[:,1])

    a2.set_xlim([x_min_2, x_max_2])
    a2.set_ylim([y_min, y_max])
    
    a2.scatter(data_sl_Q_index[:,1], data_sl_Q_index[:,0], marker = "o", color = "black")

    #a2.scatter(x_min_1 * np.ones(len(bulk_energies)), bulk_energies, marker = "o", color = "red")
            
    plt.show()



def energy_plot_spec_loc(location):
    """Plot gap for the spectral localizer."""
    parameters = gen_dictionary(location + "/parameters.txt")

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    W = parameters["W"][0]
    kappa = parameters["kappa"][0]
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')      
    bulk_energies = np.genfromtxt(location + "/bulk_energies.txt", dtype= float, delimiter=' ')      
    data_sl_gap = np.genfromtxt(location + "/result_sl_gap.txt", dtype= float, delimiter=' ')      
    
      
    r = parameters["r"][0]
    epsilon_1 = parameters["epsilon_1"][0]
    epsilon_2 = parameters["epsilon_2"][0]
    gamma = parameters["gamma"][0]
    gamma_2 = parameters["gamma_2"][0]
    
    title = rf"$g_\lambda(E)$ for $W = {W}$, $r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $\gamma_2$ = {gamma_2}, $\kappa = {kappa}$, $N_x = {Nx}$, $N_y = {Ny}$"
    
    fig = plt.figure(figsize=(24, 12), layout = "tight")

    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title)
    a1.set_xlabel(r"$g_\lambda(E)$", fontsize=30)
    a1.set_ylabel("E", fontsize=30)

    x_min_1 = 0
    x_max_1 = np.max(data_sl_gap)

    y_min = np.min(E_vals)
    y_max = np.max(E_vals)

    a1.set_xlim([x_min_1, x_max_1])
    a1.set_ylim([y_min, y_max])
    
    a1.plot(data_sl_gap, E_vals, linestyle = "--", color = "black")

    a1.scatter(x_min_1 * np.ones(len(bulk_energies)), bulk_energies, marker = "o", color = "red")
            
    plt.show()
    

def contour_plot_spec_loc(location, dots = False):
    """Contour plot for the gap of the spectral localizer."""
    parameters = gen_dictionary(location + "/parameters.txt")

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')      
    Kappa_vals = np.genfromtxt(location + "/Kappa_vals.txt", dtype= float, delimiter=' ') 
    K_array, E_array = np.meshgrid(Kappa_vals, E_vals)
    
    data_sl_gap = np.genfromtxt(location + "/result_sl_gap.txt", dtype= float, delimiter=' ')      
      
    r = parameters["r"][0]
    epsilon_1 = parameters["epsilon_1"][0]
    epsilon_2 = parameters["epsilon_2"][0]
    gamma = parameters["gamma"][0]
    gamma_2 = parameters["gamma_2"][0]
    
    title = rf"$g_\lambda(E)$ for $r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $\gamma_2$ = {gamma_2}, $N_x = {Nx}$, $N_y = {Ny}$"
        
    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title)
    a1.set_xlabel(r"$\kappa$", fontsize=30)
    a1.set_ylabel("E", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
    
    n_levels = 1000
    vmin = 0.
    vmax = 0.1
    
    if dots:
        scatterplot = a1.scatter(K_array, E_array, marker = "o", c = data_sl_gap, 
                                     label = r"$E_{n}(k_x)$", cmap=plt.cm.magma_r, vmin= vmin, vmax = vmax)
        
        cbar_title = r"$g_\lambda$"
        cbar_1 = plt.colorbar(scatterplot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)
    else:
        levels = np.linspace(vmin, vmax, n_levels)
        contour_plot = a1.contourf(K_array, E_array, data_sl_gap, levels = levels, cmap=plt.cm.magma_r, extend = "both")
        
        cbar_title = r"$g_\lambda$"
        cbar_1 = plt.colorbar(contour_plot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)  
    
    plt.show()

def main():                
    location_spec_loc_final = "s_l_final_energy_result/s_l_final_energy_run_1"
    location_spec_loc_final_contour = "s_l_final_contour_result/s_l_final_contour_run_1"
    
    #energy_plot_spec_loc(location_spec_loc_final)
    #energy_plot_spec_loc_dense(location_spec_loc_final)
    dots = False
    
    contour_plot_spec_loc(location_spec_loc_final_contour, dots)

if __name__ == '__main__':
    main()




    