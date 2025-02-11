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
                    parameters[current_line[0]] = np.array(current_line[1:], dtype = bool)                
                except:
                    pass
    return parameters
    
    
def contour_plot_conductance(location, dots = False):
    """Contour plot of conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')  

    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    PBC = int(parameters["PBC"][0])
    
    try:
        v = parameters["v"][0]
        title = rf"$v = {v}$, $N_x = {Nx}$, $N_y = {Ny}$"
    except:
        try:
            r_1 = parameters["r_1"][0]
            r_2 = parameters["r_2"][0]
            title = rf"$r_1 = {r_1}$, $r_2 = {r_2}$, $N_x = {Nx}$, $N_y = {Ny}$"
        except:
            r = parameters["r"][0]
            title = rf"$r = {r}$, $N_x = {Nx}$, $N_y = {Ny}$"
    
    title = title + rf", PBC = {PBC}"
        
        

    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel("W", fontsize=30)
    a1.set_ylabel("E", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
    
    n_levels = 1000
    vmin = 0.
    vmax = 4.
    
    if dots:
        scatterplot = a1.scatter(W_array, E_array, marker = "o", c = data, 
                                     label = r"$E_{n}(k_x)$", cmap=plt.cm.magma_r, vmin= vmin, vmax = vmax)
        
        cbar_title = r"$G[e^2/\hbar]$"
        cbar_1 = plt.colorbar(scatterplot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)
    else:
        levels = np.linspace(vmin, vmax, n_levels)
        contour_plot = a1.contourf(W_array, E_array, data, levels = levels, cmap=plt.cm.magma_r, extend = "both")
        
        cbar_title = r"$G[e^2/\hbar]$"
        cbar_1 = plt.colorbar(contour_plot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)  
    
    plt.show()
    

def contour_plot_conductance_diff(location_1, location_2, dots = False):
    """Contour plot difference of conductance between two datasets over W and E"""
    parameters = gen_dictionary(location_1 + "/parameters.txt")
    
    E_vals = np.genfromtxt(location_1 + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location_1 + "/W_vals.txt", dtype= float, delimiter=' ') 
    data_1 = np.genfromtxt(location_1 + "/result.txt", dtype= float, delimiter=' ')  
    data_2 = np.genfromtxt(location_2 + "/result.txt", dtype= float, delimiter=' ')  
    
    data = data_1 - data_2

    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    PBC = int(parameters["PBC"][0])
    
    try:
        v = parameters["v"][0]
        title = rf"$v = {v}$, $N_x = {Nx}$, $N_y = {Ny}$"
    except:
        try:
            r_1 = parameters["r_1"][0]
            r_2 = parameters["r_2"][0]
            title = rf"$r_1 = {r_1}$, $r_2 = {r_2}$, $N_x = {Nx}$, $N_y = {Ny}$"
        except:
            r = parameters["r"][0]
            title = rf"$r = {r}$, $N_x = {Nx}$, $N_y = {Ny}$"
    
    title = title + rf", PBC = {PBC}"
        
    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel("W", fontsize=30)
    a1.set_ylabel("E", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
    
    n_levels = 1000
    vmin = 0.
    vmax = 2
    
    if dots:
        scatterplot = a1.scatter(W_array, E_array, marker = "o", c = data, 
                                     label = r"$E_{n}(k_x)$", cmap=plt.cm.magma_r, vmin= vmin, vmax = vmax)
        
        cbar_title = r"$G[e^2/\hbar]$"
        cbar_1 = plt.colorbar(scatterplot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)
    else:
        levels = np.linspace(vmin, vmax, n_levels)
        contour_plot = a1.contourf(W_array, E_array, data, levels = levels, cmap=plt.cm.magma_r, extend = "both")
        
        cbar_title = r"$G[e^2/\hbar]$"
        cbar_1 = plt.colorbar(contour_plot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)  
    
    plt.show()
    

def main():                
    location_G_F_B = "G_F_B_model_results/F_B_model_run_1"
    location_G_DC_CI = "G_DC_CI_model_results/DC_CI_model_run_41"
    location_G_DC_CI_2 = "G_DC_CI_model_results/DC_CI_model_run_29"
    #location_G_DC_CI = "G_DC_CI_model_results_diff_lead/DC_CI_model_run_2"
    location_G_CI = "G_regular_CI_results/DC_regular_CI_run_12"
    
    dots =  False
    
    contour_plot_conductance(location_G_DC_CI, dots)
    #contour_plot_conductance_diff(location_G_DC_CI, location_G_DC_CI_2, dots)
    
    

if __name__ == '__main__':
    main()




    