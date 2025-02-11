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
    

def contour_plot_spec_loc(location):
    """Contour plot for spectral localizer."""
    parameters = gen_dictionary(location + "/parameters.txt")

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    W = parameters["W"][0]
    v = parameters["v"][0]
    E = parameters["E"][0]

    X_array = np.genfromtxt(location + "/X_array.txt", dtype= float, delimiter=' ') 
    Y_array = np.genfromtxt(location + "/Y_array.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')      

    title = rf"$g_\lambda(x,y, E)$ for $v = {v}$, $W = {W}$, $N_x = {Nx}$, $N_y = {Ny}$ at $E = {E}$"

    fig = plt.figure(figsize=(24, 12), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel("x", fontsize=30)
    a1.set_ylabel("y", fontsize=30)

    contour_plot = a1.contourf(X_array, Y_array, data, levels = 1000, vmin = 0, vmax = 0.05, cmap=plt.cm.magma_r)

    fig.colorbar(contour_plot)
    plt.show()



def energy_plot_spec_loc(location):
    """Contour plot for spectral localizer."""
    parameters = gen_dictionary(location + "/parameters.txt")

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    W = parameters["W"][0]
    v = parameters["v"][0]

    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')      
    bulk_energies = np.genfromtxt(location + "/bulk_energies.txt", dtype= float, delimiter=' ')      
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')      

    title = rf"$g_\lambda(E)$ for $v = {v}$, $W = {W}$, $N_x = {Nx}$, $N_y = {Ny}$"

    fig = plt.figure(figsize=(24, 12), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel(r"$g_\lambda(E)$", fontsize=30)
    a1.set_ylabel("E", fontsize=30)

    x_min = 0
    x_max = np.max(data)

    y_min = np.min(E_vals)
    y_max = np.max(E_vals)

    a1.set_xlim([x_min, x_max])
    a1.set_ylim([y_min, y_max])
    
    a1.plot(data, E_vals, linestyle = "--", color = "black")

    a1.scatter(x_min * np.ones(len(bulk_energies)), bulk_energies, marker = "o", color = "red")
    
    
    #contour_plot = a1.contourf(X_array, Y_array, data, levels = 1000, vmin = 0, vmax = 0.05, cmap=plt.cm.magma_r)
    
    plt.show()

                                         
def main():                
    location_g_rs = "spec_loc_rs_3_band/spec_loc_rs_3_band_run_3"
    location_g_energy = "spec_loc_energy_3_band/spec_loc_energy_3_band_run_7"

    #contour_plot_spec_loc(location_g_rs)
    
    energy_plot_spec_loc(location_g_energy)

    
    
    
    
    
    

    
    
    
if __name__ == '__main__':
    main()




    