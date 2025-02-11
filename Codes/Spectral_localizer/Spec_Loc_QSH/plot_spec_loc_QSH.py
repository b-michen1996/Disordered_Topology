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
    

def contour_plot_spec_loc_QSH(location):
    """Contour plot for spectral localizer."""
    parameters = gen_dictionary(location + "/parameters.txt")

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    W = parameters["W"][0]
    alpha = parameters["alpha"][0]
    beta = parameters["beta"][0]
    gamma = parameters["gamma"][0]
    m = parameters["m"][0]
    E = parameters["E"][0]

    X_array = np.genfromtxt(location + "/X_array.txt", dtype= float, delimiter=' ') 
    Y_array = np.genfromtxt(location + "/Y_array.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')      

    title = rf"$g_\lambda(x,y, E)$ for $\alpha = {alpha}$, $\beta = {beta}$, $\gamma = {gamma}$, $m = {m}$, $W = {W}$, $N_x = {Nx}$, $N_y = {Ny}$ at $E = {E}$"

    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel("x", fontsize=30)
    a1.set_ylabel("y", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)

    #contour_plot = a1.contourf(X_array, Y_array, data, levels = 1000, vmin = 0, vmax = 0.05, cmap=plt.cm.magma_r)
    contour_plot = a1.contourf(X_array, Y_array, data, levels = 1000,  vmin = 0, cmap=plt.cm.magma_r)

    fig.colorbar(contour_plot)
    plt.show()


def rs_line_plot_spec_loc_QSH(location):
    """Line plot for localizer gap in real space"""
    parameters = gen_dictionary(location + "/parameters.txt")

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    y = parameters["y"][0]
    W = parameters["W"][0]
    E = parameters["E"][0]
    alpha = parameters["alpha"][0]
    beta = parameters["beta"][0]
    gamma = parameters["gamma"][0]
    m = parameters["m"][0]

    x_vals = np.genfromtxt(location + "/x_vals.txt", dtype= float, delimiter=' ')       
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')      

    title = rf"$g_\lambda(x,y, E)$ for $\alpha = {alpha}$, $\beta = {beta}$, $\gamma = {gamma}$, $m = {m}$, $W = {W}$, $y = {y}$, $N_x = {Nx}$, $N_y = {Ny}$, $E = {E}$"

    fig = plt.figure(figsize=(24, 12), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel(r"$g_\lambda(x)$", fontsize=30)
    a1.set_ylabel("x", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
    
    a1.set_ylim(bottom = 0)
   
    a1.plot(x_vals, data, linestyle = "--", color = "black")
    
    #fig.savefig(f"spec_loc_gap_alpha_{alpha}_beta_{beta}_gamma_{gamma}_m_{m}_W_{W}.png",  bbox_inches='tight', dpi=300)
    
    plt.show()
    

def energy_plot_spec_loc_QSH(location):
    """Contour plot for spectral localizer."""
    parameters = gen_dictionary(location + "/parameters.txt")

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    W = parameters["W"][0]

    alpha = parameters["alpha"][0]
    beta = parameters["beta"][0]
    gamma = parameters["gamma"][0]
    m = parameters["m"][0]
    
    W_2 = 0
    
    try:
        W_2 = parameters["W_2"][0]
    except:
        pass

    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')      
    bulk_energies = np.genfromtxt(location + "/bulk_energies.txt", dtype= float, delimiter=' ')      
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')      

    title = rf"$g_\lambda(E)$ for $\alpha = {alpha}$, $\beta = {beta}$, $\gamma = {gamma}$, $m = {m}$, $W = {W}$, $W_2 = {W_2}$, $N_x = {Nx}$, $N_y = {Ny}$"

    fig = plt.figure(figsize=(24, 12), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel(r"$g_\lambda(E)$", fontsize=30)
    a1.set_ylabel("E", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)

    x_min = 0
    x_max = np.max(data)

    y_min = np.min(E_vals)
    y_max = np.max(E_vals)

    a1.set_xlim([x_min, x_max])
    a1.set_ylim([y_min, y_max])
    
    a1.plot(data, E_vals, linestyle = "--", color = "black")

    a1.scatter(x_min * np.ones(len(bulk_energies)), bulk_energies, marker = "o", color = "red")
    
    x_1 = 0.07
    y_1 = 0.8
    a1.text(x = x_1, y = y_1, s = r"$Q = 0$", size = 30, ha = "center", transform=a1.transAxes, color = "green")
    
    x_1 = 0.07
    y_1 = 0.34
    a1.text(x = x_1, y = y_1, s = r"$Q = 0$", size = 30, ha = "center", transform=a1.transAxes, color = "green")
    
    x_1 = 0.07
    y_1 = 0.2
    #a1.text(x = x_1, y = y_1, s = r"$Q = -1$", size = 30, ha = "center", transform=a1.transAxes, color = "green")
    
    x_1 = 0.07
    y_1 = 0.03
    a1.text(x = x_1, y = y_1, s = r"$Q = 0$", size = 30, ha = "center", transform=a1.transAxes, color = "green")
    #contour_plot = a1.contourf(X_array, Y_array, data, levels = 1000, vmin = 0, vmax = 0.05, cmap=plt.cm.magma_r)
    
    #fig.savefig(f"spec_loc_gap_alpha_{alpha}_beta_{beta}_gamma_{gamma}_m_{m}_W_{W}.png",  bbox_inches='tight', dpi=300)
    
    plt.show()

                                         
def main():                
    location_g_rs = "spec_loc_rs_QSH/spec_loc_rs_QSH_run_7"
    location_g_rs_line = "spec_loc_rs_QSH/spec_loc_rs_line_QSH_run_4"
    location_g_energy = "spec_loc_energy_QSH/spec_loc_energy_QSH_run_7"

    #contour_plot_spec_loc_QSH(location_g_rs)
    
    #rs_line_plot_spec_loc_QSH(location_g_rs_line)
    
    energy_plot_spec_loc_QSH(location_g_energy)


    
    
    
    
    
    

    
    
    
if __name__ == '__main__':
    main()




    