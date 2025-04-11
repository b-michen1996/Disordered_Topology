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
    direction = parameters["direction"][0]
    PBC = int(parameters["PBC"][0])
    

    try:
        epsilon_1 = parameters["epsilon_1"][0]
        epsilon_2 = parameters["epsilon_2"][0]
        gamma = parameters["gamma"][0]
        gamma_2 = parameters["gamma_2"][0]
                
        title = rf"$\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $\gamma_2$ = {gamma_2}, $N_x = {Nx}$, $N_y = {Ny}$, direction {direction}, PBC = {PBC}, "
    except:
        try:
            r = parameters["r"][0]
            epsilon_1 = parameters["epsilon_1"][0]
            epsilon_2 = parameters["epsilon_2"][0]
            gamma = parameters["gamma"][0]
            v = parameters["v"][0]
                    
            title = rf"$r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $v$ = {v}, $N_x = {Nx}$, $N_y = {Ny}$, direction {direction}, PBC = {PBC}, "           
        except:
            r = parameters["r"][0]
            epsilon_1 = parameters["epsilon_1"][0]
            epsilon_2 = parameters["epsilon_2"][0]
            gamma = parameters["gamma"][0]
                    
            title = rf"$r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $N_x = {Nx}$, $N_y = {Ny}$, direction {direction}, PBC = {PBC}, "           
                    
    return title

    
def contour_plot_conductance(location, dots = False):
    """Contour plot of conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')  

    W_array, E_array = np.meshgrid(W_vals, E_vals)
    
    title = title_generator_contour_plot(parameters)
                
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
    location_G_final = "G_final_results/G_final_run_14"
    
    dots = True
    
    contour_plot_conductance(location_G_final, dots)    
    

if __name__ == '__main__':
    main()




    