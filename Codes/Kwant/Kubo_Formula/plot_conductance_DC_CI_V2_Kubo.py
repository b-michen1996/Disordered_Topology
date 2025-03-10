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
    
    
def contour_plot_Kubo_DC_CI(location, dots = False):
    """Contour plot of conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    
    sigma_indices = ["xx", "xy", "yx", "yy"]
    
    data_sets = list()
    
    for sigma_index in sigma_indices:
        data_curr = np.genfromtxt(location + "/result_sigma_" + sigma_index + ".txt", dtype= float, delimiter=' ')          
        data_sets.append(data_curr)  
        #data_sets.append(np.abs(data_curr))  
        
    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    r_1 = parameters["r_1"][0]
    r_2 = parameters["r_2"][0]    
    gamma = parameters["gamma"][0]
            
    title = rf"$r_1= {r_1}$, $r_2 = {r_2}$,$\gamma$ = {gamma}, $N_x = {Nx}$, $N_y = {Ny}$"
    
    fig = plt.figure(figsize=(24, 24), layout = "tight")    
    fig.suptitle(title, fontsize = 20)
    
    n_levels = 100
    vmin = -1
    vmax = 1

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
            
            cbar_title = r"$\sigma_{" + sigma_index + "$}[e^2/\hbar]$"
            cbar = plt.colorbar(scatterplot)
            cbar.ax.set_title(cbar_title, fontsize = 20)
        else:
            levels = np.linspace(vmin, vmax, n_levels)
            contour_plot = al.contourf(W_array, E_array, data_curr, levels = levels, cmap=plt.cm.magma_r, extend = "both")
            
            cbar_title = r"$\sigma_{" + sigma_index + "}[e^2/\hbar]$"
            cbar = plt.colorbar(contour_plot)
            cbar.ax.set_title(cbar_title, fontsize = 20) 
        
    plt.show()
    
    
def contour_plot_Kubo_DC_CI_old(location, dots = False):
    """Contour plot of conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    
    sigma_indices = ["xx", "xy", "yx", "yy"]
    
    data_sets = list()
    
    for sigma_index in sigma_indices:
        data_curr = np.genfromtxt(location + "/result_sigma_" + sigma_index + ".txt", dtype= float, delimiter=' ')          
        data_sets.append(data_curr) 
        
    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    r_1 = parameters["r_1"][0]
    r_2 = parameters["r_2"][0]    
    gamma = parameters["gamma"][0]
            
    title = rf"$r_1= {r_1}$, $r_2 = {r_2}$,$\gamma$ = {gamma}, $N_x = {Nx}$, $N_y = {Ny}$"
    
    fig = plt.figure(figsize=(24, 24), layout = "tight")
    fig.suptitle(title, fontsize = 20)
    
    n_levels = 1000
    vmin = -100
    vmax = 100

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
            
            cbar_title = r"$\sigma_" + sigma_index + "[e^2/\hbar]$"
            cbar = plt.colorbar(scatterplot)
            cbar.ax.set_title(cbar_title, fontsize = 20)
        else:
            levels = np.linspace(vmin, vmax, n_levels)
            contour_plot = al.contourf(W_array, E_array, data_curr, levels = levels, cmap=plt.cm.magma_r, extend = "both")
            
            cbar_title = r"$\sigma_" + sigma_index + "[e^2/\hbar]$"
            cbar = plt.colorbar(contour_plot)
            cbar.ax.set_title(cbar_title, fontsize = 20)  
        
    plt.show()
    
    
def contour_plot_Kubo_DC_CI_V2(location, dots = False):
    """Contour plot of conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    
    sigma_indices = ["xx", "xy", "yx", "yy"]
    
    data_sets = list()
    
    for sigma_index in sigma_indices:
        data_curr = np.genfromtxt(location + "/result_sigma_" + sigma_index + ".txt", dtype= float, delimiter=' ')          
        data_sets.append(data_curr)  
        #data_sets.append(np.abs(data_curr))  
        
    sum_xy_yx = data_sets[1] + data_sets[2]
    diff_xy_yx = data_sets[1] - data_sets[2]
    #data_sets[1] = np.abs(sum_xy_yx)
    #data_sets[2] = np.abs(diff_xy_yx) 
    

    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    r = parameters["r"][0]
    epsilon_1 = parameters["epsilon_1"][0]
    epsilon_2 = parameters["epsilon_2"][0]
    gamma = parameters["gamma"][0]
            
    title = rf"$r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$,$\gamma$ = {gamma}, $N_x = {Nx}$, $N_y = {Ny}$"
    fig = plt.figure(figsize=(24, 24), layout = "tight")
    fig.suptitle(title, fontsize = 20)
    
    n_levels = 1000
    vmin = -2
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
            
            cbar_title = r"$\sigma_{" + sigma_index + "$}[e^2/\hbar]$"
            cbar = plt.colorbar(scatterplot)
            cbar.ax.set_title(cbar_title, fontsize = 20)
        else:
            levels = np.linspace(vmin, vmax, n_levels)
            contour_plot = al.contourf(W_array, E_array, data_curr, levels = levels, cmap=plt.cm.magma_r, extend = "both")
            
            cbar_title = r"$\sigma_{" + sigma_index + "}[e^2/\hbar]$"
            cbar = plt.colorbar(contour_plot)
            cbar.ax.set_title(cbar_title, fontsize = 20)  
        
    plt.show()
    

def contour_plot_Kubo_DC_CI_V2_old(location, dots = False):
    """Contour plot of conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')  

    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    r = parameters["r"][0]
    epsilon_1 = parameters["epsilon_1"][0]
    epsilon_2 = parameters["epsilon_2"][0]
    gamma = parameters["gamma"][0]
            
    title = rf"$r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$,$\gamma$ = {gamma}, $N_x = {Nx}$, $N_y = {Ny}$"

                
    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel("W", fontsize=30)
    a1.set_ylabel("E", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
    
    n_levels = 1000
    vmin = -2.2
    vmax = 2.2
    
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
    location_Kubo_DC_CI = "Kubo_DC_CI_results/Kubo_DC_CI_run_1"
    location_Kubo_DC_CI_V2 = "Kubo_DC_CI_V2_results/Kubo_DC_CI_V2_run_4"
    
    
    dots = False
    
    contour_plot_Kubo_DC_CI(location_Kubo_DC_CI, dots)    
    #contour_plot_Kubo_DC_CI_V2(location_Kubo_DC_CI_V2, dots)    
    

if __name__ == '__main__':
    main()




    