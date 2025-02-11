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
        
    
def contour_plot_Delta_G(location, dots = False):
    """Contour plot for difference of conductance L-R and R-L in 3-terminal setting over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    data_LR = np.genfromtxt(location + "/result_L_R.txt", dtype= float, delimiter=' ')  
    data_RL = np.genfromtxt(location + "/result_R_L.txt", dtype= float, delimiter=' ')  
    
    data = np.abs(data_LR - data_RL)

    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    
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
    
  
        
    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel("W", fontsize=30)
    a1.set_ylabel("E", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
    
    n_levels = 1000
    vmin = 0.
    vmax = 1.2
    
    if dots:
        scatterplot = a1.scatter(W_array, E_array, marker = "o", c = data, 
                                     label = r"$E_{n}(k_x)$", cmap=plt.cm.magma_r, vmin= vmin, vmax = vmax)
        
        cbar_title = r"$\Delta G[e^2/\hbar]$"
        cbar_1 = plt.colorbar(scatterplot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)
    else:
        levels = np.linspace(vmin, vmax, n_levels)
        contour_plot = a1.contourf(W_array, E_array, data, levels = levels, cmap=plt.cm.magma_r, extend = "both")
        
        cbar_title = r"$\Delta G[e^2/\hbar]$"
        cbar_1 = plt.colorbar(contour_plot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)  
    
    plt.show()
    

def contour_plot_G_LR_RL(location, dots = False):
    """Contour plot for L-R and R-L conductance  in 3-terminal setting over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    data_LR = np.genfromtxt(location + "/result_L_R.txt", dtype= float, delimiter=' ')  
    data_RL = np.genfromtxt(location + "/result_R_L.txt", dtype= float, delimiter=' ') 

    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    
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
    
  
        
    fig = plt.figure(figsize=(12, 6), layout = "tight")
    a1 =  fig.add_subplot(2,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel("W", fontsize=30)
    a1.set_ylabel("E", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
    
    n_levels = 1000
    vmin = 0.
    vmax = 4
    
    if dots:
        scatterplot = a1.scatter(W_array, E_array, marker = "o", c = data_LR, 
                                     label = r"$E_{n}(k_x)$", cmap=plt.cm.magma_r, vmin= vmin, vmax = vmax)
        
        cbar_title = r"$G_{LR}[e^2/\hbar]$"
        cbar_1 = plt.colorbar(scatterplot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)
    else:
        levels = np.linspace(vmin, vmax, n_levels)
        contour_plot = a1.contourf(W_array, E_array, data_LR, levels = levels, cmap=plt.cm.magma_r, extend = "both")
        
        cbar_title = r"$G_{LR}[e^2/\hbar]$"
        cbar_1 = plt.colorbar(contour_plot)
        cbar_1.ax.set_title(cbar_title, fontsize = 20)  
        
    a2 =  fig.add_subplot(2,1,2)
    a2.set_title(title , fontsize=30)
    a2.set_xlabel("W", fontsize=30)
    a2.set_ylabel("E", fontsize=30)
    a2.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
    
    
    if dots:
        scatterplot_2 = a2.scatter(W_array, E_array, marker = "o", c = data_LR, 
                                     label = r"$E_{n}(k_x)$", cmap=plt.cm.magma_r, vmin= vmin, vmax = vmax)
        
        cbar_2_title = r"$G_{RL}[e^2/\hbar]$"
        cbar_2 = plt.colorbar(scatterplot_2)
        cbar_2.ax.set_title(cbar_2_title, fontsize = 20)
    else:
        levels = np.linspace(vmin, vmax, n_levels)
        contour_plot_2 = a2.contourf(W_array, E_array, data_RL, levels = levels, cmap=plt.cm.magma_r, extend = "both")
        
        cbar_2_title = r"$G_{RL}[e^2/\hbar]$"
        cbar_2 = plt.colorbar(contour_plot_2)
        cbar_2.ax.set_title(cbar_2_title, fontsize = 20)  
        
    
    plt.show()
    

def contour_plot_all_G(location, dots = False):
    """Contour plot for conductance between all terminals in in 3-terminal setting over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    
    G_indices = ["L_R", "L_T", "R_T", "R_L", "T_L", "T_R"]
    
    data_sets = list()
    
    for G_index in G_indices:
        data_curr = np.genfromtxt(location + "/result_" + G_index + ".txt", dtype= float, delimiter=' ')  
        data_sets.append(data_curr) 


    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    
    r_1 = parameters["r_1"][0]
    r_2 = parameters["r_2"][0]
    title = rf"$r_1 = {r_1}$, $r_2 = {r_2}$, $N_x = {Nx}$, $N_y = {Ny}$"
    
    
    fig = plt.figure(figsize=(6, 18), layout = "tight")
    fig.suptitle(title, fontsize = 20)
    
    n_levels = 1000
    vmin = 0.
    vmax = 2
    
    for l in range(0, 6):
        al =  fig.add_subplot(2, 3, l + 1)
        al.set_xlabel("W", fontsize=30)
        al.set_ylabel("E", fontsize=30)
        al.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   
        
        data_curr = data_sets[l]
        
        G_index = G_indices[l]
        
        if dots:
            scatterplot = al.scatter(W_array, E_array, marker = "o", c = data_curr, 
                                         cmap=plt.cm.magma_r, vmin= vmin, vmax = vmax)
            
            cbar_title = r"G_" + G_index + "$[e^2/\hbar]$"
            cbar = plt.colorbar(scatterplot)
            cbar.ax.set_title(cbar_title, fontsize = 20)
        else:
            levels = np.linspace(vmin, vmax, n_levels)
            contour_plot = al.contourf(W_array, E_array, data_curr, levels = levels, cmap=plt.cm.magma_r, extend = "both")
            
            cbar_title = r"G_" + G_index + "$[e^2/\hbar]$"
            cbar = plt.colorbar(contour_plot)
            cbar.ax.set_title(cbar_title, fontsize = 20)  
    
        
    plt.show()
    
    
def contour_plot_all_delta_G(location, dots = False):
    """Contour plot for directional difference in conductance between all terminals in in 3-terminal setting over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    
    G_indices = ["L_R", "L_T", "R_T", "R_L", "T_L", "T_R"]
    
    data_sets = list()
    
    for G_index in G_indices:
        data_curr = np.genfromtxt(location + "/result_" + G_index + ".txt", dtype= float, delimiter=' ')  
        data_sets.append(data_curr) 


    W_array, E_array = np.meshgrid(W_vals, E_vals)

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    
    r_1 = parameters["r_1"][0]
    r_2 = parameters["r_2"][0]
    title = rf"$r_1 = {r_1}$, $r_2 = {r_2}$, $N_x = {Nx}$, $N_y = {Ny}$"
    
    
    fig = plt.figure(figsize=(6, 18), layout = "tight")
    fig.suptitle(title, fontsize = 20)
    
    n_levels = 1000
    vmin = 0.
    vmax = 1.5
    
    for l in range(0, 3):
        al =  fig.add_subplot(1, 3, l + 1)
        al.set_xlabel("W", fontsize=30)
        al.set_ylabel("E", fontsize=30)
        al.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   
        
        data_curr_1 = data_sets[l]
        data_curr_2 = data_sets[l + 3]
        
        data_curr = np.abs(data_curr_1 - data_curr_2)
        
        
                
        G_index_1 = G_indices[l]
        G_index_2 = G_indices[l + 3]
        
        al.set_title(r"G_" + G_index_1 + " - G_" + G_index_2 + "$[e^2/\hbar]$", fontsize = 20)
        
        if dots:
            scatterplot = al.scatter(W_array, E_array, marker = "o", c = data_curr, 
                                         cmap=plt.cm.magma_r, vmin= vmin, vmax = vmax)
            cbar = plt.colorbar(scatterplot)
            
        else:
            levels = np.linspace(vmin, vmax, n_levels)
            contour_plot = al.contourf(W_array, E_array, data_curr, levels = levels, cmap=plt.cm.magma_r, extend = "both")
                        
            cbar = plt.colorbar(contour_plot)
            
    
        
    plt.show()
    
    


def main():                
    location_G_DC_CI_3T = "G_DC_CI_3T_results/DC_CI_3T_run_9"    
    dots = False

    #contour_plot_Delta_G(location_G_DC_CI_3T, dots)
    #contour_plot_G_LR_RL(location_G_DC_CI_3T, dots)
    #contour_plot_all_G(location_G_DC_CI_3T, dots)
    contour_plot_all_delta_G(location_G_DC_CI_3T, dots)
    
    

if __name__ == '__main__':
    main()




    