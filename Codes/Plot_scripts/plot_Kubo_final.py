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

    
def contour_plot_Kubo_final(location, dots = False):
    """Contour plot of conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    print("n_vec_KPM ", parameters["n_vec_KPM"])
    print("n_moments_KPM ", parameters["n_moments_KPM"])
    print("n_inst ", parameters["n_inst"])
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
        
    sigma_indices = ["xx", "xy", "yx", "yy"]
    
    data_sets = list()
    
    for sigma_index in sigma_indices:
        data_curr = np.genfromtxt(location + "/result_sigma_" + sigma_index + ".txt", dtype= float, delimiter=' ')          
        #data_sets.append(data_curr)  
        data_sets.append(np.abs(data_curr))  
        
    W_array, E_array = np.meshgrid(W_vals, E_vals)
    
    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])

    epsilon_1 = parameters["epsilon_1"][0]
    epsilon_2 = parameters["epsilon_2"][0]
    gamma = parameters["gamma"][0]
    gamma_2 = parameters["gamma_2"][0]
            
    title = rf"$\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $\gamma_2$ = {gamma_2}, $N_x = {Nx}$, $N_y = {Ny}$"
            
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
    
    
def line_plot_Kubo_final(location):
    """Contour plot of conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    print("n_vec_KPM ", parameters["n_vec_KPM"])
    print("n_moments_KPM ", parameters["n_moments_KPM"])
    print("n_inst ", parameters["n_inst"])
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    
    sigma_indices = ["xx", "xy", "yx", "yy"]
    
    data_sets = list()
    
    for sigma_index in sigma_indices:
        data_curr = np.genfromtxt(location + "/result_sigma_" + sigma_index + ".txt", dtype= float, delimiter=' ')          
        data_sets.append(data_curr)  
        #data_sets.append(np.abs(data_curr))  

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    W = parameters["W"][0]
    
    epsilon_1 = parameters["epsilon_1"][0]
    epsilon_2 = parameters["epsilon_2"][0]
    gamma = parameters["gamma"][0]
    gamma_2 = parameters["gamma_2"][0]
            
    title = rf"$W = {W}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $\gamma_2$ = {gamma_2}, $N_x = {Nx}$, $N_y = {Ny}$"

    fig = plt.figure(figsize=(24, 24), layout = "tight")
    fig.suptitle(title, fontsize = 20)

    
    for l in range(0, 4):
        sigma_index = sigma_indices[l]
        y_title = r"$\sigma_{" + sigma_index + "}[e^2/\hbar]$"
        
        al =  fig.add_subplot(2, 2, l + 1)        
        al.set_xlabel("E", fontsize=20)
        al.set_ylabel(y_title, fontsize=20)
        al.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   
        data_curr = data_sets[l]
        plt.axhline(y=1, color='b', linestyle='-')
        
        al.plot(E_vals, data_curr, color = "red", label = r"$\rho_\text{av}$")
        
    
    
    plt.show()
    
    
def line_plot_Kubo_final_multi_vals(location, n_vals):
    """Contour plot of conductance over W and E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    print("n_vec_KPM ", parameters["n_vec_KPM"])
    print("n_moments_KPM ", parameters["n_moments_KPM"])
    print("n_inst ", parameters["n_inst"])
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    
    sigma_indices = ["xx", "xy", "yx", "yy"]
    
    data_sets = list()
    
    for sigma_index in sigma_indices:
        data_curr = np.genfromtxt(location + "/result_sigma_" + sigma_index + ".txt", dtype= float, delimiter=' ')          
        data_sets.append(data_curr)  
        #data_sets.append(np.abs(data_curr))  

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    W = parameters["W"][0]
    
    epsilon_1 = parameters["epsilon_1"][0]
    epsilon_2 = parameters["epsilon_2"][0]
    gamma = parameters["gamma"][0]
    gamma_2 = parameters["gamma_2"][0]
            
    title = rf"$W = {W}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $\gamma_2$ = {gamma_2}, $N_x = {Nx}$, $N_y = {Ny}$, $n_v$ = {n_vals}"

    fig = plt.figure(figsize=(24, 24), layout = "tight")
    fig.suptitle(title, fontsize = 20)

    
    for l in range(0, 4):
        sigma_index = sigma_indices[l]
        y_title = r"$\sigma_{" + sigma_index + "}[e^2/\hbar]$"
        
        al =  fig.add_subplot(2, 2, l + 1)        
        al.set_xlabel("E", fontsize=20)
        al.set_ylabel(y_title, fontsize=20)
        al.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   
        data_curr = data_sets[l]
        plt.axhline(y=1, color='b', linestyle='-')
        
        data_curr_mean = np.mean(data_curr[:, :n_vals], axis = 1)
        
        al.plot(E_vals, data_curr_mean, color = "red", label = r"$\rho_\text{av}$")
        al.set_ylim([0, 1.2])
        
    
    
    plt.show()

    
def main():                
    location_Kubo_final = "Kubo_final_results/Kubo_final_run_3"

    location_Kubo_line = "Kubo_final_line_results/Kubo_final_line_run_23"
    location_Kubo_line_multi_vals = "Kubo_final_line_results/Kubo_final_line_run_31"
    
    
    
    dots = False
    
    #contour_plot_Kubo_final(location_Kubo_final, dots)    
    #line_plot_Kubo_final(location_Kubo_line)    
    
    n_vals = 10
    
    line_plot_Kubo_final_multi_vals(location_Kubo_line_multi_vals , n_vals)
   

if __name__ == '__main__':
    main()




    