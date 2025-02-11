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
    

def plot_conductance_single_W(location):
    """Contour plot conductance as function of E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')  

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    W = int(parameters["W"][0])
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
    a1.set_xlabel("E", fontsize=30)
    a1.set_ylabel(r"$G[e^2/\hbar]$", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
        
    a1.plot(E_vals, data, color = "red")
    
    a1.set_ylim(bottom = 0)
    
    plt.show()
    

def main():                
    location_G_F_B_single_W = "G_F_B_model_single_W_results/F_B_model_single_W_run_1"
    location_G_DC_CI_single_W = "G_DC_CI_model_single_W_results/DC_CI_model_single_W_run_3"
    location_G_CI_single_W = "G_regular_CI_single_W_results/DC_regular_CI_single_W_run_2"
    
    
    
    plot_conductance_single_W(location_G_DC_CI_single_W)
    

if __name__ == '__main__':
    main()




    