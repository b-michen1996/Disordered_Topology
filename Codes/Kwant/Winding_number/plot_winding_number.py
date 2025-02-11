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
    

def plot_w_num(location):
    """Plot winding number as function of E"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')  
    
    try:        
        data = np.mean(data[:,:], axis = 1)
    except:
        pass

    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    W = parameters["W"][0]
    

    r_1 = parameters["r_1"][0]
    r_2 = parameters["r_2"][0]
    title = rf"$r_1 = {r_1}$, $r_2 = {r_2}$, $W = {W}$, $N_x = {Nx}$, $N_y = {Ny}$"

    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel("E", fontsize=30)
    a1.set_ylabel(r"$\mathcal W$", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
        
    a1.plot(E_vals, data, color = "red")
        
    plt.show()
    

def main():                
    location_G_DC_CI_single_W = "Winding_number_DC_CI/Winding_number_DC_CI_run_14"
           
    plot_w_num(location_G_DC_CI_single_W)
    

if __name__ == '__main__':
    main()




    