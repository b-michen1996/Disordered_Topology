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
    

def plot_size_scaling_final(location, plot_var = "Nx"):
    """Plot transmission as a function of plot_var"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    data = np.genfromtxt(location + "/result.txt", dtype= float, delimiter=' ')  

    Nx = parameters["Nx"]
    Ny = parameters["Ny"]
    W = parameters["W"]
    PBC =parameters["PBC"][0]
    E = parameters["E"]
    epsilon_1 = parameters["epsilon_1"][0]
    epsilon_2 = parameters["epsilon_2"][0]
    gamma = parameters["gamma"][0]
    gamma_2 = parameters["gamma_2"][0]
    
    print("Nx = ", Nx)
    print("Ny = ", Ny)
    print("W = ", W)   
    print("E = ", E)   
    
    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_xlabel(plot_var, fontsize=30)
    a1.set_ylabel(r"$T [e^2/\hbar]$", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
        
    a1.plot(parameters[plot_var], data, color = "red")
        
    plt.show()
    
    
    

def main():                
    location_size_scaling_final= "size_scaling_final_results/size_scaling_final_run_2"

    plot_var = "Nx"
    
    plot_size_scaling_final(location_size_scaling_final, plot_var)
    

    

if __name__ == '__main__':
    main()




    