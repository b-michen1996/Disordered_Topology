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


def title_generator_line_plot(parameters):
    """Generate title from parameters"""
    Nx = int(parameters["Nx"][0])
    Ny = int(parameters["Ny"][0])
    PBC =parameters["PBC"][0]
    W = parameters["W"][0]
    
    try:
        r = parameters["r"][0]
        epsilon_1 = parameters["epsilon_1"][0]
        epsilon_2 = parameters["epsilon_2"][0]
        gamma = parameters["gamma"][0]
        gamma_2 = parameters["gamma_2"][0]
        n_vec_KPM = parameters["n_vec_KPM"][0]
        
        print("n_vec_KPM ", n_vec_KPM)
                
        title = rf"$W = {W}$, $r= {r}$, $\epsilon_1 = {epsilon_1}$, $\epsilon_2 = {epsilon_2}$, $\gamma$ = {gamma}, $\gamma_2$ = {gamma_2}, $N_x = {Nx}$, $N_y = {Ny}$, PBC = {PBC}"
        
    except:
        try:
            r_1 = parameters["r_1"][0]
            r_2 = parameters["r_2"][0]    
            gamma = parameters["gamma"][0]
            v = parameters["v"][0]
                
            U = parameters["U"][0]
                    
            title = rf"$W = {W}$, $U = {U}$, $r_1= {r_1}$, $r_2 = {r_2}$, $\gamma$ = {gamma}, $v$ = {v}, $N_x = {Nx}$, $N_y = {Ny}$, PBC = {PBC}"
        except:
            r_1 = parameters["r_1"][0]
            r_2 = parameters["r_2"][0]    
            gamma = parameters["gamma"][0]
                
            U = parameters["U"][0]
                    
            title = rf"$W = {W}$, $U = {U}$, $r_1= {r_1}$, $r_2 = {r_2}$, $\gamma$ = {gamma}, $N_x = {Nx}$, $N_y = {Ny}$, PBC = {PBC}"
            
                    
    return title
       

def plot_DOS(location, plot_var = "Nx"):
    """Plot average and typical DOS"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')  
    rho_av = np.genfromtxt(location + "/rho_av.txt", dtype= float, delimiter=' ')  
    rho_typ = np.genfromtxt(location + "/rho_typ.txt", dtype= float, delimiter=' ')  

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    W = parameters["W"][0]
    PBC =parameters["PBC"][0]
    n_moments_KPM = parameters["n_moments_KPM"][0]
    
    print("Nx = ", Nx)
    print("Ny = ", Ny)
    print("W = ", W)   
    print("n_moments_KPM = ", n_moments_KPM)      
    
    title = title_generator_line_plot(parameters)

    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel(r"$E$", fontsize=30)
    a1.set_ylabel(r"$\rho$", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
            
    a1.plot(E_vals, rho_av, color = "red", label = r"$\rho_\text{av}$")
    a1.plot(E_vals, rho_typ, color = "blue", label = r"$\rho_\text{typ}$")
    
    a1.set_ylim(bottom=0)
    
    a1.legend(loc='upper left', prop={'size': 12})
        
    plt.show()  
    
  
def plot_DOS_final(location, plot_var = "Nx"):
    """Plot average, typical, and IPR DOS"""
    parameters = gen_dictionary(location + "/parameters.txt")
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')  
    rho_av = np.genfromtxt(location + "/rho_av.txt", dtype= float, delimiter=' ')  
    rho_typ = np.genfromtxt(location + "/rho_typ.txt", dtype= float, delimiter=' ')  
    rho_IPR = np.genfromtxt(location + "/rho_IPR.txt", dtype= float, delimiter=' ')  

    Nx = parameters["Nx"][0]
    Ny = parameters["Ny"][0]
    W = parameters["W"][0]
    PBC =parameters["PBC"][0]
    n_moments_KPM = parameters["n_moments_KPM"][0]
    
    print("Nx = ", Nx)
    print("Ny = ", Ny)
    print("W = ", W)   
    print("n_moments_KPM = ", n_moments_KPM)      
    
    title = title_generator_line_plot(parameters)

    fig = plt.figure(figsize=(6, 6), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel(r"$E$", fontsize=30)
    a1.set_ylabel(r"$\rho$", fontsize=30)
    a1.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)
            
    a1.plot(E_vals, rho_av, color = "red", label = r"$\rho_\text{av}$")
    a1.plot(E_vals, rho_typ, color = "blue", label = r"$\rho_\text{typ}$")
    a1.plot(E_vals, rho_IPR, color = "green", label = r"$\rho_\text{IPR}$")
    
    a1.set_ylim([0,0.5])
    
    a1.legend(loc='upper left', prop={'size': 12})
        
    plt.show() 
    

def main():                
    location_DOS_final = "DOS_final_V2_results/DOS_final_V2_run_1"
    
    #plot_DOS(location_DOS_DC_CI_V3)
    plot_DOS_final(location_DOS_final)

    

    

if __name__ == '__main__':
    main()




    