import csv
import numpy as np
import os

from matplotlib import pyplot as plt
import matplotlib.colors as colors


# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})



def fuse_data_Fig_2_3_App(location_1, location_2, output_folder):
    """Fuse data for Fig 2 App."""
    E_vals_data_1 = np.genfromtxt(location_1 + "/E_vals.txt", dtype= float, delimiter=' ') 
    data_cond_mat_1 = np.load(location_1 + "/result_cond_mat.npy")
    
    data_cond_mat_2 = np.load(location_2 + "/result_cond_mat.npy")
    
    n_cut_1 = 23
    n_cut_2 = 17
    
    n_E = len(E_vals_data_1)
    data_sigma_cond_mat_Fig_2_3 = np.zeros((n_E, 4, 4, n_cut_1 + n_cut_2))
    
    data_sigma_cond_mat_Fig_2_3[:, :, :, :n_cut_1] = data_cond_mat_1[:, :, :, :n_cut_1]
    data_sigma_cond_mat_Fig_2_3[:, :, :, n_cut_1:] = data_cond_mat_2[:, :, :, :n_cut_2]
    
    try:
        os.makedirs(output_folder)
    except:
        pass 
    
    np.savetxt(output_folder + "/E_vals.txt", E_vals_data_1, delimiter=' ')  
    np.save(output_folder + "/result_cond_mat", data_sigma_cond_mat_Fig_2_3) 
         

def get_conductance_terminal_difference(cond_mat):
    """Get sigma_xy from transmission difference."""
    
    # sigma_xy = T_LT - T_LB
    sigma_xy = (- cond_mat[0,3])  - (-cond_mat[0,2])
    
    # sigma_yx = -(T_BL - T_BR) = T_BR - T_BL
    sigma_yx = (-cond_mat[2,1]) - (- cond_mat[2,0])
   
    sigma_xx = -cond_mat[1,0]
    sigma_yy = -cond_mat[3,2]    
    
    return sigma_xx, sigma_xy, sigma_yx, sigma_yy



def get_conductance_V_Hall(cond_mat):
    """Get elements of the conductance tensor from 4 lead conduction matrix for current between lead 0 and 1 and lead 2 and 3. Lead 0 to 1. 
    Starting counterclockwise from the left edge of a square sample, the leads are assumed to be attached in the order 0,2,1, 3."""
    
    # Set V_3 = 0 and solve I = G V for I_0 = - I_1 = 1 and I_2 = 0, which implies I_3 = 0. This yields
    # V_0, V_1, and V_2, from which the conductance elements are calculated as as sigma_xy = I_0 / (V_2 - V_3) = 1 / V_2
    # and sigma_xx = I_0 / (V_0 - V_1)
    cond_mat_truc_i = cond_mat[:-1, :-1]
    V_i = np.linalg.solve(cond_mat_truc_i, [1,-1, 0])   
    sigma_xx = 1 / (V_i[0] - V_i[1])
    sigma_xy = 1 / V_i[2]
    
    # Set V_0 = 0 and solve I = G V for I_2 = - I_3 = 1 and I_1 = 0, which implies I_0 = 0. This yields
    # V_1, V_2, and V_3, from which the transverse conductance is calculated as as sigma_yx = I_2 / (V_0 - V_1) = -1 / V_1 and
    # sigma_yy = I_2 / (V_2 - V_3)
    cond_mat_truc_ii = cond_mat[1:, 1:]
    V_ii = np.linalg.solve(cond_mat_truc_ii, [0, 1,-1])
    sigma_yy = 1 / (V_ii[1] - V_ii[2])
    sigma_yx = -1 / V_ii[0]
    
    return sigma_xx, sigma_xy, sigma_yx, sigma_yy



def Fig_3_App(location, n_vals = 1):
    """Line plot for Hall conductance over E, obtained from averaged coductance matrix"""
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')
    data_result = np.load(location + "/result_cond_mat.npy")
                
    sigma_indices = ["xy", "yx"]
    
    n_E = len(E_vals)
    
    
    sigma_xy = np.zeros(n_E)
    sigma_yx = np.zeros(n_E)
    
    sigma_xy_V_Hall = np.zeros(n_E)
    sigma_yx_V_Hall = np.zeros(n_E)
    
    sigma_xy_std = np.zeros(n_E)
    sigma_yx_std = np.zeros(n_E)
    sigma_xy_V_Hall_std = np.zeros(n_E)
    sigma_yx_V_Hall_std = np.zeros(n_E)
            
    for j_E in range(0, n_E):
        sigma_curr = np.zeros((2,2, n_vals))
        sigma_curr_V_Hall = np.zeros((2,2, n_vals))
                
        for j in range(n_vals):            
            sigma_curr[0,0,j], sigma_curr[0,1,j], sigma_curr[1,0,j], sigma_curr[1,1,j] = get_conductance_terminal_difference(data_result[j_E,:,:,j])            
            sigma_curr_V_Hall[0,0,j], sigma_curr_V_Hall[0,1,j], sigma_curr_V_Hall[1,0,j], sigma_curr_V_Hall[1,1,j] = get_conductance_V_Hall(data_result[j_E,:,:,j])            
            
        
        sigma_curr_av =  np.mean(sigma_curr, axis = 2)
        sigma_curr_std = np.std(sigma_curr, axis = 2)
        
        sigma_V_Hall_curr_av =  np.mean(sigma_curr_V_Hall, axis = 2)
        sigma_V_Hall_curr_std = np.std(sigma_curr_V_Hall, axis = 2)
        
        sigma_xy[j_E] = sigma_curr_av[0,1]
        sigma_yx[j_E] = sigma_curr_av[1,0]
        sigma_xy_V_Hall[j_E] = sigma_V_Hall_curr_av[0,1]
        sigma_yx_V_Hall[j_E] = sigma_V_Hall_curr_av[1,0]
        
        sigma_xy_std[j_E] = sigma_curr_std[0,1]
        sigma_yx_std[j_E] = sigma_curr_std[1,0]        
        sigma_xy_V_Hall_std[j_E] = sigma_V_Hall_curr_std[0,1]
        sigma_yx_V_Hall_std[j_E] = sigma_V_Hall_curr_std[1,0]        
        
    data_sets = [sigma_xy, sigma_yx]
    data_sets_std = [sigma_xy_std, sigma_yx_std]
    
    data_sets_V_Hall = [sigma_xy_V_Hall, sigma_yx_V_Hall]
    data_sets_V_Hall_std = [sigma_xy_V_Hall_std, sigma_yx_V_Hall_std]
            
    fig = plt.figure(figsize=(12, 3), layout = "tight")
    
    label_size = 15
    
    E_l_plateau_1 = -1.7
    E_r_plateau_1 = -0.35
    E_l_plateau_2 = 10
    E_r_plateau_2 = 10
    E_l_plateau_3 = 0.35
    E_r_plateau_3 = 1.7
    
    mask_plateau_1 = (E_vals > E_l_plateau_1) &  (E_vals < E_r_plateau_1)
    mask_plateau_2 = (E_vals > E_l_plateau_2) &  (E_vals < E_r_plateau_2)
    mask_plateau_3 = (E_vals > E_l_plateau_3) &  (E_vals < E_r_plateau_3)
    
    for l in range(0, 2):
        sigma_index = sigma_indices[l]
        y_title = r"$\sigma_{" + sigma_index + "}[e^2/h]$"
        
        al =  fig.add_subplot(1, 2, l + 1)        
        al.set_xlabel("E", fontsize=label_size)
        al.set_ylabel(y_title, fontsize=label_size)
        al.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   
        
        data_curr = data_sets[l]
        data_curr_std = data_sets_std[l]
        
        data_curr_V_Hall = data_sets_V_Hall[l]
        data_curr_V_Hall_std = data_sets_V_Hall_std[l]        
        
        plt.axhline(y=0, color='black', linestyle='-')
        
        al.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
        al.set_xticks([-3, -2, -1, 0, 1, 2, 3])
                
        al.set_xticklabels([r"$-3$", "", "", r"$0$", "", "", r"$3$"])
        
        al.xaxis.set_label_coords(0.87, -0.06)
        al.yaxis.set_label_coords(-0.02, 0.5)
    
        # plot result
        al.plot(E_vals, data_curr, color = "red", label = r"Using $T_{p \leftarrow q}$")
        al.plot(E_vals[mask_plateau_1], data_curr_V_Hall[mask_plateau_1], color = "blue", label = r"Using $V_\text{Hall}$")
        al.plot(E_vals[mask_plateau_2], data_curr_V_Hall[mask_plateau_2], color = "blue")
        al.plot(E_vals[mask_plateau_3], data_curr_V_Hall[mask_plateau_3], color = "blue")
        
        # indicate standard deviation
        #al.fill_between(E_vals, data_curr_V_Hall - data_curr_V_Hall_std, data_curr_V_Hall + data_curr_V_Hall_std, facecolor = "gray", lw = 0)
        al.fill_between(E_vals[mask_plateau_1], data_curr_V_Hall[mask_plateau_1] - data_curr_V_Hall_std[mask_plateau_1], data_curr_V_Hall[mask_plateau_1] + data_curr_V_Hall_std[mask_plateau_1], facecolor = "blue", lw = 0, alpha = 0.2)
        al.fill_between(E_vals[mask_plateau_3], data_curr_V_Hall[mask_plateau_3] - data_curr_V_Hall_std[mask_plateau_3], data_curr_V_Hall[mask_plateau_3] + data_curr_V_Hall_std[mask_plateau_3], facecolor = "blue", lw = 0, alpha = 0.2)
        
        if l == 1:            
            al.set_ylim([-1.1, 0.1])
            
            al.set_yticks([-1, -0.5, 0])
            al.set_yticklabels([r"$-1$", "", r"$0$"])
            
            plt.axhline(y=-1, color='black', linestyle='--')
        else:        
            al.set_ylim([-0.1, 1.1])
            
            al.set_yticks([0, 0.5, 1])
            al.set_yticklabels([r"$0$", "", r"$1$"])
            
            al.legend(loc="upper left", bbox_to_anchor=(0.19, 0.4), fontsize = 0.75 * label_size)
            
            plt.axhline(y=1, color='black', linestyle='--')
        
    plt.show()
    
    fig.tight_layout()
    fig.savefig("Fig_3_App_publication.png",  bbox_inches='tight', dpi = 600)
    
    return


def Fig_2_App(location, n_vals = 1):
    """Line plot for Hall conductance over E, obtained from averaged coductance matrix"""
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    data_result = np.load(location + "/result_cond_mat.npy")
                
    sigma_indices = ["xx", "xy", "yx", "yy"]
    
    n_E = len(E_vals)
    
    sigma_xx = np.zeros(n_E)
    sigma_xy = np.zeros(n_E)
    sigma_yx = np.zeros(n_E)
    sigma_yy = np.zeros(n_E)
    
    sigma_xx_std = np.zeros(n_E)
    sigma_xy_std = np.zeros(n_E)
    sigma_yx_std = np.zeros(n_E)
    sigma_yy_std = np.zeros(n_E)
        
    for j_E in range(0, n_E):
        sigma_curr = np.zeros((2,2, n_vals))
                
        for j in range(n_vals):            
            sigma_curr[0,0,j], sigma_curr[0,1,j], sigma_curr[1,0,j], sigma_curr[1,1,j] = get_conductance_terminal_difference(data_result[j_E,:,:,j])
            
        sigma_curr_av =  np.mean(sigma_curr, axis = 2)
        sigma_curr_std = np.std(sigma_curr, axis = 2)
        
        sigma_xx[j_E] = sigma_curr_av[0,0]
        sigma_xy[j_E] = sigma_curr_av[0,1]
        sigma_yx[j_E] = sigma_curr_av[1,0]
        sigma_yy[j_E] = sigma_curr_av[1,1]
        
        sigma_xx_std[j_E] = sigma_curr_std[0,0]
        sigma_xy_std[j_E] = sigma_curr_std[0,1]
        sigma_yx_std[j_E] = sigma_curr_std[1,0]
        sigma_yy_std[j_E] = sigma_curr_std[1,1]
        
    
    data_sets = [sigma_xx, sigma_xy, sigma_yx, sigma_yy]
    data_sets_std = [sigma_xx_std, sigma_xy_std, sigma_yx_std, sigma_yy_std]
            
    fig = plt.figure(figsize=(12, 6), layout = "tight")
    
    label_size = 15
    
    for l in range(0, 4):
        sigma_index = sigma_indices[l]
        y_title = r"$\sigma_{" + sigma_index + "}[e^2/h]$"
        
        al =  fig.add_subplot(2, 2, l + 1)        
        al.set_xlabel("E", fontsize=label_size)
        al.set_ylabel(y_title, fontsize=label_size)
        al.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   
        data_curr = data_sets[l]
        data_curr_std = data_sets_std[l]
        
        plt.axhline(y=0, color='black', linestyle='-')
        
        al.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
        al.set_xticks([-3, -2, -1, 0, 1, 2, 3])
                
        al.set_xticklabels([r"$-3$", "", "", r"$0$", "", "", r"$3$"])
        
        al.xaxis.set_label_coords(0.87, -0.06)
        al.yaxis.set_label_coords(-0.02, 0.5)
    
        # plot result
        al.plot(E_vals, data_curr, color = "red")
        
        # indicate standard deviation
        al.fill_between(E_vals, data_curr - data_curr_std, data_curr + data_curr_std, facecolor = "red", lw = 0, alpha = 0.2)
        
        if l == 2:            
            al.set_ylim([-1.1, 0.1])
            
            al.set_yticks([-1, -0.5, 0])
            al.set_yticklabels([r"$-1$", "", r"$0$"])
            
            plt.axhline(y=-1, color='black', linestyle='--')
        else:        
            al.set_ylim([-0.1, 1.1])
            
            al.set_yticks([0, 0.5, 1])
            al.set_yticklabels([r"$0$", "", r"$1$"])
            
            
            plt.axhline(y=1, color='black', linestyle='--')
            
            
        
    plt.show()
    
    fig.tight_layout()
    fig.savefig("Fig_2_App_publication.png",  bbox_inches='tight', dpi = 600)
    
    return


def main():                       
    output_folder = "Data_Fig_2_3_App"    
    
    n_vals = 40

    Fig_2_App(output_folder, n_vals)
    Fig_3_App(output_folder, n_vals)
    
    data_location_1 = "G_final_V3_Hall_line_run_26"
    data_location_2 = "G_final_V3_Hall_line_run_31"
    
    #fuse_data_Fig_2_3_App(data_location_1, data_location_2, output_folder)

    
    

if __name__ == '__main__':
    main()




    