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


def Fig_3(location):
    """Generate plot for Fig_3."""
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location + "/result_sigma_xy.txt", dtype= float, delimiter=' ')  
    
    W_array, E_array = np.meshgrid(W_vals, E_vals)
    
    fig, a1 = plt.subplots(1, 1, figsize =(12, 6))
    
    label_size = 25
    
    n_levels = 1000
    vmin = 0.
    vmax = 2    
    
    """
    im_sigma_xy = a1.imshow(data,
               aspect='auto',
               origin='lower',
               extent=[np.min(W_vals), np.max(W_vals), np.min(E_vals), np.max(E_vals)],
               vmin = 0,
               vmax = 2,
               cmap=plt.cm.magma.reversed()
              )
    """
    
    # coarse-grain data
    n_W = len(W_vals)
    n_E = len(E_vals)
    
    E_min = np.min(E_vals)
    E_max = np.max(E_vals)
    
    n_E_coarse = 250
    E_vals_coarse = np.linspace(E_min, E_max, n_E_coarse )
    
    data_coarse = np.zeros((n_E_coarse-1, n_W))
    
    # generate index array for coarse graining, such that 
    # E_vals[index_lower_coarse[j]] <= E_vals_coarse[j]
    
    index_lower_coarse = np.zeros(n_E_coarse,  dtype=int)
    
    for l in range(0, n_E_coarse-1):                    
        for j in range(index_lower_coarse [l], n_E):
            if E_vals[j] < E_vals_coarse[l+1]:                      
                index_lower_coarse[l+1] = j 
            else:
                break
            
    index_lower_coarse[-1] = n_E-1
                    
    im_sigma_xy = a1.imshow(data_coarse,
               aspect='auto',
               origin='lower',
               extent=[np.min(W_vals), np.max(W_vals), np.min(E_vals), np.max(E_vals)],
               vmin = 0,
               vmax = 2,
               cmap=plt.cm.magma.reversed()
              )
    
    cbar_title = r"$\sigma_{xy}[e^2/h]$"

    cbar = plt.colorbar(im_sigma_xy, shrink=0.6, ticks=[0, 1, 2], ax=a1, location='left', pad = 0.06)
    cbar.ax.tick_params(labelsize=label_size) 
    cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25)
    
    a1.set_xlabel(r"$W$", fontsize=label_size)
    a1.set_ylabel(r"$E_\mathrm{F}$", fontsize=label_size, rotation='horizontal')
    a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
        
    a1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    a1.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    
    a1.set_yticks([-3, -2, -1, 0., 1, 2, 3])
    a1.set_yticklabels([r"$-3$", "","", r"$0$","","", r"$3$"])
    a1.yaxis.set_ticks_position('right')
    
    a1.xaxis.set_label_coords(0.75, -0.04)
    a1.yaxis.set_label_coords(1.03, 0.86)
    
    a1.set_ylim(-3,3)
        
    fig.savefig("Fig_3_publication.png",  bbox_inches='tight', dpi = 600)
    
    plt.show()
    

def fuse_data_Fig_3(location_1, location_2_p1, location_2_p2, output_folder):
    """Fuse data for Fig 3"""
    E_vals = np.genfromtxt(location_1 + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location_1 + "/W_vals.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location_1 + "/result_sigma_xy.txt", dtype= float, delimiter=' ')  
    
    data_2_p1 = np.genfromtxt(location_2_p1 + "/result_sigma_xy.txt", dtype= float, delimiter=' ')  
    data_2_p2 = np.genfromtxt(location_2_p2 + "/result_sigma_xy.txt", dtype= float, delimiter=' ')  
    
    n_data_cut = 45
    
    data[:, :n_data_cut] = (10/14) * data[:, :n_data_cut] +  (4/14) * data_2_p1[:, :n_data_cut]
    data[:, n_data_cut:] = (1/3) * data[:, n_data_cut:] +  (2/3) * data_2_p2[:, n_data_cut:]
    
    try:
        os.makedirs(output_folder)
    except:
        pass 
    
    np.savetxt(output_folder + "/E_vals.txt", E_vals, delimiter=' ')  
    np.savetxt(output_folder + "/W_vals.txt", W_vals, delimiter=' ')  
    np.savetxt(output_folder + "/result_sigma_xy.txt", data, delimiter=' ')       
    
   
def Fig_3_ii(location, location_inset_1, location_inset_2, location_inset_3):
    """Generate plot for Fig_3."""
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location + "/W_vals.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location + "/result_sigma_xy.txt", dtype= float, delimiter=' ')    
    
    W_array, E_array = np.meshgrid(W_vals, E_vals)
    
    fig, a1 = plt.subplots(1, 1, figsize =(12, 6))
    
    label_size = 25
    label_scale_inset = 0.72
    
    n_levels = 1000
    vmin = 0.
    vmax = 2    
    
    # coarse-grain data
    n_W = len(W_vals)
    n_E = len(E_vals)
    
    E_min = np.min(E_vals)
    E_max = np.max(E_vals)
    
    n_E_coarse = 250
    E_vals_coarse = np.linspace(E_min, E_max, n_E_coarse )
    
    data_coarse = np.zeros((n_E_coarse-1, n_W))
    
    # generate index array for coarse graining, such that 
    # E_vals[index_lower_coarse[j]] <= E_vals_coarse[j]
    
    index_lower_coarse = np.zeros(n_E_coarse,  dtype=int)
    
    for l in range(0, n_E_coarse-1):                    
        for j in range(index_lower_coarse [l], n_E):
            if E_vals[j] < E_vals_coarse[l+1]:                      
                index_lower_coarse[l+1] = j 
            else:
                break
            
    index_lower_coarse[-1] = n_E-1
                        
    for j in range(0, n_E_coarse-1):
        ind_low = index_lower_coarse[j]
        ind_high = index_lower_coarse[j+1]
                                      
        data_coarse[j,:] = np.mean(data[ind_low:ind_high ,:], axis = 0)
    
    for j in range(0, n_E_coarse-1):
        ind_low = index_lower_coarse[j]
        ind_high = index_lower_coarse[j+1]
                                      
        data_coarse[j,:] = np.mean(data[ind_low:ind_high ,:], axis = 0)
    
    folder_coarse_data = "Data_Fig_3_contourplot"
    
    try:
        os.makedirs(folder_coarse_data)
    except:
        pass 
    
    np.savetxt(folder_coarse_data + "/E_vals.txt", E_vals_coarse[:-1], delimiter=' ')  
    np.savetxt(folder_coarse_data + "/W_vals.txt", W_vals, delimiter=' ')  
    np.savetxt(folder_coarse_data + "/result_sigma_xy.txt", data_coarse, delimiter=' ')  
    
    im_sigma_xy = a1.imshow(data_coarse,
               aspect='auto',
               origin='lower',
               extent=[np.min(W_vals), np.max(W_vals), np.min(E_vals), np.max(E_vals)],
               vmin = 0,
               vmax = 2,
               cmap=plt.cm.magma.reversed()
              )
    
    cbar_title = r"$\sigma_{xy}[e^2/h]$"

    cbar = plt.colorbar(im_sigma_xy, shrink=0.6, ticks=[0, 1, 2], ax=a1, location='left', pad = 0.06)
    cbar.ax.tick_params(labelsize=label_size) 
    cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25)
    
    a1.set_xlabel(r"$W$", fontsize=label_size)
    a1.set_ylabel(r"$E_\mathrm{F}$", fontsize=label_size, rotation='horizontal')
    a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
        
    a1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    a1.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    
    a1.set_yticks([-3, -2, -1, 0., 1, 2, 3])
    a1.set_yticklabels([r"$-3$", "","", r"$0$","","", r"$3$"])
    a1.yaxis.set_ticks_position('right')
    
    a1.xaxis.set_label_coords(0.75, -0.04)
    a1.yaxis.set_label_coords(1.03, 0.86)
    
    a1.set_ylim(-3,3)
    
    # generate inset    
    x_0 = 0.68
    y_0 = 0.75
    inset_width = 0.3
    inset_height = 0.2  
    
    a1_inset = a1.inset_axes(
    [x_0, y_0, inset_width, inset_height],
    xlim=(0.9, 1.1), ylim=(-3.,3))
    
    a1_inset.tick_params(direction='out', length=4, width=2,  labelsize = label_scale_inset * label_size, pad = 2)
    a1_inset.set_xlabel(r"$\sigma_{xy}$", fontsize = label_scale_inset * label_size)
    a1_inset.set_ylabel(r"$E_\mathrm{F}$", fontsize = label_scale_inset * label_size, rotation='horizontal')        
    
    a1_inset.set_xticks([0, 0.5,  1])
    a1_inset.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
    a1_inset.set_yticks([-3, -2, -1, 0., 1, 2, 3])
    a1_inset.set_yticklabels([r"$-3$", "","", r"$0$","","", r"$3$"])
    
    a1_inset.xaxis.set_label_coords(0.75, -0.1, transform=a1_inset.transAxes)
    a1_inset.yaxis.set_label_coords(-0.1, 0.61, transform=a1_inset.transAxes)
    
    a1_inset.text(0.05, 0.6, r"$W = 1.5$", fontsize =  label_scale_inset * label_size, transform=a1_inset.transAxes)
    #a1_inset.text(0.05, 0.2, r"$\sigma_{xy} = 0.93$", fontsize =  label_scale_inset * label_size, transform=a1_inset.transAxes)
    
    data_inset_1 = np.genfromtxt(location_inset_1 + "/result_sigma_xy.txt", dtype= float, delimiter=' ') 
    data_inset_2 = np.genfromtxt(location_inset_2 + "/result_sigma_xy.txt", dtype= float, delimiter=' ') 
    data_inset_3 = np.genfromtxt(location_inset_3 + "/result_sigma_xy.txt", dtype= float, delimiter=' ') 
    
    n_vals_1 = 30
    n_vals_2 = 7
    n_vals_3 = 50
    
    data_inset_1_mean = np.mean(data_inset_1[:, :n_vals_1], axis = 1)
    data_inset_2_mean = np.mean(data_inset_2[:, :n_vals_2], axis = 1)
    data_inset_3_mean = np.mean(data_inset_3[:, :n_vals_3], axis = 1)
    
    E_vals_inset = np.genfromtxt(location_inset_1 + "/E_vals.txt", dtype= float, delimiter=' ') 
    
    # coarse-grain data
    n_W = len(W_vals)
    n_E = len(E_vals)
    
    E_min = np.min(E_vals)
    E_max = np.max(E_vals)
    
    n_E_coarse = 250
    E_vals_coarse = np.linspace(E_min, E_max, n_E_coarse)
    
    data_coarse_inset_1 = np.zeros(n_E_coarse-1)
    data_coarse_inset_2 = np.zeros(n_E_coarse-1)
    data_coarse_inset_3 = np.zeros(n_E_coarse-1)
    
    # generate index array for coarse graining, such that 
    # E_vals[index_lower_coarse[j]] <= E_vals_coarse[j]
    
    index_lower_coarse = np.zeros(n_E_coarse,  dtype=int)
    
    for l in range(0, n_E_coarse-1):                    
        for j in range(index_lower_coarse [l], n_E):
            if E_vals[j] < E_vals_coarse[l+1]:                      
                index_lower_coarse[l+1] = j 
            else:
                break
            
    index_lower_coarse[-1] = n_E-1
                        
    for j in range(0, n_E_coarse-1):
        ind_low = index_lower_coarse[j]
        ind_high = index_lower_coarse[j+1]
                                      
        data_coarse_inset_1[j] = np.mean(data_inset_1_mean[ind_low:ind_high])
        data_coarse_inset_2[j] = np.mean(data_inset_2_mean[ind_low:ind_high])
        data_coarse_inset_3[j] = np.mean(data_inset_3_mean[ind_low:ind_high])
    
    a1_inset.axvline(x=1, color='black', linestyle='--')
    a1_inset.axvline(x=0.93, color='gray', linestyle=':')
    #a1_inset.axvline(x=0.95, color='black', linestyle='--')
    #a1_inset.axvline(x=0.98, color='black', linestyle='--')
    
    #a1_inset.plot(E_vals_inset, data_inset_1_mean, color = "red", lw = 0.5)# , marker = "o")#, label = r"$\rho_\text{av}$")
    #a1_inset.plot(E_vals_inset, data_inset_2_mean, color = "green", lw = 0.5)#, marker = "^")
    #a1_inset.plot(data_inset_3_mean, E_vals_inset, color = "red", lw = 0.5)#, marker = "x")
    
    #a1_inset.plot(data_coarse_inset_1, E_vals_coarse[:-1], color = "red", lw = 0.5)
    #a1_inset.plot(data_coarse_inset_2, E_vals_coarse[:-1], color = "green", lw = 0.5)
    a1_inset.plot(data_coarse_inset_3, E_vals_coarse[:-1], color = "red", lw = 0.5)
    
    #a1_inset.set_xlim(0.9,1.1)
        
    fig.savefig("Fig_3_publication.png",  bbox_inches='tight', dpi = 600)
    
    folder_coarse_data_line = "Data_Fig_3_line"
    try:
        os.makedirs(folder_coarse_data_line)
    except:
        pass 
    
    np.savetxt(folder_coarse_data_line + "/E_vals.txt", E_vals_coarse[:-1], delimiter=' ')  
    np.savetxt(folder_coarse_data_line + "/result_sigma_xy.txt", data_coarse_inset_3, delimiter=' ')  
            
    plt.show()
    
    
def main():                
    location_1 = "Kubo_final_V3_run_6"
    location_2_p1 = "Kubo_final_V3_run_8"
    location_2_p2 = "Kubo_final_V3_run_7"
    
    output_folder = "Data_Fig_3"
    
    #fuse_data_Fig_3(location_1, location_2_p1, location_2_p2, output_folder)
    
    location_inset_1 = "Kubo_final_DC_CI_V3_line_run_21"
    location_inset_2 = "Kubo_final_DC_CI_V3_line_run_20"
    location_inset_3 = "Kubo_final_DC_CI_V3_line_run_19"
    
    #Fig_3(output_folder)
    Fig_3_ii(location_1, location_inset_1, location_inset_2, location_inset_3)
   

if __name__ == '__main__':
    main()




    