import csv
import numpy as np

#import matplotlib.colors as colors
from matplotlib import pyplot as plt



# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})



def Fig_2(location_a, location_b, location_data_inset_a, location_data_inset_b):
    """Generate plot for Fig_2."""
    
    E_vals_a = np.genfromtxt(location_a + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals_a = np.genfromtxt(location_a + "/W_vals.txt", dtype= float, delimiter=' ') 
    data_a = np.genfromtxt(location_a + "/result.txt", dtype= float, delimiter=' ')  
    
    E_vals_b = np.genfromtxt(location_b + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals_b = np.genfromtxt(location_b + "/W_vals.txt", dtype= float, delimiter=' ') 
    data_b = np.genfromtxt(location_b + "/result.txt", dtype= float, delimiter=' ')  

    W_array_a, E_array_a = np.meshgrid(W_vals_a, E_vals_a)
    W_array_b, E_array_b = np.meshgrid(W_vals_b, E_vals_b)
    
    #fig, axs = plt.subplots(1, 3, figsize =(12, 6), width_ratios = [0.05,1,1])
    fig, axs = plt.subplots(1, 2, figsize =(12, 6))
    fig.subplots_adjust(wspace=0.1)
    #a0 = axs[0]
    a1 = axs[0]
    a2 = axs[1]
    
    label_size = 25
    label_scale_inset = 0.6
    
    n_levels = 1000
    vmin = 0.
    vmax = 2    
        
    im_G_a = a1.imshow(data_a,
               aspect='auto',
               origin='lower',
               extent=[np.min(W_vals_a), np.max(W_vals_a), np.min(E_vals_a), np.max(E_vals_a)],
               vmin = 0,
               vmax = 2,
               cmap=plt.cm.magma.reversed()
              )
    
    im_G_b = a2.imshow(data_b,
               aspect='auto',
               origin='lower',
               extent=[np.min(W_vals_b), np.max(W_vals_b), np.min(E_vals_b), np.max(E_vals_b)],
               vmin = 0,
               vmax = 2,
               cmap=plt.cm.magma.reversed()
               
              )
    
    cbar_title = r"$\sigma_{xx}[e^2/\hbar]$"
    
    cbar = plt.colorbar(im_G_a, shrink=0.6, ticks=[0, 1, 2], ax=axs.ravel().tolist(), location='left', pad = 0.06)
    cbar.ax.tick_params(labelsize=label_size) 
    cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25)
    
    
    a1.set_xlabel(r"$W$", fontsize=label_size)
    a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
        
    a1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    a1.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    a1.set_yticks([])
    
    a1.xaxis.set_label_coords(0.9, -0.025)
    
    a2.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
    a2.set_xticks([0, 1, 2, 3, 4, 5, 6])
    a2.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    
    a2.set_xlabel(r"$W$", fontsize=label_size)
    a2.set_ylabel(r"$E_\mathrm{F}$", fontsize=label_size, rotation='horizontal')
    
    a2.set_yticks([-3, -2, -1, 0., 1, 2, 3])
    a2.set_yticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    a2.yaxis.set_ticks_position('right')
    
    a2.xaxis.set_label_coords(0.9, -0.025)
    a2.yaxis.set_label_coords(1.06, 0.86)

    color_wnum = "blue"    

    a1.text(0.5,0.94, r"a)", fontsize = label_size, transform=a1.transAxes)
    a1.text(0.8, 0.9, r"OBC", color = "gray", fontsize = 0.75 * label_size, transform=a1.transAxes, va = "center_baseline")
    
    a2.text(0.5, 0.94, r"b)", fontsize = label_size, transform=a2.transAxes)
    
    a2.text(0.15, 0.5, r"$\nu = 0$", color = color_wnum, fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
    a2.plot(0.12, 0.5, marker = "o", color = color_wnum, transform=a2.transAxes, markersize = 5)
    
    a2.text(0.43, 0.66, r"$\nu = 1$", color = color_wnum, fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
    a2.plot(0.4, 0.66, marker = "o", color = color_wnum, transform=a2.transAxes, markersize = 5)
    
    a2.text(0.43, 0.34, r"$\nu = 1$", color = color_wnum, fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
    a2.plot(0.4, 0.34, marker = "o", color = color_wnum, transform=a2.transAxes, markersize = 5)
    
    a2.text(0.8, 0.9, r"PBC", color = "gray", fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
    
    # generate inset
    
    x_data_inset_a = np.genfromtxt(location_data_inset_b + "/Nx_vals.txt", dtype= float, delimiter=' ') 
    data_inset_a = np.genfromtxt(location_data_inset_a + "/result.txt", dtype= float, delimiter=' ') 
    
    data_inset_a = data_inset_a[:, :7]
    
    data_inset_a_mean = np.mean(data_inset_a, axis = 1)
    data_inset_a_std = np.std(data_inset_a, axis = 1)
    
    x_data_inset_b = np.genfromtxt(location_data_inset_b + "/Nx_vals.txt", dtype= float, delimiter=' ') 
    data_inset_b = np.genfromtxt(location_data_inset_b + "/result.txt", dtype= float, delimiter=' ') 
    
    data_inset_b = data_inset_b[:, :7]
    
    data_inset_b_mean = np.mean(data_inset_b, axis = 1)
    data_inset_b_std = np.std(data_inset_b, axis = 1)
            
    x_0 = 0.58
    y_0 = 0.025
    inset_width = 0.4
    inset_height = 0.1
        
    a1_inset = a1.inset_axes(
    [x_0, y_0, inset_width, inset_height],
    xlim=(400, 2500), ylim=(0, 2))
    
    a2_inset = a2.inset_axes(
    [x_0, y_0, inset_width, inset_height],
    xlim=(400, 2500), ylim=(0, 2))
    
    a1_inset.fill_between(x_data_inset_a, data_inset_a_mean - data_inset_a_std, data_inset_a_mean + data_inset_a_std, facecolor = "gray", lw = 0)
    a1_inset.plot(x_data_inset_a, data_inset_a_mean, color = "red")
    a1_inset.axhline(y=1, color='black', linestyle='--')
    
    a2_inset.fill_between(x_data_inset_b, data_inset_b_mean - data_inset_b_std, data_inset_b_mean + data_inset_b_std, facecolor = "gray", lw = 0)
    a2_inset.plot(x_data_inset_b, data_inset_b_mean, color = "red")
    a2_inset.axhline(y=1, color='black', linestyle='--')
    
    a1_inset.set_xlabel(r"$N_{x}$", fontsize = label_scale_inset * label_size)    
    a1_inset.set_ylabel(r"$\sigma_{xx}$", fontsize = label_scale_inset * label_size, rotation='horizontal')
    a1_inset.xaxis.set_label_coords(0.8, 2.05)
    a1_inset.yaxis.set_label_coords(-0.25, 0.1)
    
    a2_inset.set_xlabel(r"$N_{x}$", fontsize = label_scale_inset * label_size)    
    a2_inset.set_ylabel(r"$\sigma_{xx}$", fontsize = label_scale_inset * label_size, rotation='horizontal')
    a2_inset.xaxis.set_label_coords(0.8, 2.05)
    a2_inset.yaxis.set_label_coords(-0.25, 0.1)

    a1_inset.tick_params(direction='out', length=4, width=2,  labelsize = label_scale_inset * label_size, pad = 1)
    a1_inset.xaxis.set_ticks_position('top')
    
    a2_inset.tick_params(direction='out', length=4, width=2,  labelsize = label_scale_inset * label_size, pad = 1)
    a2_inset.xaxis.set_ticks_position('top')
    
    a1_inset.minorticks_off()
    a2_inset.minorticks_off()
    
    a1_inset.set_xticks([600, 1200, 1800])
    a1_inset.set_xticklabels([r"$6\mathrm{e}2$", r"$1.2\mathrm{e}3$", r"$1.8\mathrm{e}3$"])
    
    a1_inset.set_yticks([0, 1, 2])
    a1_inset.set_yticklabels([r"$0$", r"$1$", r"$2$"])
    
    a1_inset.text(0.2, 0.62, r"$E_\mathrm{F} = -1$", fontsize = label_scale_inset * label_size, transform=a1_inset.transAxes)
    
    a2_inset.set_xticks([600, 1200, 1800])
    a2_inset.set_xticklabels([r"$6\mathrm{e}2$", r"$1.2\mathrm{e}3$", r"$1.8\mathrm{e}3$"])
    
    a2_inset.set_yticks([0, 1, 2])
    a2_inset.set_yticklabels([r"$0$", r"$1$", r"$2$"])
    
    a2_inset.text(0.2, 0.625, r"$E_\mathrm{F} = W = 1$", fontsize = label_scale_inset * label_size, transform=a2_inset.transAxes)
    
    fig.savefig("Fig_2_publication.png",  bbox_inches='tight', dpi = 600)
    
    plt.show()
    
        
def Fig_2_vertical(location_a, location_b):
    """Generate plot for Fig_2, vertical alignment."""
    
    E_vals_a = np.genfromtxt(location_a + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals_a = np.genfromtxt(location_a + "/W_vals.txt", dtype= float, delimiter=' ') 
    data_a = np.genfromtxt(location_a + "/result.txt", dtype= float, delimiter=' ')  
    
    E_vals_b = np.genfromtxt(location_b + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals_b = np.genfromtxt(location_b + "/W_vals.txt", dtype= float, delimiter=' ') 
    data_b = np.genfromtxt(location_b + "/result.txt", dtype= float, delimiter=' ')  

    W_array_a, E_array_a = np.meshgrid(W_vals_a, E_vals_a)
    W_array_b, E_array_b = np.meshgrid(W_vals_b, E_vals_b)
    
    #fig, axs = plt.subplots(1, 3, figsize =(12, 6), width_ratios = [0.05,1,1])
    fig, axs = plt.subplots(2, 1, figsize =(12, 24))
    fig.subplots_adjust(wspace=0.1)
    #a0 = axs[0]
    a1 = axs[0]
    a2 = axs[1]
    
    label_size = 25
    
    
    n_levels = 1000
    vmin = 0.
    vmax = 2    
        
    im_G_a = a1.imshow(data_a,
               aspect='auto',
               origin='lower',
               extent=[np.min(W_vals_a), np.max(W_vals_a), np.min(E_vals_a), np.max(E_vals_a)],
               vmin = 0,
               vmax = 2,
               cmap=plt.cm.magma.reversed()
              )
    
    im_G_b = a2.imshow(data_b,
               aspect='auto',
               origin='lower',
               extent=[np.min(W_vals_b), np.max(W_vals_b), np.min(E_vals_b), np.max(E_vals_b)],
               vmin = 0,
               vmax = 2,
               cmap=plt.cm.magma.reversed()
               
              )
    
    cbar_title = r"$\sigma_{xx}[e^2/\hbar]$"
    

    cbar = plt.colorbar(im_G_a, shrink=0.6, ticks=[0, 1, 2], ax=axs.ravel().tolist(), location='left', pad = 0.051)
    cbar.ax.tick_params(labelsize=label_size) 
    cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25)
    
    
    a1.set_xlabel("W", fontsize=label_size)
    a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
        
    a1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    a1.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    a1.set_yticks([])
    
    a1.xaxis.set_label_coords(0.9, -0.025)
    
    a2.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
    a2.set_xticks([0, 1, 2, 3, 4, 5, 6])
    a2.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    
    a2.set_xlabel("W", fontsize=label_size)
    a2.set_ylabel("E", fontsize=label_size, rotation='horizontal')
    
    a2.set_yticks([-3, -2, -1, 0., 1, 2, 3])
    a2.set_yticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    a2.yaxis.set_ticks_position('right')
    
    a2.xaxis.set_label_coords(0.9, -0.025)
    a2.yaxis.set_label_coords(1.05, 0.86)
    
    a1.text(0.5,0.94, r"a)", fontsize = label_size, transform=a1.transAxes)
    a2.text(0.5, 0.94, r"b)", fontsize = label_size, transform=a2.transAxes)
    
    a2.text(0.04, 0.5, r"$\nu = 0$", color = "gray", fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
    a2.plot(0.25, 0.5, marker = "o", color = "gray", transform=a2.transAxes, markersize = 5)
    
    fig.savefig("Fig_2_publication.png",  bbox_inches='tight')
    
    plt.show()
    

def main():                
    location_G_final_V3_a = "G_final_V3_results/G_final_V3_run_3"
    location_G_final_V3_b = "G_final_V3_results/G_final_V3_run_4"
    
    location_data_inset_a = "G_final_V3_size_scaling_run_10"
    location_data_inset_b = "G_final_V3_size_scaling_run_11"
        
    Fig_2(location_G_final_V3_a, location_G_final_V3_b, location_data_inset_a, location_data_inset_b)
    #Fig_2_vertical(location_G_final_V3_a, location_G_final_V3_b)
    

if __name__ == '__main__':
    main()




    
