import numpy as np
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
    data_a = np.genfromtxt(location_a + "/sigma_xx.txt", dtype= float, delimiter=' ')  
    
    E_vals_b = np.genfromtxt(location_b + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals_b = np.genfromtxt(location_b + "/W_vals.txt", dtype= float, delimiter=' ') 
    data_b = np.genfromtxt(location_b + "/sigma_xx.txt", dtype= float, delimiter=' ')  
    
    W_array_a, E_array_a = np.meshgrid(W_vals_a, E_vals_a)
    W_array_b, E_array_b = np.meshgrid(W_vals_b, E_vals_b)
    
    #fig, axs = plt.subplots(1, 3, figsize =(12, 6), width_ratios = [0.05,1,1])
    fig, axs = plt.subplots(1, 2, figsize =(12, 6))
    fig.subplots_adjust(wspace=0.1)
    #a0 = axs[0]
    a1 = axs[0]
    a2 = axs[1]
    
    label_size = 25
    label_scale_inset = 0.72
        
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
    
    cbar_title = r"$\sigma_{xx}[e^2/h]$"
    
    cbar = plt.colorbar(im_G_a, shrink=0.6, ticks=[0, 1, 2], ax=axs.ravel().tolist(), location='left', pad = 0.06)
    cbar.ax.tick_params(labelsize=label_size) 
    cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25)
    
    
    a1.set_xlabel(r"$W$", fontsize=label_size)
    a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
        
    a1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    a1.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    a1.set_yticks([])
    
    a1.xaxis.set_label_coords(0.75, -0.04)
    
    a2.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
    a2.set_xticks([0, 1, 2, 3, 4, 5, 6])
    a2.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    
    a2.set_xlabel(r"$W$", fontsize=label_size)
    a2.set_ylabel(r"$E_\mathrm{F}$", fontsize=label_size, rotation='horizontal')
    
    a2.set_yticks([-3, -2, -1, 0., 1, 2, 3])
    a2.set_yticklabels([r"$-3$", "","", r"$0$","","", r"$3$"])
    a2.yaxis.set_ticks_position('right')
    
    a2.xaxis.set_label_coords(0.75, -0.04)
    a2.yaxis.set_label_coords(1.06, 0.86)

    color_wnum = "blue"    

    a1.text(0.5,0.94, r"(a)", fontsize = label_size, transform=a1.transAxes)
    a1.text(0.7, 0.92, r"$y$-OBC", color = "gray", fontsize = 0.75 * label_size, transform=a1.transAxes, va = "center_baseline")
    
    a2.text(0.5, 0.94, r"(b)", fontsize = label_size, transform=a2.transAxes)
    
    a2.text(0.15, 0.5, r"$\nu = 0$", color = color_wnum, fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
    a2.plot(0.1, 0.5, marker = "o", color = color_wnum, transform=a2.transAxes, markersize = 5)
    
    a2.text(0.43, 0.66, r"$\nu = 1$", color = color_wnum, fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
    a2.plot(0.4, 0.66666, marker = "o", color = color_wnum, transform=a2.transAxes, markersize = 5)
    
    a2.text(0.43, 0.333333, r"$\nu = 1$", color = color_wnum, fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
    a2.plot(0.4, 0.333333, marker = "o", color = color_wnum, transform=a2.transAxes, markersize = 5)
    
    a2.text(0.7, 0.92, r"$y$-PBC", color = "gray", fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
    
    # generate inset
    x_data_inset_a = np.genfromtxt(location_data_inset_b + "/Nx_vals.txt", dtype= float, delimiter=' ') 
    data_inset_a = np.genfromtxt(location_data_inset_a + "/sigma_xx.txt", dtype= float, delimiter=' ') 
    
    data_inset_a_mean = np.mean(data_inset_a, axis = 1)
    data_inset_a_std = np.std(data_inset_a, axis = 1)
    
    x_data_inset_b = np.genfromtxt(location_data_inset_b + "/Nx_vals.txt", dtype= float, delimiter=' ') 
    data_inset_b = np.genfromtxt(location_data_inset_b + "/sigma_xx.txt", dtype= float, delimiter=' ') 
    
    data_inset_b_mean = np.mean(data_inset_b, axis = 1)
    data_inset_b_std = np.std(data_inset_b, axis = 1)
            
    x_0 = 0.5
    y_0 = 0.025
    inset_width = 0.48
    inset_height = 0.12
        
    a1_inset = a1.inset_axes(
    [x_0, y_0, inset_width, inset_height],
    xlim=(400, 3000), ylim=(0, 2))
    
    a2_inset = a2.inset_axes(
    [x_0, y_0, inset_width, inset_height],
    xlim=(400, 3000), ylim=(0, 2))
    
    a1_inset.fill_between(x_data_inset_a, data_inset_a_mean - data_inset_a_std, data_inset_a_mean + data_inset_a_std, facecolor = "red", lw = 0, alpha = 0.2)
    a1_inset.plot(x_data_inset_a, data_inset_a_mean, color = "red", lw = 1)
    a1_inset.axhline(y=1, color='black', linestyle='--')
    
    a2_inset.fill_between(x_data_inset_b, data_inset_b_mean - data_inset_b_std, data_inset_b_mean + data_inset_b_std, facecolor = "red", lw = 0, alpha = 0.2)
    a2_inset.plot(x_data_inset_b, data_inset_b_mean, color = "red", lw = 1)
    a2_inset.axhline(y=1, color='black', linestyle='--')
    
    a1_inset.set_xlabel(r"$10^{-3} N_{x}$", fontsize = label_scale_inset * label_size)    
    a1_inset.set_ylabel(r"$\sigma_{xx}$", fontsize = label_scale_inset * label_size, rotation='horizontal')
    a1_inset.xaxis.set_label_coords(0.8, 2.05)
    a1_inset.yaxis.set_label_coords(-0.25, 0.1)
    
    a2_inset.set_xlabel(r"$10^{-3} N_{x}$", fontsize = label_scale_inset * label_size)    
    a2_inset.set_ylabel(r"$\sigma_{xx}$", fontsize = label_scale_inset * label_size, rotation='horizontal')
    a2_inset.xaxis.set_label_coords(0.8, 2.05)
    a2_inset.yaxis.set_label_coords(-0.25, 0.1)

    a1_inset.tick_params(direction='out', length=4, width=2,  labelsize = label_scale_inset * label_size, pad = 1)
    a1_inset.xaxis.set_ticks_position('top')
    
    a2_inset.tick_params(direction='out', length=4, width=2,  labelsize = label_scale_inset * label_size, pad = 1)
    a2_inset.xaxis.set_ticks_position('top')
    
    a1_inset.minorticks_off()
    a2_inset.minorticks_off()
    
    a1_inset.set_xticks([500, 1000, 1500, 2000, 2500, 3000])
    a1_inset.set_xticklabels(["", r"$1$", "", r"$2$","", r"$3$"])
    
    a1_inset.set_yticks([0, 1, 2])
    a1_inset.set_yticklabels([r"$0$", r"$1$", r"$2$"])
    
    a1_inset.text(0.2, 0.1, r"$E_\mathrm{F} = W = 1$", fontsize = label_scale_inset * label_size, transform=a1_inset.transAxes)
    
    a2_inset.set_xticks([500, 1000, 1500, 2000, 2500, 3000])
    a2_inset.set_xticklabels(["", r"$1$", "", r"$2$","", r"$3$"])
    
    a2_inset.set_yticks([0, 1, 2])
    a2_inset.set_yticklabels([r"$0$", r"$1$", r"$2$"])
    
    a2_inset.text(0.2, 0.625, r"$E_\mathrm{F} = W = 1$", fontsize = label_scale_inset * label_size, transform=a2_inset.transAxes)
    
    fig.savefig("Fig_2_publication.png",  bbox_inches='tight', dpi = 300)
    
    plt.show()
    

def main():                    
    location_data_inset_a = "Data_Fig_2_a_inset"
    location_data_inset_b = "Data_Fig_2_b_inset"
    
    Data_Fig_2_a = "Data_Fig_2_a"
    Data_Fig_2_b = "Data_Fig_2_b"
        
    Fig_2(Data_Fig_2_a, Data_Fig_2_b, location_data_inset_a, location_data_inset_b)
  
    
if __name__ == '__main__':
    main()




    
