import numpy as np
from matplotlib import pyplot as plt


# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})
    
   
def Fig_3(location_data_contour, location_data_inset):
    """Generate plot for Fig_3."""    
    E_vals = np.genfromtxt(location_data_contour + "/E_vals.txt", dtype= float, delimiter=' ') 
    W_vals = np.genfromtxt(location_data_contour + "/W_vals.txt", dtype= float, delimiter=' ') 
    data = np.genfromtxt(location_data_contour + "/result_sigma_xy.txt", dtype= float, delimiter=' ')    
    
    fig, a1 = plt.subplots(1, 1, figsize =(12, 6))
    
    label_size = 25
    label_scale_inset = 0.72
    
    im_sigma_xy = a1.imshow(data,
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

    data_inset = np.genfromtxt(location_data_inset + "/result_sigma_xy.txt", dtype= float, delimiter=' ') 
    
    E_vals_inset = np.genfromtxt(location_data_inset + "/E_vals.txt", dtype= float, delimiter=' ') 
        
    a1_inset.axvline(x=1, color='black', linestyle='--')
    a1_inset.axvline(x=0.93, color='gray', linestyle=':')

    a1_inset.plot(data_inset, E_vals_inset, color = "red", lw = 0.5)
    
    
    fig.savefig("Fig_3_publication.png",  bbox_inches='tight', dpi = 600, transparent = True)
                
    plt.show()
    
    
location_data_contour = "Data_Fig_3_contourplot"
location_data_inset = "Data_Fig_3_line"
    
Fig_3(location_data_contour, location_data_inset)




    