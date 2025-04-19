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
        
    im_sigma_xy = a1.imshow(np.abs(data),
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
    a1.set_yticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    a1.yaxis.set_ticks_position('right')
    
    a1.xaxis.set_label_coords(0.9, -0.025)
    a1.yaxis.set_label_coords(1.025, 0.86)
        
    fig.savefig("Fig_3_publication.png",  bbox_inches='tight', dpi = 600)
    
    plt.show()
    
    
def main():                
    location = "Kubo_final_V3_run_5"
    
    Fig_3(location)
   

if __name__ == '__main__':
    main()




    