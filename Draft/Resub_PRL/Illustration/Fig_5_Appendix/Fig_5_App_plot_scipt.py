import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import Berry_codes_Fig_5_App

# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})
           


def Fig_5_App(v_vals, N = 100):
    """Line plot for Hall conductance over E, obtained from averaged coductance matrix"""
    fig = plt.figure(figsize=(12, 6), layout = "tight")
    
    label_size = 15
    
    sub_fig_labels = [r"(a)", r"(b)", r"(c)", r"(d)"]
    
    for l in range(0, 4):
        v_l = v_vals[l]
        energies_l, sigma_xy_l, e_band_l = Berry_codes_Fig_5_App.Berry_curvature_integral_FB(N, v_l)
        
        al = fig.add_subplot(2, 2, l + 1)        
        al.set_ylabel(r"$E_\mathrm{F}$", fontsize=label_size, rotation='horizontal')
        al.set_xlabel(r"$\sigma_{xy}(E_\mathrm{F})[e^2/h]$", fontsize=label_size)
        al.tick_params(direction='in', length=6, width=2,  labelsize = 20, pad = 5)   
                
        plt.axvline(x=1, color='gray', linestyle=':')
        plt.axvline(x=0.5, color='gray', linestyle=':')
        
        al.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
        al.set_xticks([0, 0.5, 1])
        al.set_yticks([-9, -6, -3, 0, 3, 6, 9])
        
        al.set_xticklabels([r"$0$", r"$0.5$", r"$1$"])
        al.set_yticklabels([r"$-9$", "", "", r"$0$", "", "", r"$9$"])
        
        al.xaxis.set_label_coords(0.75, -0.05)
        al.yaxis.set_label_coords(-0.05, 0.75)
            
        # plot result
        al.plot(sigma_xy_l, energies_l, color = "black", label = r"$\sigma_{xy}$")
        #al.scatter(0 * energies_l, energies_l, marker = "o", color = "red", label = r"Bulk energies")
        
        al.fill_betweenx(e_band_l[:,0], np.zeros(len(e_band_l[:, 0])), 2 * np.ones(len(e_band_l[:, 0])), facecolor = "red", lw = 0, alpha = 0.2)
        al.fill_betweenx(e_band_l[:,1], np.zeros(len(e_band_l[:, 1])), 2 * np.ones(len(e_band_l[:, 1])), facecolor = "red", lw = 0, alpha = 0.2)
        al.fill_betweenx(e_band_l[:,2], np.zeros(len(e_band_l[:, 2])), 2 * np.ones(len(e_band_l[:, 2])), facecolor = "red", lw = 0, alpha = 0.2,
                         label = r"Bulk energies")
        #a1.fill_betweenx(E_low[mask_I_low], np.zeros(N_low), np.ones(N_low), facecolor = "gray", lw = 0, alpha = 0.2)
        
        #a1.fill_betweenx(E_up[mask_I_up], np.zeros(N_up), np.ones(N_up), facecolor = "gray", lw = 0, alpha = 0.2, label = r"$\sigma_{xy}^{W = 0} > 0.5 e^2 / h$")
        
        al.set_ylim([-9, 9])
        al.set_xlim([0, 1.1])
        
        if l == 0:
            al.legend(loc="center left", bbox_to_anchor=(0., 0.5), fontsize = label_size, facecolor='white', framealpha=1)
        
        al.text(0.2, 0.9, sub_fig_labels[l], fontsize = label_size, transform=al.transAxes)
        al.text(0.6, 0.9, rf"$v = {v_l}$", fontsize = label_size, transform=al.transAxes)
        
        """
        for j in range(n_bands):
            al.plot(data_l[:,1,j], data_l[:,0,j], color = "black")
        
        #al.set_ylim([-8, 8])
        #al.set_xlim([0, 1])
        
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
        """
            
    plt.show()
    
    fig.tight_layout()
    fig.savefig("Fig_5_App_publication.png",  bbox_inches='tight', dpi = 600)
    
    return


def main():                       
    
    v_vals = [2, 3.3, 4.5, 6.5]
    
    y_lims = [[]]
    
    N = 200
    
    Fig_5_App(v_vals, N)

    
    

if __name__ == '__main__':
    main()




    