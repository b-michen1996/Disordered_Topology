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
    "font.family": "cm"
    
})


def Fig_4(location):
    """Plot gap and topological index for the spectral localizer."""
    
    E_vals = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')      
    bulk_energies = np.genfromtxt(location + "/bulk_energies.txt", dtype= float, delimiter=' ')      
    data_sl_gap = np.genfromtxt(location + "/result_sl_gap.txt", dtype= float, delimiter=' ')      
    data_sl_Q_index = np.genfromtxt(location + "/result_sl_Q_index.txt", dtype= float, delimiter=' ')      
    
    fig, axs = plt.subplots(1, 2, figsize =(12, 6))
    fig.subplots_adjust(wspace=0.1)
    
    a1 = axs[0]
    a2 = axs[1]
    
    label_size = 25
        
    a1.set_xlabel(r"$g_{\mathcal L}$", fontsize=label_size)
    a1.set_ylabel(r"$E$", fontsize=label_size)

    x_min_1 = 0
    x_max_1 = np.max(data_sl_gap)

    y_min = np.min(E_vals)
    y_max = np.max(E_vals)

    a1.set_xlim([x_min_1, x_max_1])
    a1.set_ylim([y_min, y_max])
    
    a1.plot(data_sl_gap, E_vals, linestyle = "--", color = "black")

    a1.scatter(x_min_1 * np.ones(len(bulk_energies)), bulk_energies, marker = "o", color = "red")
        
    a2.set_xlabel(r"$Q$", fontsize=label_size)
    
    a2.axvline(0, color = "black")

    x_min_2 = 1.2 * np.min(data_sl_Q_index[:,1])
    x_max_2 = 1.2 * np.max(data_sl_Q_index[:,1])

    a2.set_xlim([x_min_2, x_max_2])
    a2.set_ylim([y_min, y_max])
    
    a2.scatter(data_sl_Q_index[:,1], data_sl_Q_index[:,0], marker = "o", color = "black")
    
    #a1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    #a1.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
    
    a1.text(0.1,0.94, r"a)", fontsize = label_size, transform=a1.transAxes)
    
    a2.text(0.1, 0.94, r"b)", fontsize = label_size, transform=a2.transAxes)
    
    fig.savefig("Fig_4_publication.png",  bbox_inches='tight', dpi = 600)
            
    plt.show()


def main():                
    location = "s_l_final_V3_energy_run_3"
    

    
    Fig_4(location)


if __name__ == '__main__':
    main()




    