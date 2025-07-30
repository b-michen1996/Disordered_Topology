import csv
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib
import json

import Berry_codes_Fig_4

# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams.update({
    "text.usetex": True,
    #"font.family": "cm"
    "font.family": "times"
})


def Fig_4_App(locations, r, gamma, gamma_2, epsilon_1, epsilon_2, N = 400, add_Q = True):
    """Plot gap and topological index for the spectral localizer."""
    
    E_vals = list()
    bulk_energies = list()
    data_sl_gap = list()
    
    for location in locations:
        E_vals_curr = np.genfromtxt(location + "/E_vals.txt", dtype= float, delimiter=' ')      
        bulk_energies_curr = np.genfromtxt(location + "/bulk_energies.txt", dtype= float, delimiter=' ')      
        data_sl_gap_curr = np.genfromtxt(location + "/result_sl_gap.txt", dtype= float, delimiter=' ')      
        
        try: 
            data_sl_Q_index = np.genfromtxt(location + "/result_sl_Q_index.txt", dtype= float, delimiter=' ')      
            L_data_gap = len(data_sl_gap_curr)
            
            for l in range(L_data_gap):
                int_dev = np.abs(data_sl_Q_index[l, 1] - np.round(data_sl_Q_index[l, 1]))
                                
                if int_dev > 0.01:
                    data_sl_gap_curr[l] = 0
        except:
            pass
                
        E_vals.append(E_vals_curr)
        bulk_energies.append(bulk_energies_curr)
        data_sl_gap.append(data_sl_gap_curr)          
    
    fig, a1 = plt.subplots(1, 1, figsize =(12, 6))
    
    title = location
    #a1.set_title(title , fontsize=30)
    
    # indicate sigma_xy = 0.5
    I_low, I_up, E_low, E_up, energies_band_low, energies_band_up  = Berry_codes_Fig_4.sigma_xy_V3_analytically(N, r, epsilon_1, epsilon_2, gamma, gamma_2, 0)
    
    I_low = np.array(I_low)
    I_up = np.array(I_up)  
    E_low = np.array(E_low)
    E_up = np.array(E_up)  
    
    mask_I_low = I_low > 0.5
    mask_I_up = I_up > 0.5
    
    N_low = len(E_low[mask_I_low])
    N_up = len(E_low[mask_I_up])
        
    a1.fill_betweenx(E_low[mask_I_low], np.zeros(N_low), np.ones(N_low), facecolor = "gray", lw = 0, alpha = 0.2)
    a1.fill_betweenx(E_up[mask_I_up], np.zeros(N_up), np.ones(N_up), facecolor = "gray", lw = 0, alpha = 0.2, label = r"$\sigma_{xy}^{W = 0} > 0.5 e^2 / h$")
           
    label_size = 15
        
    a1.set_xlabel(r"$g_L$", fontsize=label_size)
    a1.set_ylabel(r"$E$", fontsize=label_size, rotation='horizontal')
    
    a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
    
    a1.set_xticks([0, 0.025, 0.05])
    a1.set_xticklabels([r"$0$", r"$0.025$", r"$0.05$"])
    a1.set_yticks([-3, -2, -1, 0., 1, 2, 3])
    a1.set_yticklabels([r"$-3$", "","", r"$0$","","", r"$3$"])
    
    a1.xaxis.set_label_coords(0.87, -0.03)
    a1.yaxis.set_label_coords(-0.02, 0.86)

    x_min_1 = 0
    x_max_1 = np.max(data_sl_gap)

    y_min = np.min(E_vals)
    y_max = np.max(E_vals)

    a1.set_xlim([0, 0.05])
    a1.set_ylim([y_min, y_max])
    
    # plot data for finite size 
    
    legend_entries = ["N = 10", "N = 15", "N = 20", "N = 25"]
    colors = ["b", "orange", "g", "black"]
    linestyles= ["--", ":", "-.", "-"]
    line_widths = [2, 3, 4, 5]
    
    line_widths = [1, 1, 1, 1]
    
    #for l in range(len(locations)):        
    for l in range(4):        
        l = 3 - l
        E_vals_curr = E_vals[l] 
        bulk_energies_curr = bulk_energies[l]
        data_sl_gap_curr = data_sl_gap[l]
        
        a1.plot(data_sl_gap_curr, E_vals_curr, linestyle = linestyles[l], color = colors[l], label = legend_entries[l], lw = line_widths[l])
        #a1.plot(data_sl_gap_curr, E_vals_curr, linestyle = "-", color = colors[l], label = legend_entries[l], lw = line_widths[l])

    a1.scatter(x_min_1 * np.ones(len(bulk_energies[0])), bulk_energies[0], marker = "o", color = "red", label = r"Bulk energies")
    
    a1.legend(loc="upper left", bbox_to_anchor=(0.65, 0.9), fontsize = label_size, facecolor='white', framealpha=1)
    
    if add_Q:
        a1.text(0.05, 0.05, r"$Q = 0$", color = "Black", fontsize = label_size, transform=a1.transAxes, va = "center_baseline")
        a1.text(0.05, 0.5, r"$Q = 0$", color = "Black", fontsize = label_size, transform=a1.transAxes, va = "center_baseline")
        a1.text(0.05, 0.95, r"$Q = 0$", color = "Black", fontsize = label_size, transform=a1.transAxes, va = "center_baseline")
        
        a1.text(0.05, 0.34, r"$Q = -1$", color = "Black", fontsize = label_size, transform=a1.transAxes, va = "center_baseline")
        a1.text(0.05, 0.67, r"$Q = -1$", color = "Black", fontsize = label_size, transform=a1.transAxes, va = "center_baseline")
    
    fig.savefig("Fig_4_App_publication.png",  bbox_inches='tight', dpi = 600)
            
    plt.show()


def main():                
    
    location_1 = "Data_N_10"    
    location_2 = "Data_N_15"    
    location_3 = "Data_N_20"    
    location_4 = "Data_N_25"    
    
    locations = [location_1, location_2, location_3, location_4]
    
    r = 1.5
    epsilon_1 = 0.3
    epsilon_2 = 2

    gamma = 2
    gamma_2 = 0.3
    
    Fig_4_App(locations, r, gamma, gamma_2, epsilon_1, epsilon_2)

if __name__ == '__main__':
    main()





    