import numpy as np
from matplotlib import pyplot as plt


# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})


def H_Fulga_Bergholtz(kx, ky, v = 0.5):
    """Bloch Hamiltonian for Chern insulator"""
    h_11 = 2 * (np.cos(kx) - np.cos(ky))
    
    h_12 = np.sqrt(2) * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx)
            + np.exp(1j * ky) +  1j * np.exp(1j * (kx + ky)) + 1) 

    h_12 = np.sqrt(2) * np.exp(-1j * np.pi / 4) * (np.exp(-1j * kx)
            + np.exp(-1j * ky) +  1j * (np.exp(-1j * (kx + ky)) + 1))
    
    res = np.array([[h_11, h_12, v],
                [np.conj(h_12), -h_11, 0],
                [v, 0, 0]])
    
    return res
           

def Berry_curvature(N, H_bloch):
    """Calculate berry curvature of H_bloch and return it as an array along with k_x, k_y values suitable for contour plot."""
    k_vals = np.linspace(-np.pi, np.pi, N, endpoint= False)
    
    kx_array, ky_array = np.meshgrid(k_vals,k_vals)
    
    n_band = H_bloch(0,0).shape[0]

    energies =  np.zeros((N,N,n_band))
    ES = 1j * np.zeros((N,N,n_band, n_band))

    Phi_vals =  np.zeros((N,N,n_band))
    
    # calculate all eigenstates
    for jx in range(N):        
        for jy in range(N):            
            kx = kx_array[jy,jx] 
            ky = ky_array[jy,jx] 

            H_k = H_bloch(kx, ky)

            E_k, EV_k = np.linalg.eigh(H_k)

            energies[jy, jx, :] = E_k
            ES[jy, jx, :, :] = EV_k     

    # calculate Berry curvature
    for jx in range(N):
        jx_p1 = (jx + 1) % N
        for jy in range(N):
            jy_p1 = (jy + 1) % N

            p_vals = [(jy,jx), (jy, jx_p1), (jy_p1, jx_p1), (jy_p1, jx)]
            
            # generate array of products for each band
            prod_arrray = np.einsum("jk,jk->k", np.conj(ES[p_vals[0]]), ES[p_vals[1]])

            for l in range(1, 4):
                l_p1 = (l+1)%4
                prod_arrray = prod_arrray * np.einsum("jk,jk->k", np.conj(ES[p_vals[l]]), ES[p_vals[l_p1]])

            # calculate flux through plaquette
            Phi_vals[jy, jx, :] = -np.angle(prod_arrray)
    
    # calculate chern number for each band
    C_n = (1 / (2 * np.pi)) * np.sum(Phi_vals, axis = (0,1)) 

    # give back results and arrays of k_vals
    return ((N / (2 * np.pi)) ** 2) * Phi_vals, C_n, kx_array, ky_array, energies


def Berry_curvature_integral_FB(N, v):
    """Integral of Berry curvature up to Fermi energy"""
    H_bloch = lambda kx, ky: H_Fulga_Bergholtz(kx, ky, v)
    Phi_vals, C_n, k_x_vals, k_y_vals, energies = Berry_curvature(N, H_bloch)
    
    n_band = H_bloch(0,0).shape[0]
    
    data = np.zeros((N ** 2, 2, n_band))
    
    # Flatten data for Berry curvature and energy and multiply by prefactor  delta_k^2 / (2 pi)
    prefactor = (2 * np.pi / N) ** 2
    
    for j_band in range(n_band):
        # get list of Berry curvatures values and energies in each cell
        Berry_curvature_list = prefactor * Phi_vals[:,:,j_band].flatten()
        Energy_list = energies[:,:,j_band].flatten()
        
        # get index to sort by energy
        idx = np.argsort(Energy_list)
        
        # save to data
        data[:,0,j_band] = Berry_curvature_list[idx]
        data[:,1,j_band] = Energy_list[idx]
        
    # Integrate    
    for j in range(N**2):
        try:
            data[j, 0, :] += data[j-1, 0, :]
        except:
            pass    
    
    data[:,0,:] = (1 / (2 * np.pi)) * data[:,0,:]
    
    len_data = len(data[:,0,0])
    
    energies_total = np.concatenate((data[:,1,0], data[:,1,1], data[:,1,2]))
    sigma_xy_total = np.concatenate((data[:,0,0], data[-1,0,0] * np.ones(len_data) +  data[:,0,1], 
                               (data[-1,0,0] + data[-1,0,1]) * np.ones(len_data) + data[:,0,2]))
        
    return energies_total, sigma_xy_total, data[:,1,:]


def Fig_5_App(v_vals, N = 100):
    """Line plot for Hall conductance over E, obtained from averaged coductance matrix"""
    fig = plt.figure(figsize=(12, 6), layout = "tight")
    
    label_size = 15
    
    sub_fig_labels = [r"(a)", r"(b)", r"(c)", r"(d)"]
    
    for l in range(0, 4):
        v_l = v_vals[l]
        energies_l, sigma_xy_l, e_band_l = Berry_curvature_integral_FB(N, v_l)
        
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
        
        al.fill_betweenx(e_band_l[:,0], np.zeros(len(e_band_l[:, 0])), 2 * np.ones(len(e_band_l[:, 0])), facecolor = "red", lw = 0, alpha = 0.2)
        al.fill_betweenx(e_band_l[:,1], np.zeros(len(e_band_l[:, 1])), 2 * np.ones(len(e_band_l[:, 1])), facecolor = "red", lw = 0, alpha = 0.2)
        al.fill_betweenx(e_band_l[:,2], np.zeros(len(e_band_l[:, 2])), 2 * np.ones(len(e_band_l[:, 2])), facecolor = "red", lw = 0, alpha = 0.2,
                         label = r"Bulk energies")
        
        al.set_ylim([-9, 9])
        al.set_xlim([0, 1.1])
        
        if l == 0:
            al.legend(loc="center left", bbox_to_anchor=(0., 0.5), fontsize = label_size, facecolor='white', framealpha=1)
        
        al.text(0.2, 0.9, sub_fig_labels[l], fontsize = label_size, transform=al.transAxes)
        al.text(0.6, 0.9, rf"$v = {v_l}$", fontsize = label_size, transform=al.transAxes)
                    
    plt.show()
    
    fig.tight_layout()
    fig.savefig("Fig_5_App_publication.png",  bbox_inches='tight', dpi = 600)
    
    return


def main():                       
    v_vals = [2, 3.3, 4.5, 6.5]    
    
    N = 200
    
    Fig_5_App(v_vals, N)
    

if __name__ == '__main__':
    main()




    