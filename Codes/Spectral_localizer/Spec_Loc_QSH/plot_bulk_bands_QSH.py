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
    "font.family": "times"
})


def H_QSH_k(kx, ky, alpha, beta, gamma, m):
        """Bloch Hamiltonian"""
        epsilon = 2 * gamma * (2 - np.cos(kx) - np.cos(ky))

        dx_m_idy = alpha * (np.sin(kx) + 1j * np.sin(ky))
        dz = m + 2 * beta * (2 - np.cos(kx) - np.cos(ky))                                   
        
        res = np.array([[epsilon + dz, dx_m_idy],
                  [np.conj(dx_m_idy), epsilon - dz]])
        
        return res 


def plot_bulk_bands_QSH(Nx, Ny, alpha, beta, gamma, m):
    """Plot bulk bands."""

    bulk_energies = np.zeros((Nx * Ny, 4))
    for jx in range(0, Nx):
                kx = -np.pi + jx * 2 * np.pi / Nx
                for jy in range(0, Ny):
                        ky = -np.pi + jy * 2 * np.pi / Ny

                        H_k_curr = H_QSH_k(kx, ky, alpha, beta, gamma, m)                      

                        j = jx * Ny +jy
                        
                        bulk_energies[j, :2] = kx, ky
                        bulk_energies[j, 2:] = np.linalg.eigvalsh(H_k_curr)   

    title = rf"Bulk bands for $\alpha = {alpha}$, $\beta = {beta}$, $\gamma = {gamma}$, $m = {m}$, $N_x = {Nx}$, $N_y = {Ny}$"

    fig = plt.figure(figsize=(24, 12), layout = "tight")
    a1 =  fig.add_subplot(1,1,1)
    a1.set_title(title , fontsize=30)
    a1.set_xlabel(r"$k_x$", fontsize=30)
    a1.set_ylabel("E", fontsize=30)

    x_min = -np.pi
    x_max = np.pi

    a1.set_xlim([x_min, x_max])

    a1.scatter(bulk_energies[:,0], bulk_energies[:,2], marker = "o", color = "red", label = r"$E_1(k_x)$")
    a1.scatter(bulk_energies[:,0], bulk_energies[:,3], marker = "o", color = "green", label = r"$E_2(k_x)$")

    a1.legend(loc="upper left")

    plt.show()

                                         
def main():   
    Nx = 100
    Ny = 100                

    alpha = 0.3645
    beta = 0.686
    gamma = 0.512 
    m = 0.001
    
    plot_bulk_bands_QSH(Nx, Ny, alpha, beta, gamma, m)

    
    
    
if __name__ == '__main__':
    main()




    