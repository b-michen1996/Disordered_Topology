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


def H_k(kx, ky, v):
    """Bloch Hamiltonian"""
    h_11 = 2 * (np.cos(kx) - np.cos(ky))
    
    h_12 = np.sqrt(2) * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx)
            + np.exp(1j * ky) +  1j * np.exp(1j * (kx + ky)) + 1) 
    
    h_12 = np.sqrt(2) * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx)
            + np.exp(-1j * ky) +  1j * (np.exp(1j * (kx - ky)) + 1)) 
    
    res = np.array([[h_11, h_12, v],
                [np.conj(h_12), -h_11, 0],
                [v, 0, 0]])
    
    return res 


def plot_bulk_bands(Nx, Ny, v):
    """Plot bulk bands."""

    bulk_energies = np.zeros((Nx * Ny, 5))
    for jx in range(0, Nx):
                kx = -np.pi + jx * 2 * np.pi / Nx
                for jy in range(0, Ny):
                        ky = -np.pi + jy * 2 * np.pi / Ny

                        H_k_curr = H_k(kx, ky, v)                        

                        j = jx * Ny +jy
                        
                        bulk_energies[j, :2] = kx, ky
                        bulk_energies[j, 2:] = np.linalg.eigvalsh(H_k_curr)   

    title = rf"Bulk bands for $v = {v}$, $N_x = {Nx}$, $N_y = {Ny}$"

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
    a1.scatter(bulk_energies[:,0], bulk_energies[:,4], marker = "o", color = "blue", label = r"$E_3(k_x)$")

    a1.legend(loc="upper left")

    plt.show()

                                         
def main():   
    Nx = 100
    Ny = 100                

    v = 3.5
    
    plot_bulk_bands(Nx, Ny, v)

    
    
    
if __name__ == '__main__':
    main()




    