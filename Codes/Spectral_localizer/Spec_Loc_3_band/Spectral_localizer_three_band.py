import numpy as np
import scipy.sparse
import scipy.linalg
import os

def coord_ind(jx, jy, gamma, Ny):
        "Return coordinates, gamma = 0,1,2 corresponds to a,b,c" 
        return 3 * (jx * Ny + jy) + gamma


def H_k(kx, ky, v):
        """Bloch Hamiltonian"""
        h_11 = 2 * (np.cos(kx) - np.cos(ky))
        h_12 = np.sqrt(2) * np.exp(-1j * np.pi / 4) * (np.exp(1j * kx)
            + np.exp(-1j * ky) +  1j * (np.exp(1j * (kx - ky)) + 1)) 
        
        res = np.array([[h_11, h_12, v],
                  [np.conj(h_12), -h_11, 0],
                  [v, 0, 0]])
        
        return res 
        

def bulk_energies(Nx, Ny, v):
        """Calculate bulk energies"""
        energies = np.zeros((Nx * Ny, 3))
        for jx in range(0, Nx):
                kx = jx * 2 * np.pi / Nx
                for jy in range(0, Ny):
                        ky = jy * 2 * np.pi / Ny

                        H_k_curr = H_k(kx, ky, v)

                        j = jx * Ny + jy 
                        energies[j, :] = np.linalg.eigvalsh(H_k_curr)
        energies = np.sort(energies.flatten())
        
        return energies


def Hamiltonian_three_band(Nx, Ny, v, PBC = 0.):
        """Create tight binding matrix for Hamiltonian."""
        # lists with non-zero array elements
        data = list()
        row = list()
        col = list()

        f = np.sqrt(2) * np.exp(-1j * np.pi / 4)

        # just loop over all elements, shouldn't be critical. Only set upper diagonal elements, 
        # fill in the rest by adding hermitian conjugate
        for jx in range(0, Nx):
                for jy in range(0, Ny):
                        j_a = coord_ind(jx, jy, 0, Ny)
                        j_b = coord_ind(jx, jy, 1, Ny)
                        j_c = coord_ind(jx, jy, 2, Ny)

                        # set on-site terms
                        row.append(j_a)
                        col.append(j_b)
                        data.append(1j * f)

                        row.append(j_a)
                        col.append(j_c)
                        data.append(v)

                        # set delta_x hopping terms
                        if (jx < Nx - 1):
                                j_a_p_delta_x = coord_ind(jx + 1, jy, 0, Ny)                         
                                j_b_p_delta_x = coord_ind(jx + 1, jy, 1, Ny)

                                row.append(j_a)
                                col.append(j_a_p_delta_x)
                                data.append(1.)

                                row.append(j_a)
                                col.append(j_b_p_delta_x)
                                data.append(f)

                                row.append(j_b)
                                col.append(j_b_p_delta_x)
                                data.append(-1.)

                                # set delta_x + delta_y hopping terms
                                if (jy > 0):                                                                    
                                        j_b_p_delta_x_my = coord_ind(jx + 1, jy - 1, 1, Ny)

                                        row.append(j_a)
                                        col.append(j_b_p_delta_x_my)
                                        data.append(1j * f)
                                else:
                                        j_b_p_delta_x_my = coord_ind(jx + 1, Ny - 1, 1, Ny)

                                        row.append(j_a)
                                        col.append(j_b_p_delta_x_my)
                                        data.append(PBC * 1j * f)

                        else:
                                j_a_p_delta_x = coord_ind(0, jy, 0, Ny)                         
                                j_b_p_delta_x = coord_ind(0, jy, 1, Ny)

                                row.append(j_a)
                                col.append(j_a_p_delta_x)
                                data.append(PBC * 1.)

                                row.append(j_a)
                                col.append(j_b_p_delta_x)
                                data.append(PBC * f)

                                row.append(j_b)
                                col.append(j_b_p_delta_x)
                                data.append(-PBC * 1.)

                                # set delta_x + delta_y hopping terms
                                if (jy > 0):                                                                    
                                        j_b_p_delta_x_my = coord_ind(0, jy - 1, 1, Ny)

                                        row.append(j_a)
                                        col.append(j_b_p_delta_x_my)
                                        data.append(PBC * 1j * f)
                                else:
                                        j_b_p_delta_x_my = coord_ind(0, Ny - 1, 1, Ny)

                                        row.append(j_a)
                                        col.append(j_b_p_delta_x_my)
                                        data.append(PBC * 1j * f)
                        
                        # set delta_y hopping terms
                        if (jy < Ny - 1):
                                j_a_p_delta_y = coord_ind(jx, jy + 1, 0, Ny)                    
                                j_b_p_delta_y = coord_ind(jx, jy + 1, 1, Ny)

                                row.append(j_a)
                                col.append(j_a_p_delta_y)
                                data.append(-1.)

                                row.append(j_b)
                                col.append(j_a_p_delta_y)
                                data.append(1j * f)

                                row.append(j_b)
                                col.append(j_b_p_delta_y)
                                data.append(1.)
                        else:
                                j_a_p_delta_y = coord_ind(jx, 0, 0, Ny)                    
                                j_b_p_delta_y = coord_ind(jx, 0, 1, Ny)

                                row.append(j_a)
                                col.append(j_a_p_delta_y)
                                data.append(-PBC *1.)

                                row.append(j_b)
                                col.append(j_a_p_delta_y)
                                data.append(1j * PBC *f)

                                row.append(j_b)
                                col.append(j_b_p_delta_y)
                                data.append(PBC * 1.)
                               
        H = scipy.sparse.csr_matrix((data, (row,col)), shape = (3 * Nx * Ny, 3 * Nx * Ny), dtype= complex)  

        H = H + H.conj().T   

        return H


def operator_X(Nx, Ny):
        diagonal_entries_X = np.tensordot(np.arange(Nx), np.ones(3 * Ny), axes = 0).flatten()

        #print("X entries: \n", diagonal_entries_X)

        X = scipy.sparse.diags(diagonals = diagonal_entries_X)

        return X


def operator_Y(Nx, Ny):
        diagonal_entries_Y = np.tensordot(np.ones(Nx), np.tensordot(np.arange(Ny), np.ones(3), axes = 0), axes = 0).flatten()

        #print("Y entries: \n", diagonal_entries_Y)

        Y = scipy.sparse.diags(diagonals = diagonal_entries_Y)

        return Y


def operator_V_rand(Nx, Ny, W):
        random_array = np.random.rand(3 * Nx * Ny)
        disorder_vals = W * (np.ones(3 * Nx * Ny) - 2 * random_array)

        V_rand = scipy.sparse.diags(diagonals = disorder_vals)

        return V_rand


def spec_loc(kappa, x, y, E, H, V_rand,  X, Y):
        """Build spectral localizer matrix."""
        size = H.shape[0]
        Id_matrix = scipy.sparse.identity(size)

        L = scipy.sparse.block_array([[H + V_rand - E * Id_matrix, kappa * (X - 1j * Y - (x - 1j * y) * Id_matrix)],
        [kappa * (X + 1j * Y - (x + 1j * y) * Id_matrix), - (H + V_rand) + E * Id_matrix]], format = "csr")

        return L


def export_localizer_gap_rs(kappa, Nx, Ny, v, W, E, x_min, x_max, y_min, y_max, n_pos, run_nr = 1):
        """Export spectral localizer gap in real space at fixed energy."""

        # create folder
        foldername = "spec_loc_rs_3_band/spec_loc_rs_3_band_run_" + str(run_nr)
        
        try:
                os.makedirs(foldername)
        except:
                pass 

        # create arrays X, Y for reference positions
        x_vals = np.linspace(x_min, x_max, n_pos)
        y_vals = np.linspace(y_min, y_max, n_pos)

        X_array, Y_array = np.meshgrid(x_vals, y_vals)

        np.savetxt(foldername + "/X_array.txt", X_array, delimiter=' ')   
        np.savetxt(foldername + "/Y_array.txt", Y_array, delimiter=' ')   

        # Build operator matrices
        X = operator_X(Nx, Ny)
        Y = operator_Y(Nx, Ny)
        H = Hamiltonian_three_band(Nx, Ny, v, 0)
        disorder = operator_V_rand(Nx, Ny, W)
                
        # export parameters
        f_parameter = open(foldername + "/parameters.txt", "w")
        
        f_parameter.write("kappa " + str(kappa) + str("\n"))
        f_parameter.write("v " + str(v) + str("\n"))
        f_parameter.write("W " + str(W) + str("\n"))

        f_parameter.write("Nx " + str(Nx) + str("\n"))
        f_parameter.write("Ny " + str(Ny) + str("\n"))

        f_parameter.write("E " + str(E) + str("\n"))
        f_parameter.write("x_min " + str(x_min) + str("\n"))
        f_parameter.write("x_max " + str(x_max) + str("\n"))
        f_parameter.write("y_min " + str(y_min) + str("\n"))
        f_parameter.write("y_max " + str(y_max) + str("\n"))

        f_parameter.write("n_pos " + str(n_pos) + str("\n"))
        f_parameter.write("run_nr " + str(run_nr) + str("\n"))
                
        f_parameter.close()

        # calculate and export localizer gaps at reference positions
        result = np.zeros((n_pos, n_pos))
        for jx in range(n_pos):
                for jy in range(n_pos):
                        x = X_array[jx, jy]
                        y = Y_array[jx, jy]

                        localizer = spec_loc(kappa, x, y, E, H, disorder, X, Y)

                        sigma, ev = scipy.sparse.linalg.eigsh(localizer, k = 3, which = "SM")     
                        gap = min(np.sort(np.abs(sigma)))

                        result[jx,jy] = gap

                        np.savetxt(foldername + "/result.txt", result, delimiter=' ')    
        
        return result


def export_index_Q(kappa, Nx, Ny, v, W, E_vals, run_nr = 1):
        """Export the topological index Q = sgn(L(x,y,E)) at the center of the system 
        for the energies specified in E_vals."""

        # create folder
        foldername = "spec_loc_top_index_3_band/spec_loc_top_index_3_band_run_" + str(run_nr)
        
        try:
                os.makedirs(foldername)
        except:
                pass         

        # Build operator matrices
        X = operator_X(Nx, Ny)
        Y = operator_Y(Nx, Ny)
        H = Hamiltonian_three_band(Nx, Ny, v, 0)
        disorder = operator_V_rand(Nx, Ny, W)
                
        # export parameters
        f_parameter = open(foldername + "/parameters.txt", "w")
        
        f_parameter.write("kappa " + str(kappa) + str("\n"))
        f_parameter.write("v " + str(v) + str("\n"))
        f_parameter.write("W " + str(W) + str("\n"))

        f_parameter.write("Nx " + str(Nx) + str("\n"))
        f_parameter.write("Ny " + str(Ny) + str("\n"))

        f_parameter.write("E_vals " + str(E_vals) + str("\n"))
        
        f_parameter.write("run_nr " + str(run_nr) + str("\n"))
                
        f_parameter.close()

        # calculate and export localizer gaps at reference positions
        n_E = len(E_vals)
        x = (Nx - 1) / 2
        y = (Ny - 1) / 2

        result = np.zeros((n_E, 2))

        for j_E in range(n_E):
                        E = E_vals[j_E]                        

                        localizer = spec_loc(kappa, x, y, E, H, disorder, X, Y)

                        eigenvalues = np.linalg.eigvalsh(localizer.todense())     
                                                
                        n_positive = sum(eigenvalues > 0)
                        n_negative = sum(eigenvalues < 0)

                        Q_E = (n_positive - n_negative)/2

                        result[j_E,:] = E, Q_E

                        np.savetxt(foldername + "/result.txt", result, delimiter=' ')    
        
        return result


def export_localizer_gap_energy(kappa, Nx, Ny, v, W, n_pos, E_min, E_max, n_E, run_nr = 1):
        """Export the minimal spectral localizer gap of n_pos^2 reference positions in the bulk as a function of energy"""
        # create folder
        foldername = "spec_loc_energy_3_band/spec_loc_energy_3_band_run_" + str(run_nr)
        
        try:
                os.makedirs(foldername)
        except:
                pass 

        # arrays for datapoint
        x_vals = np.linspace((Nx - 1)/3 , (Nx - 1)/4, n_pos)
        y_vals = np.linspace(Ny/2 - 1, Ny/2, n_pos)

        # create arrays X, Y for reference positions in the center Wigner-Seitz cell
        x_vals = np.linspace(Nx/2 - 1, Nx/2, n_pos, endpoint=False)
        y_vals = np.linspace(Ny/2 - 1, Ny/2, n_pos, endpoint=False)

        if n_pos == 1:
                x_vals = (Nx - 1)/2
                y_vals = (Ny - 1)/2      

        X_array, Y_array = np.meshgrid(x_vals, y_vals)

        # create array E_vals of energies
        E_vals = np.linspace(E_min, E_max, n_E)

        np.savetxt(foldername + "/E_vals.txt", E_vals, delimiter=' ')           

        # export bulk energies of the clean system
        E_bulk = bulk_energies(100, 100, v)

        np.savetxt(foldername + "/bulk_energies.txt", E_bulk, delimiter=' ')           

        # Build operator matrices
        X = operator_X(Nx, Ny)
        Y = operator_Y(Nx, Ny)
        H = Hamiltonian_three_band(Nx, Ny, v, 0)
        disorder = operator_V_rand(Nx, Ny, W)
                        
        # export parameters
        f_parameter = open(foldername + "/parameters.txt", "w")
        
        f_parameter.write("kappa " + str(kappa) + str("\n"))
        f_parameter.write("v " + str(v) + str("\n"))
        f_parameter.write("W " + str(W) + str("\n"))

        f_parameter.write("Nx " + str(Nx) + str("\n"))
        f_parameter.write("Ny " + str(Ny) + str("\n"))

        f_parameter.write("E_min " + str(E_min) + str("\n"))
        f_parameter.write("E_max " + str(E_max) + str("\n"))
        f_parameter.write("n_E " + str(n_E) + str("\n"))

        f_parameter.write("n_pos " + str(n_pos) + str("\n"))
        f_parameter.write("run_nr " + str(run_nr) + str("\n"))
                
        f_parameter.close()
        
        # calculate and export minimum of localizer gap in the bulk at the given energies
        result = np.zeros(n_E)
        for j_E in range(n_E):
                E = E_vals[j_E]
                gaps = np.zeros((n_pos, n_pos))

                for jx in range(n_pos):
                        for jy in range(n_pos):
                                x = X_array[jx, jy]
                                y = Y_array[jx, jy]
                                
                                localizer = spec_loc(kappa, x, y, E, H, disorder, X, Y)

                                sigma, ev = scipy.sparse.linalg.eigsh(localizer, k = 3, which = "SM")     
                                gaps[jx, jy] = min(np.sort(np.abs(sigma)))

                result[j_E] = np.min(gaps)

                np.savetxt(foldername + "/result.txt", result, delimiter=' ')    
                                                
        return result


def main():        
        Nx = 10
        Ny = 10
        v = -4.5
        W = 4

        E = 1.5
        kappa = 0.25

        x_min = 0
        x_max = Nx - 1

        y_min = 0
        y_max = Ny - 1

        E_min = -5.5
        E_max = 5.5

        n_pos = 10
        n_E = 50
        run_nr = 1

        E_vals = [-5, -2, 0, 2, 5]

        """
        x = 1.85
        y = 2.83
                
        #Hamiltonian_three_band(Nx, Ny, v)

        test_X = operator_X(Nx, Ny)
        test_Y = operator_Y(Nx, Ny)
        test_H = Hamiltonian_three_band(Nx, Ny, v, 1)
        test_disorder = operator_V_rand(Nx, Ny, W)


        #test_localizer = spec_loc(kappa, x, y, E, test_H, test_disorder, test_X, test_Y)

        #sigma, ev = scipy.sparse.linalg.eigsh(test_localizer, k = 3, which = "SM")     
        #print(sigma)
        
        energies = scipy.linalg.eigvalsh(test_H.todense())     
        #print(energies)
        E_bulk = bulk_energies(Nx, Ny, v )

        diff = np.linalg.norm(energies - E_bulk)

        print("Difference Bloch energies to TB: ", diff)
        """

        #localizer_gap = export_localizer_gap_rs(kappa, Nx, Ny, v, W, E, x_min, x_max, y_min, y_max, n_pos, run_nr)
        #localizer_gap_energy = export_localizer_gap_energy(kappa, Nx, Ny, v, W, n_pos, E_min, E_max, n_E, run_nr)
        export_index_Q(kappa, Nx, Ny, v, W, E_vals, run_nr)

        #print(localizer_gap)

        #print("Bulk min of localizer gap: ", min(localizer_gap.flatten()))



if __name__ == '__main__':
    main()