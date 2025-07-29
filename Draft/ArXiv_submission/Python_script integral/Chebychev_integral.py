import csv
import numpy as np
import scipy.integrate as integrate


from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib
import json


def integrand(x, m, E, eta):
    """Define integrand"""
    res = np.cos(m * np.arccos(x)) / ((E + 1j * eta -x) * (np.sqrt(1  -x**2)))
    
    return res

def integrand_2(u, m, E, eta):
    """Define integrand"""
    res = np.cos(m * u) / (E + 1j * eta - np.cos(u))
    
    res = E-np.cos(u)
    
    return res

def solve_integral(m, E, eta):
    """Calculate integral"""
    f_R = lambda x : integrand(x, m, E, eta).real
    f_imag = lambda x : integrand(x, m, E, eta).imag
    
    res_R, error_R = integrate.quad(f_R, -1, 1)
    res_Im, error_Im = integrate.quad(f_imag, -1, 1)
    
    return res_R, res_Im


def solve_integral_2(m, E, eta):
    """Calculate integral"""
    f_R = lambda x : integrand_2(x, m, E, eta).real
    f_imag = lambda x : integrand_2(x, m, E, eta).imag
    
    res_R, error_R = integrate.quad(f_R, 0, np.pi)
    res_Im, error_Im = integrate.quad(f_imag, 0, np.pi)
    
    return res_R, res_Im
    



			  
def main():
    m = 2
    
    E = 0.1
    eta = 0.001
    
    print(solve_integral(m, E, eta))
    print(solve_integral_2(m, E, eta))
    




if __name__ == '__main__':
	main()
