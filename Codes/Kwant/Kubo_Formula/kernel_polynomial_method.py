#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Tutorial 2.8. Calculating spectral density with the Kernel Polynomial Method
# ============================================================================
#
# Physics background
# ------------------
#  - Chebyshev polynomials, random trace approximation, spectral densities.
#
# Kwant features highlighted
# --------------------------
#  - kpm module,kwant operators.

import scipy
from matplotlib import pyplot


# In[2]:


import matplotlib
import matplotlib.pyplot
#from matplotlib_inline.backend_inline import set_matplotlib_formats

matplotlib.rcParams['figure.figsize'] = matplotlib.pyplot.figaspect(1) * 2
#set_matplotlib_formats('svg')


# In[3]:


# necessary imports
import kwant
import numpy as np


# define the system
def make_syst(r=30, t=-1, a=1):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1)

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.shape(circle, (0, 0))] = 0.
    syst[lat.neighbors()] = t
    syst.eradicate_dangling()

    return syst


# In[4]:


## common plotting routines ##

# Plot several density of states curves on the same axes.
def plot_dos(labels_to_data):
    pyplot.figure()
    for label, (x, y) in labels_to_data:
        pyplot.plot(x, y.real, label=label, linewidth=2)
    pyplot.legend(loc=2, framealpha=0.5)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("DoS [a.u.]")
    pyplot.show()


# Plot fill density of states plus curves on the same axes.
def plot_dos_and_curves(dos, labels_to_data):
    pyplot.figure()
    pyplot.fill_between(dos[0], dos[1], label="DoS [a.u.]",
                     alpha=0.5, color='gray')
    for label, (x, y) in labels_to_data:
        pyplot.plot(x, y, label=label, linewidth=2)
    pyplot.legend(loc=2, framealpha=0.5)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("$σ [e^2/h]$")
    pyplot.show()


def site_size_conversion(densities):
    return 3 * np.abs(densities) / max(densities)


# Plot several local density of states maps in different subplots
def plot_ldos(syst, densities):
    fig, axes = pyplot.subplots(1, len(densities), figsize=(7*len(densities), 7))
    for ax, (title, rho) in zip(axes, densities):
        kwant.plotter.density(syst, rho.real, ax=ax)
        ax.set_title(title)
        ax.set(adjustable='box', aspect='equal')
    pyplot.show()


# In[5]:
"""

fsyst = make_syst().finalized()

spectrum = kwant.kpm.SpectralDensity(fsyst)


# In[6]:


energies, densities = spectrum()


# In[7]:


energy_subset = np.linspace(0, 2)
density_subset = spectrum(energy_subset)


# In[8]:


plot_dos([
    ('densities', (energies, densities)),
    ('density subset', (energy_subset, density_subset)),
])


# In[9]:


print('identity resolution:', spectrum.integrate())


# In[10]:


# Fermi energy 0.1 and temperature 0.2
fermi = lambda E: 1 / (np.exp((E - 0.1) / 0.2) + 1)

print('number of filled states:', spectrum.integrate(fermi))


# In[11]:


def make_syst_staggered(r=30, t=-1, a=1, m=0.1):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1)

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.a.shape(circle, (0, 0))] = m
    syst[lat.b.shape(circle, (0, 0))] = -m
    syst[lat.neighbors()] = t
    syst.eradicate_dangling()

    return syst


# In[12]:


fsyst_staggered = make_syst_staggered().finalized()
# find 'A' and 'B' sites in the unit cell at the center of the disk
center_tag = np.array([0, 0])
where = lambda s: s.tag == center_tag
# make local vectors
vector_factory = kwant.kpm.LocalVectors(fsyst_staggered, where)


# In[13]:


# 'num_vectors' can be unspecified when using 'LocalVectors'
local_dos = kwant.kpm.SpectralDensity(fsyst_staggered, num_vectors=None,
                                      vector_factory=vector_factory,
                                      mean=False)
energies, densities = local_dos()


# In[14]:


plot_dos([
    ('A sublattice', (energies, densities[:, 0])),
    ('B sublattice', (energies, densities[:, 1])),
])


# In[15]:


spectrum = kwant.kpm.SpectralDensity(fsyst)
original_dos = spectrum()


# In[16]:


spectrum.add_moments(energy_resolution=0.03)


# In[17]:


spectrum.add_moments(100)
spectrum.add_vectors(5)


# In[18]:


increased_moments_dos = spectrum()
plot_dos([
    ('density', original_dos),
    ('higher number of moments', increased_moments_dos),
])


# In[19]:


# identity matrix
matrix_op = scipy.sparse.eye(len(fsyst.sites))
matrix_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=matrix_op)


# In[20]:


# 'sum=True' means we sum over all the sites
kwant_op = kwant.operator.Density(fsyst, sum=True)
operator_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)


# In[21]:


# 'sum=False' is the default, but we include it explicitly here for clarity.
kwant_op = kwant.operator.Density(fsyst, sum=False)
local_dos = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)


# In[22]:


zero_energy_ldos = local_dos(energy=0)
finite_energy_ldos = local_dos(energy=1)


# In[23]:

plot_ldos(fsyst, [
    ('energy = 0', zero_energy_ldos),
    ('energy = 1', finite_energy_ldos)
])
"""

# In[24]:


def make_syst_topo(r=30, a=1, t=1, t2=0.5):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1, name=['a', 'b'])

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.shape(circle, (0, 0))] = 0.
    syst[lat.neighbors()] = t
    # add second neighbours hoppings
    syst[lat.a.neighbors()] = 1j * t2
    syst[lat.b.neighbors()] = -1j * t2
    syst.eradicate_dangling()

    return lat, syst.finalized()


# In[25]:


# construct the Haldane model
lat, fsyst_topo = make_syst_topo(r = 50)
# find 'A' and 'B' sites in the unit cell at the center of the disk
where = lambda s: np.linalg.norm(s.pos) < 1

# component 'xx'
s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)

cond_xx = kwant.kpm.conductivity(fsyst_topo, alpha='x', beta='x', mean=True,
                                 num_vectors=None, vector_factory=s_factory)
# component 'xy'
#s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
s_factory = kwant.kpm.RandomVectors(fsyst_topo, where)
#cond_xy = kwant.kpm.conductivity(fsyst_topo, alpha='x', beta='y', mean=True)#, num_vectors=None, vector_factory=s_factory)
cond_xy = kwant.kpm.conductivity(fsyst_topo, alpha='x', beta='y', vector_factory=s_factory, num_vectors = 10)


energies = cond_xx.energies
cond_array_xx = np.array([cond_xx(e, temperature=0.01) for e in energies])
cond_array_xy = np.array([cond_xy(e, temperature=0.01) for e in energies])

# area of the unit cell per site
area_per_site = np.abs(np.cross(*lat.prim_vecs)) / len(lat.sublattices)
print("Area per site: ", area_per_site)
print("np.cross(*lat.prim_vecs) = ", np.cross(*lat.prim_vecs))
print("len(lat.sublattices) = ", len(lat.sublattices))

cond_array_xx /= area_per_site
cond_array_xy /= np.pi * 50 **2 #area_per_site


# In[26]:


s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
spectrum = kwant.kpm.SpectralDensity(fsyst_topo, num_vectors=None,
                                     vector_factory=s_factory)

plot_dos_and_curves(
(spectrum.energies, spectrum.densities * 8),
[
    (r'Longitudinal conductivity $\sigma_{xx} / 4$',
     (energies, cond_array_xx.real / 4)),
    (r'Hall conductivity $\sigma_{xy}$',
     (energies, cond_array_xy.real))],
)


# In[27]:

"""
# construct a generator of vectors with n random elements -1 or +1.
n = fsyst.hamiltonian_submatrix(sparse=True).shape[0]
def binary_vectors():
    while True:
       yield np.rint(np.random.random_sample(n)) * 2 - 1

custom_factory = kwant.kpm.SpectralDensity(fsyst,
                                           vector_factory=binary_vectors())


# In[28]:


rho = kwant.operator.Density(fsyst, sum=True)

# sesquilinear map that does the same thing as `rho`
def rho_alt(bra, ket):
    return np.vdot(bra, ket)

rho_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho)
rho_alt_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho_alt)
"""
