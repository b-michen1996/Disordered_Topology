import csv
import numpy as np

#import matplotlib.colors as colors
from matplotlib import pyplot as plt



# enable latex
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams.update({
	"text.usetex": True,
	"font.family": "times"
})



def Fig_2(location_a, location_b):
	"""Generate plot for Fig_2."""
	
	E_vals_a = np.genfromtxt(location_a + "/E_vals.txt", dtype= float, delimiter=' ') 
	W_vals_a = np.genfromtxt(location_a + "/W_vals.txt", dtype= float, delimiter=' ') 
	data_a = np.genfromtxt(location_a + "/result.txt", dtype= float, delimiter=' ')  
	
	E_vals_b = np.genfromtxt(location_b + "/E_vals.txt", dtype= float, delimiter=' ') 
	W_vals_b = np.genfromtxt(location_b + "/W_vals.txt", dtype= float, delimiter=' ') 
	data_b = np.genfromtxt(location_b + "/result.txt", dtype= float, delimiter=' ')  

	W_array_a, E_array_a = np.meshgrid(W_vals_a, E_vals_a)
	W_array_b, E_array_b = np.meshgrid(W_vals_b, E_vals_b)
	
	#fig, axs = plt.subplots(1, 3, figsize =(12, 6), width_ratios = [0.05,1,1])
	fig, axs = plt.subplots(1, 2, figsize =(12, 6))
	fig.subplots_adjust(wspace=0.1)
	#a0 = axs[0]
	a1 = axs[0]
	a2 = axs[1]
	
	label_size = 25
	
	
	n_levels = 1000
	vmin = 0.
	vmax = 2    
		
	im_G_a = a1.imshow(data_a,
			   aspect='auto',
			   origin='lower',
			   extent=[np.min(W_vals_a), np.max(W_vals_a), np.min(E_vals_a), np.max(E_vals_a)],
			   vmin = 0,
			   vmax = 2,
			   cmap=plt.cm.magma.reversed()
			  )
	
	im_G_b = a2.imshow(data_b,
			   aspect='auto',
			   origin='lower',
			   extent=[np.min(W_vals_b), np.max(W_vals_b), np.min(E_vals_b), np.max(E_vals_b)],
			   vmin = 0,
			   vmax = 2,
			   cmap=plt.cm.magma.reversed()
			   
			  )
	
	cbar_title = r"$G[e^2/\hbar]$"
	

	cbar = plt.colorbar(im_G_a, shrink=0.6, ticks=[0, 1, 2], ax=axs.ravel().tolist(), location='left', pad = 0.051)
	cbar.ax.tick_params(labelsize=label_size) 
	cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25)
	
	
	a1.set_xlabel("W", fontsize=label_size)
	a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
		
	a1.set_xticks([0, 1, 2, 3, 4, 5, 6])
	a1.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
	a1.set_yticks([])
	
	a1.xaxis.set_label_coords(0.9, -0.025)
	
	a2.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
	a2.set_xticks([0, 1, 2, 3, 4, 5, 6])
	a2.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
	
	a2.set_xlabel("W", fontsize=label_size)
	a2.set_ylabel("E", fontsize=label_size, rotation='horizontal')
	
	a2.set_yticks([-3, -2, -1, 0., 1, 2, 3])
	a2.set_yticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
	a2.yaxis.set_ticks_position('right')
	
	a2.xaxis.set_label_coords(0.9, -0.025)
	a2.yaxis.set_label_coords(1.05, 0.86)

	color_wnum = "blue"	

	a1.text(0.5,0.94, r"a)", fontsize = label_size, transform=a1.transAxes)
	a1.text(0.8, 0.9, r"OBC", color = "gray", fontsize = 0.75 * label_size, transform=a1.transAxes, va = "center_baseline")
	
	a2.text(0.5, 0.94, r"b)", fontsize = label_size, transform=a2.transAxes)
	
	a2.text(0.15, 0.5, r"$\nu = 0$", color = color_wnum, fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
	a2.plot(0.12, 0.5, marker = "o", color = color_wnum, transform=a2.transAxes, markersize = 5)
	
	a2.text(0.43, 0.66, r"$\nu = 1$", color = color_wnum, fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
	a2.plot(0.4, 0.66, marker = "o", color = color_wnum, transform=a2.transAxes, markersize = 5)
	
	a2.text(0.43, 0.34, r"$\nu = 1$", color = color_wnum, fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
	a2.plot(0.4, 0.34, marker = "o", color = color_wnum, transform=a2.transAxes, markersize = 5)
	
	a2.text(0.8, 0.9, r"PBC", color = "gray", fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
	
	fig.savefig("Fig_2_publication.png",  bbox_inches='tight', dpi = 600)
	
	plt.show()
	
		
def Fig_2_vertical(location_a, location_b):
	"""Generate plot for Fig_2, vertical alignment."""
	
	E_vals_a = np.genfromtxt(location_a + "/E_vals.txt", dtype= float, delimiter=' ') 
	W_vals_a = np.genfromtxt(location_a + "/W_vals.txt", dtype= float, delimiter=' ') 
	data_a = np.genfromtxt(location_a + "/result.txt", dtype= float, delimiter=' ')  
	
	E_vals_b = np.genfromtxt(location_b + "/E_vals.txt", dtype= float, delimiter=' ') 
	W_vals_b = np.genfromtxt(location_b + "/W_vals.txt", dtype= float, delimiter=' ') 
	data_b = np.genfromtxt(location_b + "/result.txt", dtype= float, delimiter=' ')  

	W_array_a, E_array_a = np.meshgrid(W_vals_a, E_vals_a)
	W_array_b, E_array_b = np.meshgrid(W_vals_b, E_vals_b)
	
	#fig, axs = plt.subplots(1, 3, figsize =(12, 6), width_ratios = [0.05,1,1])
	fig, axs = plt.subplots(2, 1, figsize =(12, 24))
	fig.subplots_adjust(wspace=0.1)
	#a0 = axs[0]
	a1 = axs[0]
	a2 = axs[1]
	
	label_size = 25
	
	
	n_levels = 1000
	vmin = 0.
	vmax = 2    
		
	im_G_a = a1.imshow(data_a,
			   aspect='auto',
			   origin='lower',
			   extent=[np.min(W_vals_a), np.max(W_vals_a), np.min(E_vals_a), np.max(E_vals_a)],
			   vmin = 0,
			   vmax = 2,
			   cmap=plt.cm.magma.reversed()
			  )
	
	im_G_b = a2.imshow(data_b,
			   aspect='auto',
			   origin='lower',
			   extent=[np.min(W_vals_b), np.max(W_vals_b), np.min(E_vals_b), np.max(E_vals_b)],
			   vmin = 0,
			   vmax = 2,
			   cmap=plt.cm.magma.reversed()
			   
			  )
	
	cbar_title = r"$G[e^2/\hbar]$"
	

	cbar = plt.colorbar(im_G_a, shrink=0.6, ticks=[0, 1, 2], ax=axs.ravel().tolist(), location='left', pad = 0.051)
	cbar.ax.tick_params(labelsize=label_size) 
	cbar.ax.set_title(cbar_title, fontsize = label_size, pad = 25)
	
	
	a1.set_xlabel("W", fontsize=label_size)
	a1.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
		
	a1.set_xticks([0, 1, 2, 3, 4, 5, 6])
	a1.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
	a1.set_yticks([])
	
	a1.xaxis.set_label_coords(0.9, -0.025)
	
	a2.tick_params(direction='out', length=4, width=2,  labelsize = label_size, pad = 5)
	a2.set_xticks([0, 1, 2, 3, 4, 5, 6])
	a2.set_xticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
	
	a2.set_xlabel("W", fontsize=label_size)
	a2.set_ylabel("E", fontsize=label_size, rotation='horizontal')
	
	a2.set_yticks([-3, -2, -1, 0., 1, 2, 3])
	a2.set_yticklabels([r"$0$", "","", r"$3$","","", r"$6$"])
	a2.yaxis.set_ticks_position('right')
	
	a2.xaxis.set_label_coords(0.9, -0.025)
	a2.yaxis.set_label_coords(1.05, 0.86)
	
	a1.text(0.5,0.94, r"a)", fontsize = label_size, transform=a1.transAxes)
	a2.text(0.5, 0.94, r"b)", fontsize = label_size, transform=a2.transAxes)
	
	a2.text(0.04, 0.5, r"$\nu = 0$", color = "gray", fontsize = 0.75 * label_size, transform=a2.transAxes, va = "center_baseline")
	a2.plot(0.25, 0.5, marker = "o", color = "gray", transform=a2.transAxes, markersize = 5)
	
	fig.savefig("Fig_2_publication.png",  bbox_inches='tight')
	
	plt.show()
	

def main():                
	location_G_final_V3_a = "G_final_V3_results/G_final_V3_run_3"
	location_G_final_V3_b = "G_final_V3_results/G_final_V3_run_4"
		
	Fig_2(location_G_final_V3_a, location_G_final_V3_b)
	#Fig_2_vertical(location_G_final_V3_a, location_G_final_V3_b)
	

if __name__ == '__main__':
	main()




	
