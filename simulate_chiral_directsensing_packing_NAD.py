import sys
import numpy as np

from chiral_directsensing_packing_NAD_class import chiral_direct_packing_NAD

import os

#name for folder
folder_tag = "results_chiral_direct_packing/"+"fig_3_c"

#Define parameters
U = 0.25
V = 0.25
D = 4
#note: for top-hat kernel xi should correspond to an integer number of meshpoints
xi_matrix = np.array([ [1,1],[1,1] ])
mu_matrix = np.array([ [50,50],[-50,50] ])
L_x = 8.
L_y = 8.
kernel_name_matrix = np.array([ ["tophat","tophat"], ["tophat","tophat"] ])

angle_matrix = np.array([ [0,-20*np.pi/180], [70*np.pi/180,0] ])

mesh_points_x = 100
mesh_points_y = int(mesh_points_x*(L_y/L_x)) #not used in simulation, just for plots
time_span = (0, 10)

RNG_seed = 15


#create folder to save the data to
os.makedirs(folder_tag, exist_ok=False)

#save text file specifying all the parameters
parameter_string = ("kernel(uu,uv,vu,vv)=%s,%s,%s,%s \n"
                    "angle(uu,uv,vu,vv)=%.10f,%.10f,%.10f,%.10f \n"
                    "U=%.10f \nV=%.10f \nD=%.10f \n"
                    "xi(uu,uv,vu,vv)=%.10f,%.10f,%.10f,%.10f \n"
                    "mu(uu,uv,vu,vv)=%.10f,%.10f,%.10f,%.10f \n"
                    "L_x=%.10f \nL_y=%.10f \nmesh_points_x=%d \n"
                    "mesh_points_y=%d \ninitial_time=%.10f \n"
                    "final_time=%.10f \nRNG_Seed=%d" 
                    %(kernel_name_matrix[0,0], kernel_name_matrix[0,1], kernel_name_matrix[1,0], kernel_name_matrix[1,1],
                      angle_matrix[0,0], angle_matrix[0,1], angle_matrix[1,0], angle_matrix[1,1],
                      U, V, D,
                      xi_matrix[0,0], xi_matrix[0,1], xi_matrix[1,0], xi_matrix[1,1], 
                      mu_matrix[0,0], mu_matrix[0,1], mu_matrix[1,0], mu_matrix[1,1],
                      L_x, L_y, mesh_points_x, mesh_points_y, time_span[0], time_span[1], RNG_seed)
                    )
with open("%s/parameters.txt" %folder_tag, "w") as text_file:
    text_file.write(parameter_string)
    
#Create instance of nonlocal_react_advect_diffuse_system
system = chiral_direct_packing_NAD(angle_matrix, U, V, D, xi_matrix, mu_matrix, 
                                      L_x, L_y, kernel_name_matrix, 
                                      mesh_points_x, time_span, seed=RNG_seed)

# Running simulation, checking and saving progress
print("Beginning simulation")
solution = system.simulate_with_progress("%s/data_u_v.csv" %folder_tag, "%s/data_t.csv" %folder_tag)

#note: we have no plotting here as ARC's scipy bundle doesn't have matplotlib
print("Everything finished")