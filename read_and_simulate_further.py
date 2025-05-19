import numpy as np
import os
import gc
from chiral_directsensing_packing_NAD_class import chiral_direct_packing_NAD
from chiral_gradient_packing_NAD_class import chiral_gradient_packing_NAD


############   READ IN DATA    ######################
input_folder_tag = "My_directory/my_simulation"
output_folder_tag = input_folder_tag

parameter_file_name = r"%s/parameters.txt" %input_folder_tag
u_v_file_name = r"%s/data_u_v.csv" %input_folder_tag
t_file_name = r"%s/data_t.csv" %input_folder_tag


# what type of model
#model_type = "gradient"
model_type= "direct"

# time to continue the simulation to
t_final = 100

#Reading in parameters
print("Reading in parameters...")
with open(parameter_file_name) as param_file:
    parameters_string = param_file.read()

#All the values we need are between '=' and '\n'
#change all '\n' to '='
parameters_string = parameters_string.replace('\n','=')
#remove all white space
parameters_string = ''.join(parameters_string.split())
#take everything in between the '='
parameters_list = parameters_string.split(sep='=')
#only take every other element so we only take parameter values not names
parameters_list = parameters_list[1::2]

kernel_name_matrix = np.full((2,2), None)
kernel_name_matrix[0,0] = parameters_list[0].split(sep=',')[0]
kernel_name_matrix[0,1] = parameters_list[0].split(sep=',')[1]
kernel_name_matrix[1,0] = parameters_list[0].split(sep=',')[2]
kernel_name_matrix[1,1] = parameters_list[0].split(sep=',')[3]

angle_matrix = np.full((2,2), None)
angle_matrix[0,0] = float(parameters_list[1].split(sep=',')[0])
angle_matrix[0,1] = float(parameters_list[1].split(sep=',')[1])
angle_matrix[1,0] = float(parameters_list[1].split(sep=',')[2])
angle_matrix[1,1] = float(parameters_list[1].split(sep=',')[3])


U = float(parameters_list[2])
V = float(parameters_list[3])
D = float(parameters_list[4])

xi_matrix = np.full((2,2), None)
xi_matrix[0,0] = float(parameters_list[5].split(sep=',')[0])
xi_matrix[0,1] = float(parameters_list[5].split(sep=',')[1])
xi_matrix[1,0] = float(parameters_list[5].split(sep=',')[2])
xi_matrix[1,1] = float(parameters_list[5].split(sep=',')[3])

mu_matrix = np.full((2,2), None)
mu_matrix[0,0] = float(parameters_list[6].split(sep=',')[0])
mu_matrix[0,1] = float(parameters_list[6].split(sep=',')[1])
mu_matrix[1,0] = float(parameters_list[6].split(sep=',')[2])
mu_matrix[1,1] = float(parameters_list[6].split(sep=',')[3])

L_x = float(parameters_list[7])
L_y = float(parameters_list[8])
mesh_points_x = int(parameters_list[9])
mesh_points_y = int(parameters_list[10])
time_span = [float(parameters_list[11]), float(parameters_list[12])]
RNG_Seed = float(parameters_list[13])

#Reading in u and t data
print("Reading in u data...")
solution_y = np.genfromtxt(u_v_file_name, delimiter=',')
solution_t = np.genfromtxt(t_file_name, delimiter=',')
t_start = solution_t[-1]
initial_condition = solution_y[:,-1]
del solution_y
del solution_t
gc.collect()

if (t_start >= t_final):
    raise AttributeError("Please propose a final time that is larger than the current time simulated to.")

## Simulate Further ##

time_span = [t_start, t_final]

#create folder to save this batch of figures to
os.makedirs(output_folder_tag, exist_ok=True)

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
                      L_x, L_y, mesh_points_x, mesh_points_y, t_start, t_final, RNG_Seed)
                    )
with open("%s/parameters_continued.txt" %output_folder_tag, "w") as text_file:
    text_file.write(parameter_string)
    

if (model_type == "direct"):
    system = chiral_direct_packing_NAD(angle_matrix, U, V, D, xi_matrix, mu_matrix, 
                                          L_x, L_y, kernel_name_matrix, 
                                          mesh_points_x, time_span,
                                          initial_conditions=initial_condition, time_evaluations=np.linspace(time_span[0],time_span[1],int(time_span[1]-time_span[0])+1))

if (model_type == "gradient"):
    system = chiral_gradient_packing_NAD(angle_matrix, U, V, D, xi_matrix, mu_matrix, 
                                          L_x, L_y, kernel_name_matrix, 
                                          mesh_points_x, time_span,
                                          initial_conditions=initial_condition, time_evaluations=np.linspace(time_span[0],time_span[1],int(time_span[1]-time_span[0])+1))

# Running simulation, checking and saving progress
print("Beginning simulation")
solution = system.simulate_with_progress("%s/data_u_v_continued.csv" %output_folder_tag, "%s/data_t_continued.csv" %output_folder_tag)
