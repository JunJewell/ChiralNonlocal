import numpy as np
import matplotlib.pyplot as plt

from plot_class import plotter
from dispersion_relation import dispersion

############   READ IN DATA    ######################
input_folder_tag = "My_directory/my_simulation"

output_folder_tag = input_folder_tag

parameter_file_name = r"%s/parameters.txt" %input_folder_tag
u_v_file_name = r"%s/data_u_v.csv" %input_folder_tag
t_file_name = r"%s/data_t.csv" %input_folder_tag

#added to the start of the graph file name, can leave blank as r""
output_file_tag = r""

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
print("Reading in u and t data...")
solution_y = np.genfromtxt(u_v_file_name, delimiter=',')
solution_t = np.genfromtxt(t_file_name, delimiter=',')




##########   PLOT FIGURES   ####################
print("Beginning plotting...")
dispersion_calculator = dispersion(np.linspace(0.01, 30, 300),
                                   angle_matrix, U, V, D, mu_matrix,
                                   xi_matrix, kernel_name_matrix)

plot = plotter( L_x=L_x, L_y=L_y, mesh_points_x=mesh_points_x,
                       mesh_points_y=mesh_points_y, solution_y=solution_y,
                       solution_t=solution_t, folder_tag=output_folder_tag, 
                       file_tag=output_file_tag,
                       use_PGF=False, dispersion_relation=dispersion_calculator)

plot.animation_three_panels(sample_rate=1, num_ticks=3)

print("heatmap both species...")
plot.heatmap_both_species(normalisation_method=0)


print("dispersion relation...")
plot.dispersion_plot()
plot.dispersion_plot(imaginary=True)

print("heatmaps...")
#plotting heatmap of u at final time
plot.heatmap(u_not_v=True)
#plotting heatmap of v at final time
plot.heatmap(u_not_v=False)

print("spaghetti evolution...")
plot.evolution_with_time(u_not_v=True)
plot.evolution_with_time(u_not_v=False)

print("spaghetti derivative...")
plot.time_derivative(u_not_v=True)
plot.time_derivative(u_not_v=False)

plt.close('all')

print("animation both...")
plot.animation_both_species(sample_rate=1, normalisation_method=1)
plt.close('all')

max_t = np.max(plot.u_v_over_time[0], axis=(0, 1))  # Shape (202,)
min_t = np.min(plot.u_v_over_time[0], axis=(0, 1))  # Shape (202,)
diff_t = max_t - min_t
print(np.max(diff_t))
print(diff_t[-1])

max_t_v = np.max(plot.u_v_over_time[1], axis=(0, 1))  # Shape (202,)
min_t_v = np.min(plot.u_v_over_time[1], axis=(0, 1))  # Shape (202,)
diff_t_v = max_t_v - min_t_v


print("Finished.")
