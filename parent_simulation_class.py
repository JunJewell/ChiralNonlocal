#Parent class for 2 species, 2D, nonlocal reaction-advection-diffusion systems
#Class takes input of system and simulation parameters,
# and has `simulate' and `simulate_with_progress' methods outputting
# solution u(x,y,t) and v(x,y,t) and t, for all x,y in the discretised space, 
# and all t in the integration limits.
# Uses the method-of-lines to discretise integro-PDE into multiple integro-ODEs,
# integrates over time using scipy.solve_ivp, computing the integral term
# using a fast Fourier transform convolution method.
# See README.md for more details

import numpy as np
import scipy.integrate as integrator

from scipy.fftpack import fft2, ifft2

#Assuming:
# Rectangular domain [0,L_x]*[0,L_y]
# Periodic boundary conditions
# Default initial conditions of a Gaussian perturbation about the homogeneous steady state
# (but can specify own initial conditions)
class parent_nonlocal_react_advect_diffuse():
    
    def __init__(self, L_x, L_y, kernel_name_matrix, 
                 mesh_points_x, time_span, time_evaluations=None,
                 integrator_method="BDF",  rtol=1e-11, atol=1e-11,
                 initial_conditions=None, seed=15, initial_Gaussian_std=10**(-3)):
        
        self.L_x = L_x                       #domain is [0, L_x]*[0,L_y]
        self.L_y = L_y
        #2x2 of kernel names.  "tophat", "exponential", or "o3"
        self.kernel_name_matrix = kernel_name_matrix
        
        #simulation parameters
        self.mesh_points_x = mesh_points_x #number of points along x side of the grid
        self.h = self.L_x/self.mesh_points_x #use same spatial step in both x and y directions
        self.mesh_points_y = int(self.L_y/self.h)
        
        if (self.L_y/self.h%int(self.L_y/self.h) != 0):
            raise ValueError("ERROR: Values L_x, L_y, mesh_points_x do not allow for equal spatial step and integeter mesh_points_y")
        
            
        self.time_span = time_span      #[start_time, end_time]
        
        # which time values to store the solution.
        if (time_evaluations is None):
            if (self.time_span[1]>5 and self.time_span[0]<5):
                # Store more points for early time to see linear regime
                self.time_evaluations = np.concatenate([
                    np.linspace(self.time_span[0], 5, 100, endpoint=False),
                    np.linspace(5, self.time_span[1], 101)
                    ])
                # Include a very early time point for the future for if we ever
                # want to calculate the linear dispersion relation from simulation
                self.time_evaluations = np.insert(self.time_evaluations, 1, 1e-6)
            
            else:
                # for when we don't start near the linear regime
                self.time_evaluations = np.linspace(self.time_span[0], self.time_span[1], 101)
        else:
            # for when we specifically specify the points we want
            self.time_evaluations = time_evaluations
        
        #Using backwards differentiation formula, 'BDF', by default        
        self.integrator_method = integrator_method
        #Error tolerances for integration over time
        self.rtol = rtol
        self.atol = atol
        
        #seed for random intial conditions
        self.seed = seed
        self.initial_Gaussian_std = initial_Gaussian_std
        
        if (initial_conditions is None):
            #homogeneous steady state + random perturbation as ICs
            self.initial_conditions = np.zeros([2, self.mesh_points_x, self.mesh_points_y])
            #first add steady state
            self.initial_conditions[0,:,:] = self.homogeneous_steady_state()[0]
            self.initial_conditions[1,:,:] = self.homogeneous_steady_state()[1]
            #add perturbation: Gaussian zero mean, 10^-3 std
            self.initial_conditions += np.random.default_rng(self.seed).normal(loc=0, scale=self.initial_Gaussian_std, size = self.initial_conditions.shape)
        else:
            self.initial_conditions = initial_conditions
        
        if (self.initial_conditions[self.initial_conditions<0].size != 0):
            print("ERROR WARNING: Some initial conditions sampled with negative densities. Results will not be valid.")

    #Many of the following methods are virtual and are implemented in the specific
        # model classes, which are daughter classes of this class    

    def homogeneous_steady_state(self):
        
        raise NotImplementedError()
        
    
    #Compute integral term
    # ij in {uu, uv, vu, vv}
    def non_local_integral(self, Z, ij):
        
        raise NotImplementedError()
   
    def time_derivative(self, time, densities):
        
        raise NotImplementedError()
    
    #converts (2 by mesh_points_x by mesh_points_y) array into 1D for use in solve_ivp
    def grid_to_one_dimension(self, grid):
        return grid.reshape(grid.size,)
    
    #converts the 1D state vector back to (2 by mesh_points_x by mesh_points_y) array
    def one_dimension_to_grid(self, one_dimensional_vector):
        return one_dimensional_vector.reshape(2,self.mesh_points_x, self.mesh_points_y)
    
    
    #5-point stencil finite difference method with equal lengths
    def laplacian(self, Z):
        
        laplacian_grid = np.zeros_like(Z)
        
        # bottom + left + top + right -4central
        laplacian_grid[1:-1, 1:-1] = (Z[1:-1, 0:-2] + Z[0:-2, 1:-1] + Z[1:-1, 2:] + Z[2:, 1:-1] - 4*Z[1:-1, 1:-1]) / self.h**2
        
        #periodic BCs
        
        #vertical sides
        laplacian_grid[0, 1:-1] = (Z[1, 1:-1] + Z[-1, 1:-1] + Z[0, 0:-2] + Z[0, 2:] - (4*Z[0, 1:-1])) / self.h**2
        laplacian_grid[-1, 1:-1] = (Z[0,1:-1] + Z[-2,1:-1] + Z[-1:, 0:-2] + Z[-1, 2:] - (4*Z[-1, 1:-1])) / self.h**2
        
        #horizontal sides
        laplacian_grid[1:-1,0] = (Z[1:-1,1] + Z[1:-1,-1] + Z[0:-2,0] + Z[2:,0] - (4*Z[1:-1,0])) / self.h**2
        laplacian_grid[1:-1,-1] = (Z[1:-1,0] + Z[1:-1,-2] + Z[0:-2,-1] + Z[2:,-1] - (4*Z[1:-1,-1])) / self.h**2
        
        #corners
        laplacian_grid[0,0] = (Z[0,-1] + Z[-1,0] + Z[0,1] + Z[1,0] - 4*Z[0,0])/ self.h**2
        laplacian_grid[0,-1] = (Z[0,-2] + Z[-1,-1] + Z[0,0] + Z[1,-1] - 4*Z[0,-1])/ self.h**2
        laplacian_grid[-1,-1] = (Z[-1,-2] + Z[-2,-1] + Z[-1,0] + Z[0,-1] - 4*Z[-1,-1])/ self.h**2
        laplacian_grid[-1,0] = (Z[-1,-1] + Z[-2,0] + Z[-1,1] + Z[0,0] - 4*Z[-1,0])/ self.h**2
        
        return laplacian_grid
    
    
    #input a field Z, outputs the vector (dZ/dx, dZ/dy), where 'd' is partial derivative
    #central difference, so dZ/dx = Z_(i+1,y)-Z(i-1,y) /2dx
    def gradient(self, Z):
        
        dZ_dx = np.zeros_like(Z)
        dZ_dy = np.zeros_like(Z)
        
        dZ_dx[1:-1,:] = (Z[2:,:] - Z[0:-2,:])/(2*self.h)
        #periodic BC
        dZ_dx[0,:] = (Z[1,:] - Z[-1,:])/(2*self.h)
        dZ_dx[-1,:] = (Z[0,:] - Z[-2,:])/(2*self.h)
        
        dZ_dy[:, 1:-1] = (Z[:,2:] - Z[:, 0:-2])/(2*self.h)
        #periodic BC
        dZ_dy[:,0] = (Z[:,1] - Z[:,-1])/(2*self.h)
        dZ_dy[:,-1] = (Z[:,0] - Z[:,-2])/(2*self.h)
        
        return np.array([dZ_dx, dZ_dy])
    
    #input a vector of fields (Z_x, Z_y), output will be (dZ_x/dx + dZ_y/dy)
    #central difference so du_i/dx = u_(i+1)-u(i-1) /2dx
    def divergence(self, Z_vector):
        
        derivative_x = np.zeros_like(Z_vector[0])
        derivative_y = np.zeros_like(Z_vector[1])
        
        #use central finite difference
        
        derivative_x[1:-1,:] = (Z_vector[0][2:,:] - Z_vector[0][0:-2,:])/(2*self.h)
        #periodic BC
        derivative_x[0,:] = (Z_vector[0][1,:] - Z_vector[0][-1,:])/(2*self.h)
        derivative_x[-1,:] = (Z_vector[0][0,:] - Z_vector[0][-2,:])/(2*self.h)
        
        derivative_y[:, 1:-1] = (Z_vector[1][:,2:] - Z_vector[1][:, 0:-2])/(2*self.h)
        #periodic BC
        derivative_y[:,0] = (Z_vector[1][:,1] - Z_vector[1][:,-1])/(2*self.h)
        derivative_y[:,-1] = (Z_vector[1][:,0] - Z_vector[1][:,-2])/(2*self.h)
        
        return derivative_x + derivative_y
    
    #Multiply a vector field ( Z_x(x,y), Z_y(x,y) ) by a 2x2 tensor/matrix
        #e.g. a rotation matrix would correspond to a chiral model
    def tensor_multiply(self, tensor, Z_vector):
        
        return np.tensordot(tensor, Z_vector, axes=1)

    
    #Uses the convolution theorem and fast fourier transorm algorithm to find 
    # convolution(Z, kernel) = inverse_fourier( fourier(Z)*fourier(kernel) )
    def fft_convolve_2d(self, Z, kernel):
        
        convolution = ifft2(fft2(Z, shape=Z.shape) * fft2(kernel, shape=Z.shape)).real
        convolution = np.roll(convolution, (-((kernel.shape[0] - 1)//2), -((kernel.shape[1] - 1)//2)), axis=(0, 1))
        return convolution
    
    
    # top-hat interaction kernel
    def circular_weighted_kernel(self, xi_ij):
    
        #normalisation needed so that integral of kernel over all space is xi^2
        omega_0 = 1/np.pi
        
        radius_length = xi_ij
        number_of_points_per_radius = int(radius_length/self.h)

        coordinate = np.linspace(-radius_length, +radius_length, 2*number_of_points_per_radius+1)

        y,x = np.meshgrid(coordinate, coordinate)
        
        distance_from_centre = np.sqrt(y**2 + x**2)
        
        #set the centre value (which is zero) to an arbitrary value to avoid
        #division by zero. This will only be used in multiplications with zero
        #so the value is not important and will always end up as zero
        distance_from_centre[number_of_points_per_radius, number_of_points_per_radius] = -1
        
        kernel = np.ones_like(distance_from_centre)
        kernel[ np.where(distance_from_centre>radius_length) ] = 0

        #flip kernel as it gets flipped in convolution
        return omega_0*np.flip(kernel)
    
    
    # exponential interaction kernel
    def exponential_kernel(self, xi_ij):
        
        #normalisation needed so that integral of kernel over all space is xi^2
        omega_0 = 1/(2*np.pi)
        
        #We do not extend the kernel's support further than the length of the 
        #domain because we do not want points wrapping round and interacting
        #with more than once. Choose the smallest out of L_x or L_y to be the support
        #diameter in all directions so there is no directional bias. Some points
        #will be missed out, but on a sufficiently large domain compared to 
        #interaction length, xi, this should not make a difference.
        radius_length = np.min([self.L_x/2, self.L_y/2])
        
        #make the support one space-step smaller than the domain to ensure
        #there is no 'wrapping round' both for odd or even number of meshpoints
        radius_length -= self.h
        
        number_of_points_per_radius = int(radius_length/self.h)
        
        coordinate = np.linspace(-radius_length, +radius_length, 2*number_of_points_per_radius+1)
        
        y,x = np.meshgrid(coordinate, coordinate)
        distance_from_centre = np.sqrt(y**2 + x**2)
        
        #set the centre value (which is zero) to an arbitrary value to avoid
        #division by zero. This will only be used in multiplications with zero
        #so the value is not important and will always end up as zero
        distance_from_centre[number_of_points_per_radius, number_of_points_per_radius] = -1
        
        #the first term is from taking the component of "d hat" in either direction
        #"d hat" being the direction from the cell is deflected
        kernel = np.exp(-distance_from_centre/xi_ij)
        
        #make the support of the kernel circular, so we don't have any directional biases
        kernel[ np.where(distance_from_centre>radius_length) ] = 0
        
        #flip kernel as it gets flipped in convolution
        return omega_0*np.flip(kernel)
    
    
    # (r/xi)*e^(-(r/xi)^2) kernel, i.e. linear multiplied by gaussian
    # multiplied by s_hat
    #see comments on exponential kernel function above
    def o3_kernel(self, xi_ij, angle_ij):
        
        #normalisation needed so that integral of kernel over all space is xi^2
        omega_0 = 1/(np.sqrt(2)*np.power(np.pi, 3/2))

        
        radius_length = np.min([self.L_x/2, self.L_y/2])
        
        radius_length -= self.h
        
        number_of_points_per_radius = int(radius_length/self.h)
        
        coordinate = np.linspace(-radius_length, +radius_length, 2*number_of_points_per_radius+1)
        
        y,x = np.meshgrid(coordinate, coordinate)
        distance_from_centre = np.sqrt(y**2 + x**2)
        
        distance_from_centre[number_of_points_per_radius, number_of_points_per_radius] = -1
        
        kernel = (distance_from_centre/xi_ij) * (np.exp( -0.5*(distance_from_centre/xi_ij)**2 ))
        kernel[ np.where(distance_from_centre>radius_length) ] = 0
        
        return omega_0*np.flip(kernel)
    
    
    # Used for `direct sensing' models
     # input (N x N) array representing the interaction kernel
     # outputs (2 x N x N) array of (s_hat_x * kernel, s_hat_y * kernel),
     # where s_hat_x, s_hat_y are the x, y (respectively) components of the
     # radial unit vector, defined as s_hat = ( x/(x^2+y^2), y/(x^2+y^2) )

    def multiply_kernel_by_s_hat(self, kernel):
                
        coordinate = np.linspace(-1, +1, kernel.shape[0])
        
        y,x = np.meshgrid(coordinate, coordinate)
        distance_from_centre = np.sqrt(y**2 + x**2)
        
        # Temporarily make the middle value -1 so we don't divide by 0
         # This will only be used in multiplications with zero
         # so the value is not important and will always end up as zero
        np.put(distance_from_centre, distance_from_centre.size//2, -1)
        
        kernel_s_hat_x = (x/distance_from_centre)*np.flip(kernel)
        kernel_s_hat_y = (y/distance_from_centre)*np.flip(kernel)
        
        return np.array([np.flip(kernel_s_hat_x), np.flip(kernel_s_hat_y)])
    
    
    #Call this function to begin simulation and output the solution
    def simulate(self):
        
        solution = integrator.solve_ivp(fun=self.time_derivative,
                                        t_span=self.time_span, 
                                        y0=self.grid_to_one_dimension(self.initial_conditions),
                                        method=self.integrator_method,
                                        t_eval=self.time_evaluations,
                                        rtol=self.rtol, atol=self.atol)
        return solution


    #Same as above but saves data to textfile    
    def simulate_with_progress(self, y_file_name, t_file_name, number_of_checks=4):
        #storing every 'save_rate' datapoint
        #checks and saves progress 'number_of_checks' times, defaults to 4
        
        #split integration into quarters
        t_eval_sub_arrays = np.array_split(self.time_evaluations, number_of_checks) 
        
        solution = integrator.solve_ivp(fun=self.time_derivative,
                                        t_span=(self.time_span[0],  t_eval_sub_arrays[0][-1]), 
                                        y0=self.grid_to_one_dimension(self.initial_conditions),
                                        method=self.integrator_method,
                                        t_eval= t_eval_sub_arrays[0],
                                        rtol=self.rtol, atol=self.atol)
        
        if (solution.success == False):
            print("ERROR: Integration failed during the first quarter.")
            return solution
        

        
        np.savetxt(y_file_name, solution.y, delimiter=',')
        np.savetxt(t_file_name, solution.t, delimiter=',')
        
        next_solution = solution
        for i in range( 1, len(t_eval_sub_arrays) ):
            
            print("%d%% complete" %(100*i/len(t_eval_sub_arrays))  )
            
            next_solution = integrator.solve_ivp(fun=self.time_derivative,
                                            t_span=(t_eval_sub_arrays[i-1][-1],  t_eval_sub_arrays[i][-1]), 
                                            y0=self.grid_to_one_dimension(next_solution.y[:,-1]),
                                            method=self.integrator_method,
                                            t_eval= t_eval_sub_arrays[i],
                                            rtol=self.rtol, atol=self.atol)
            
            #concatenate the next solution with the previous solution
            #Note: leaving "sol" attribute unchanged as assuming dense_output=False
            solution.t=np.concatenate((solution.t, next_solution.t))
            solution.y=np.concatenate((solution.y, next_solution.y), 1)
            
            if(solution.t_events != None and next_solution.t_events!=None):
                solution.t_events=np.concatenate((solution.t_events, next_solution.t_events))
            if(solution.y_events != None and next_solution.y_events!=None):    
                solution.y_events=np.concatenate((solution.y_events, next_solution.y_events))
            
            solution.nfev=solution.nfev + next_solution.nfev
            solution.njev=solution.njev + next_solution.njev
            solution.nlu=solution.nlu + next_solution.nlu
            solution.status=next_solution.status
            solution.message=next_solution.message
            solution.success=next_solution.success
            
            
            if(next_solution.success == False):
                print("ERROR: Integration failed during the %d quarter." %(i+1))
                return solution
            
            #update data file
            np.savetxt(y_file_name, solution.y, delimiter=',')
            np.savetxt(t_file_name, solution.t, delimiter=',')
            
        print("100% complete")
        return solution
    
