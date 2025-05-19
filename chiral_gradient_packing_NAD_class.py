# Daughter class of 'parent_nonlocal_react_advect_diffuse'
# specificially implements a 2 species, 2D,nonlocal advection-diffusion model
    # with chirality, gradient sensing, and volume-filling (packing)
    
from parent_simulation_class import parent_nonlocal_react_advect_diffuse

import numpy as np

class chiral_gradient_packing_NAD(parent_nonlocal_react_advect_diffuse):
    def __init__(self, angle_matrix, U, V, D, xi_matrix, mu_matrix, L_x, L_y, kernel_name_matrix, 
                 mesh_points_x, time_span, time_evaluations=None,
                 integrator_method="BDF",  rtol=1e-11, atol=1e-11,
                 initial_conditions=None, seed=15, initial_Gaussian_std=10**(-3)):
        
        #Need to define homogeneous state first as used in init of parent class
        self.U = U
        self.V = V
        
        super().__init__(L_x, L_y, kernel_name_matrix, 
                 mesh_points_x, time_span, time_evaluations=time_evaluations,
                 integrator_method=integrator_method,  rtol=rtol, atol=atol,
                 initial_conditions=initial_conditions, seed=seed, 
                 initial_Gaussian_std=initial_Gaussian_std)
        
        #Equation parameters
        self.D = D                            #diffusion coefficient
        #2x2 matrix of nonlocal-signalling-ranges. xi_00=xi_uu, xi_01=xi_uv, xi_10=xi_vu, xi_11=xi_vv
        self.xi_matrix = xi_matrix 
        #2x2 matrix of nonlocal-interaction strengths. mu_00=mu_uu, mu_01=xi_uv, mu_10=xi_vu, mu_11=mu_vv
        self.mu_matrix = mu_matrix
        
        self.angle_matrix = angle_matrix
        
        self.rotation_matrix_matrix = np.full(self.angle_matrix.shape, None)
        for (ij, rotation_matrix) in np.ndenumerate(self.rotation_matrix_matrix):
            self.rotation_matrix_matrix[ij] = np.array([[np.cos(self.angle_matrix[ij]),-np.sin(self.angle_matrix[ij])],
                                 [np.sin(self.angle_matrix[ij]),np.cos(self.angle_matrix[ij])]])
        
        self.kernel_name_matrix = kernel_name_matrix
        
        #Interaction kernel, kernel is scalar because gradient model
        self.kernels_matrix = np.full(self.kernel_name_matrix.shape, None)
        
        
        for (ij, kernel_name) in np.ndenumerate(self.kernel_name_matrix):
            
            if (kernel_name == "tophat"):
                self.kernels_matrix[ij] = self.circular_weighted_kernel(self.xi_matrix[ij])
                
            elif (kernel_name == "exponential"):
                self.kernels_matrix[ij] = self.exponential_kernel(self.xi_matrix[ij])
                
            elif (kernel_name == "o3"):
                self.kernels_matrix[ij] = self.o3_kernel(self.xi_matrix[ij])
            
            else:
                raise AttributeError("Please enter \"tophat\", \"exponential\", or \"o3\" for the kernel name.")
                
        
    def homogeneous_steady_state(self):
        
        return np.array([self.U, self.V])
        
    #Compute Rotation(theta)*gradient(integral(interaction_kernel*Z))
     # where Z is a particular species density, for each interaction ij
      # ij in {uu, uv, vu, vv}
    def non_local_integral(self, Z, ij):
        
        return ( (self.h**2) *self.tensor_multiply(self.rotation_matrix_matrix[ij], self.gradient(self.fft_convolve_2d(Z, self.kernels_matrix[ij]))) )
    
   
    def time_derivative(self, time, densities):
    
        u, v = self.one_dimension_to_grid(densities)
        u_derivative = np.zeros_like(u)
        v_derivative = np.zeros_like(v)
        
        u_derivative = ( self.laplacian(u) 
                        - self.divergence( u*(1-u-v)*( ((self.mu_matrix[0,0]/self.xi_matrix[0,0])*self.non_local_integral(u, (0,0)))
                                                      +((self.mu_matrix[0,1]/self.xi_matrix[0,1])*self.non_local_integral(v, (0,1)))
                                                      )
                                          )
                        )
        
        v_derivative = ( self.D*self.laplacian(v)
                        - self.divergence( v*(1-u-v)*( ((self.mu_matrix[1,0]/self.xi_matrix[1,0])*self.non_local_integral(u, (1,0)))
                                                      +((self.mu_matrix[1,1]/self.xi_matrix[1,1])*self.non_local_integral(v, (1,1)))
                                                      )
                                          )
                        )
        
        return self.grid_to_one_dimension(np.array([u_derivative,v_derivative]))
                