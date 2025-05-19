#For the 2 species, 2D, nonlocal reaction-advection-diffusion system
#Class takes input of system parameters,
# and has a method outputting a value of the growth rate, lambda, for a given
# spatial mode, k. Also works with an array of ks, outputting an arry of lambdas
# the formula is based on linear stability analys

import numpy as np
import scipy.special as sps
import scipy.integrate as spi

class dispersion():
    
    def __init__(self, k, angle_matrix, U, V, D, mu_matrix, xi_matrix, kernel_name_matrix):
        
        #all matrices are 2x2 with indices 00=uu, 01=uv, 10=vu, 11=vv
        
        self.k = k
        self.U = U
        self.D = D
        self.V = V
        self.mu_matrix = mu_matrix
        self.xi_matrix = xi_matrix
        self.kernel_name_matrix = kernel_name_matrix
        
        self.angle_matrix = angle_matrix
        
        #F(k*xi) = (k/xi)*H_1(k,xi), where H_1 is the canonical Hankel transform
        self.H_matrix = np.full(self.kernel_name_matrix.shape, None)
        # normalisation of kernel
        self.omega_0_matrix = np.full(self.kernel_name_matrix.shape, None)
        
        
        for (ij, kernel_name) in np.ndenumerate(self.kernel_name_matrix):
            
            if (kernel_name == "tophat"):
                self.omega_0_matrix[ij] = 1/np.pi
                self.H_matrix[ij] = self.omega_0_matrix[ij]* (self.xi_matrix[ij]/self.k) * -( sps.j0(self.xi_matrix[ij]*self.k) - sps.itj0y0(self.xi_matrix[ij]*self.k)[0]/(self.xi_matrix[ij]*self.k) )
                
            elif (kernel_name == "exponential"):
                self.omega_0_matrix[ij] = 1/(2*np.pi)
                self.H_matrix[ij] = self.omega_0_matrix[ij]*self.k*((self.xi_matrix[ij])**3)/np.power( (1 + (self.xi_matrix[ij]*self.k)**2), 3/2 )
                
            elif (kernel_name == "o3"):
                self.omega_0_matrix[ij] = 1/(np.sqrt(2)*np.power(np.pi, 3/2))
                self.H_matrix[ij] = self.omega_0_matrix[ij]*self.k*((self.xi_matrix[ij])**3)*np.exp(-0.5 * (self.xi_matrix[ij]*self.k)**2)
            
            elif (kernel_name == "tophat_gradient_model"):
                self.omega_0_matrix[ij] = 1/np.pi
                self.H_matrix[ij] = self.omega_0_matrix[ij]*(self.xi_matrix[ij]**2)*sps.j1(self.xi_matrix[ij]*self.k)
                
            else:
                raise AttributeError("Please enter \"tophat\", \"exponential\", or \"o3\" for the kernel name.")
                
        
        
        #integral transform of kernels with approporiate prefactors for each interaction
        # See equation 39 in the paper
        Gamma_uu = np.cos(self.angle_matrix[0,0])*(2*np.pi*self.U*(1-self.U-self.V)*self.mu_matrix[0,0]/self.xi_matrix[0,0]**2)*self.H_matrix[0,0]
        Gamma_uv = np.cos(self.angle_matrix[0,1])*(2*np.pi*self.U*(1-self.U-self.V)*self.mu_matrix[0,1]/self.xi_matrix[0,1]**2)*self.H_matrix[0,1]
        Gamma_vu = np.cos(self.angle_matrix[1,0])*(2*np.pi*self.V*(1-self.U-self.V)*self.mu_matrix[1,0]/self.xi_matrix[1,0]**2)*self.H_matrix[1,0]
        Gamma_vv = np.cos(self.angle_matrix[1,1])*(2*np.pi*self.V*(1-self.U-self.V)*self.mu_matrix[1,1]/self.xi_matrix[1,1]**2)*self.H_matrix[1,1]

        #from equation lambda^2 + C(k)*lambda + D(k)=0
        #ADDED COSINE ANGLE TERMS TO GAMMA RATHER THAN C AND D
        self.C_k = (self.k**2)*(1+self.D) - self.k*(Gamma_vv + Gamma_uu)
        
        self.D_k = ( self.D*(self.k**4) - (self.k**3)*(Gamma_vv + self.D*Gamma_uu) 
             + (self.k**2)*(Gamma_uu*Gamma_vv - Gamma_vu*Gamma_uv)
             )
    
    #See equation 40 in the paper
    def lambda_plus(self):
        #TAKING ON THE POSITIVE ROOT OF THE QUADRATIC EQUATION FOR LAMBDA
        #answers may be complex
        return 0.5* ( -self.C_k + np.emath.sqrt(self.C_k**2 - 4*self.D_k) )

    
    def lambda_minus(self):
        #TAKING ON THE NEGATIVE ROOT OF THE QUADRATIC EQUATION FOR LAMBDA
        #answers may be complex
        return 0.5* ( -self.C_k - np.emath.sqrt(self.C_k**2 - 4*self.D_k) )
    



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib_style import matplotlib_style
    
    # Apply matplotlib style
    plt.rcParams.update(matplotlib_style(**{"text.usetex": True}))
    use_PGF = False
    save_figures = True
    file_format = "pdf"
    if use_PGF:
        matplotlib.use('pgf')
    
    
    # Function to parse parameter sets
    def parse_params(params):
        """
        Parse the parameter list into a structured dictionary.
        """
        return {
            "id": params[0],
            "U": params[1],
            "V": params[2],
            "D": params[3],
            "xi_matrix": np.array([[params[4], params[5]], [params[6], params[7]]]),
            "mu_matrix": np.array([[params[8], params[9]], [params[10], params[11]]]),
            "kernel_name_matrix": np.array([[params[12], params[13]],[params[14], params[15]]]),
            "angle_matrix": np.array([[params[16], params[17]],[params[18], params[19]]])
        }
    
    # Define parameter sets (example data)
    parallel_parameter_sets = [
        # [U, V, D, xi_matrix[0,0], xi_matrix[1,0], xi_matrix[0,1], xi_matrix[1,1], mu_matrix[0,0], mu_matrix[1,0], mu_matrix[0,1], mu_matrix[1,1], kernel_name_matrix[0,0], kernel_name_matrix[1,0], kernel_name_matrix[0,1], kernel_name_matrix[1,1], angle_matrix[0,0], angle_matrix[1,0], angle_matrix[0,1], angle_matrix[1,1]]
        
        ["Direct, Tophat, Chaser Self-Attracts, Runner Self-Attracts", 0.25, 0.25, 4, 1, 1, 1, 1, 50, 50, -50, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["Direct, Tophat, Chaser Self-Repels, Runner Self-Attracts", 0.25, 0.25, 1, 1, 1, 1, 1, -50, 50, -50, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["Direct, Tophat, Chaser Self-Attracts, Runner Self-Repels", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 50, -50, -50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["Direct, Exponential, Chaser Self-Attracts, Runner Self-Attracts", 0.25, 0.25, 4, 0.5, 0.5, 0.5, 0.5, 40, 40, -40, 40, "exponential", "exponential", "exponential", "exponential", 0, 0, 0, 0],
        ["Gradient, Tophat, Chaser Self-Attracts, Runner Self-Attracts", 0.25, 0.25, 4, 1, 1, 1, 1, 20, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
        ["Gradient, Tophat, Chaser Self-Repels, Runner Self-Attracts", 0.25, 0.25, 1, 1, 1, 1, 1, -20, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
        #["Tophat, Gradient, Self-Attract Self-Repel", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, -20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],

        #["c) Tophat, Direct, Self-Attract Self-Attract", 0.25, 0.25, 4, 1, 1, 1, 1, 50, 50, -50, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        #["d) Tophat, Direct, Self-Repel Self-Attract", 0.25, 0.25, 1, 1, 1, 1, 1, -50, 50, -50, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        #["e) Tophat, Direct, Self-Attract Self-Repel", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 50, -50, -50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        #["f) Exponential, Direct, Self-Attract Self-Attract", 0.25, 0.25, 4, 0.5, 0.5, 0.5, 0.5, 40, 40, -40, 40, "exponential", "exponential", "exponential", "exponential", 0, 0, 0, 0],
        #["g) Tophat, Gradient, Self-Attract Self-Attract", 0.25, 0.25, 4, 1, 1, 1, 1, 20, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
        #["h) Tophat, Gradient, Self-Repel Self-Attract", 0.25, 0.25, 1, 1, 1, 1, 1, -20, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
        #["Tophat, Gradient, Self-Attract Self-Repel", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, -20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
        #["exp_ra_para", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, -40, 40, -40, 40, "exponential", "exponential", "exponential", "exponential", 0, 0, 0, 0],
        #["exp_ar_para", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 40, 40, -40, -40, "exponential", "exponential", "exponential", "exponential", 0, 0, 0, 0],
    ]
    
    chiral_parameter_sets = [
        # [U, V, D, xi_matrix[0,0], xi_matrix[1,0], xi_matrix[0,1], xi_matrix[1,1], mu_matrix[0,0], mu_matrix[1,0], mu_matrix[0,1], mu_matrix[1,1], kernel_name_matrix[0,0], kernel_name_matrix[1,0], kernel_name_matrix[0,1], kernel_name_matrix[1,1], angle_matrix[0,0], angle_matrix[1,0], angle_matrix[0,1], angle_matrix[1,1]]
        
        ["c)", 0.25, 0.25, 4, 1, 1, 1, 1, 50, 50, -50, 50, "tophat", "tophat", "tophat", "tophat", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["d)", 0.25, 0.25, 1, 1, 1, 1, 1, -50, 50, -50, 50, "tophat", "tophat", "tophat", "tophat", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["e)", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 50, -50, -50, "tophat", "tophat", "tophat", "tophat", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["f)", 0.25, 0.25, 4, 0.5, 0.5, 0.5, 0.5, 40, 40, -40, 40, "exponential", "exponential", "exponential", "exponential", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["g)", 0.25, 0.25, 4, 1, 1, 1, 1, 20, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["h)", 0.25, 0.25, 1, 1, 1, 1, 1, -20, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, -0.3490658503988659, 1.2217304763960306, 0],
        #["Tophat, Gradient, Self-Attract Self-Repel", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, -20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, -0.3490658503988659, 1.2217304763960306, 0],
        #["exp_ra_para", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, -40, 40, -40, 40, "exponential", "exponential", "exponential", "exponential", 0, -0.3490658503988659, 1.2217304763960306, 0],
        #["exp_ar_para", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 40, 40, -40, -40, "exponential", "exponential", "exponential", "exponential", 0, -0.3490658503988659, 1.2217304763960306, 0],
    ]
    
    perp_parameter_sets = [
        # [U, V, D, xi_matrix[0,0], xi_matrix[1,0], xi_matrix[0,1], xi_matrix[1,1], mu_matrix[0,0], mu_matrix[1,0], mu_matrix[0,1], mu_matrix[1,1], kernel_name_matrix[0,0], kernel_name_matrix[1,0], kernel_name_matrix[0,1], kernel_name_matrix[1,1], angle_matrix[0,0], angle_matrix[1,0], angle_matrix[0,1], angle_matrix[1,1]]
        
        ["D) Tophat, Direct, Self-Attract Self-Attract", 0.25, 0.25, 4, 1, 1, 1, 1, 50, 50, -50, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 1.5707963267948966, 0],
        ["E) Tophat, Direct, Self-Repel Self-Attract", 0.25, 0.25, 1, 1, 1, 1, 1, -50, 50, -50, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 1.5707963267948966, 0],
        ["F) Tophat, Direct, Self-Attract Self-Repel", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 50, -50, -50, "tophat", "tophat", "tophat", "tophat", 0, 0, 1.5707963267948966, 0],
        ["G) Tophat, Gradient, Self-Attract Self-Attract", 0.25, 0.25, 4, 1, 1, 1, 1, 20, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 1.5707963267948966, 0],
        ["Tophat, Gradient, Self-Repel Self-Attract", 0.25, 0.25, 1, 1, 1, 1, 1, -20, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 1.5707963267948966, 0],
        ["Tophat, Gradient, Self-Attract Self-Repel", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, -20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 1.5707963267948966, 0],
        ["J) Exponential, Direct, Self-Attract Self-Attract", 0.25, 0.25, 4, 0.5, 0.5, 0.5, 0.5, 40, 40, -40, 40, "exponential", "exponential", "exponential", "exponential", 0, 0, 1.5707963267948966, 0],
        #["exp_ra_para", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, -40, 40, -40, 40, "exponential", "exponential", "exponential", "exponential", 0, 0, 1.5707963267948966, 0],
        #["exp_ar_para", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 40, 40, -40, -40, "exponential", "exponential", "exponential", "exponential", 0, 0, 1.5707963267948966, 0],
    ]
    
    other_parameter_sets = [
        ["Re($\lambda+$)", 0.48, 0.48, 1, 0.4, 0.4, 0.4, 0.4, 600, 400, -30, 300, "exponential", "exponential", "exponential", "exponential", 0, 0,0, 0]
        ]
    
    
    # parameter sets for complex lambda with parallel run and chase
    parallel_complex_sets = [
        # [U, V, D, xi_matrix[0,0], xi_matrix[1,0], xi_matrix[0,1], xi_matrix[1,1], mu_matrix[0,0], mu_matrix[1,0], mu_matrix[0,1], mu_matrix[1,1], kernel_name_matrix[0,0], kernel_name_matrix[1,0], kernel_name_matrix[0,1], kernel_name_matrix[1,1], angle_matrix[0,0], angle_matrix[1,0], angle_matrix[0,1], angle_matrix[1,1]]
        
        ["Direct, Tophat, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["Direct, Tophat, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 20, -20, 20, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["Gradient, Tophat, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 8, 8, -8, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
        ["Gradient, Tophat, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 8, -8, 8, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
        ["Direct, Exponential, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 20, 20, -20, 50, "exponential", "exponential", "exponential", "exponential", 0, 0, 0, 0],
        ["Direct, Exponential, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 50, 20, -20, 20, "exponential", "exponential", "exponential", "exponential", 0, 0, 0, 0],
        
        #["A) Tophat, Direct, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        #["B) Tophat, Gradient, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
        #["C) Exponential, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 20, 20, -20, 50, "exponential", "exponential", "exponential", "exponential", 0, 0, 0, 0],
        #["D) Tophat, Direct, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 20, -20, 20, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        #["E) Tophat, Gradient, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
        #["F) Exponential, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 50, 20, -20, 20, "exponential", "exponential", "exponential", "exponential", 0, 0, 0, 0],
        #["G)22 Tophat, Gradient, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 8, 8, -8, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, 0, 0, 0],
    ]
    
    # parameter sets for complex lambda with 70 degree running and -20 chasing
    chiral_complex_sets = [
        # [U, V, D, xi_matrix[0,0], xi_matrix[1,0], xi_matrix[0,1], xi_matrix[1,1], mu_matrix[0,0], mu_matrix[1,0], mu_matrix[0,1], mu_matrix[1,1], kernel_name_matrix[0,0], kernel_name_matrix[1,0], kernel_name_matrix[0,1], kernel_name_matrix[1,1], angle_matrix[0,0], angle_matrix[1,0], angle_matrix[0,1], angle_matrix[1,1]]
        
        ["D, T, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["D, T, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 20, -20, 20, "tophat", "tophat", "tophat", "tophat", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["G, T, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 8, 8, -8, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["G, T, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 8, -8, 8, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["D, E, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 20, 20, -20, 50, "exponential", "exponential", "exponential", "exponential", 0, -0.3490658503988659, 1.2217304763960306, 0],
        ["D, E, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 50, 20, -20, 20, "exponential", "exponential", "exponential", "exponential", 0, -0.3490658503988659, 1.2217304763960306, 0],
        
        #["A) Tophat, Direct, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, -0.3490658503988659, 1.2217304763960306, 0],
        #["B) Tophat, Gradient, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, -0.3490658503988659, 1.2217304763960306, 0],
        #["C) Exponential, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 20, 20, -20, 50, "exponential", "exponential", "exponential", "exponential", 0, -0.3490658503988659, 1.2217304763960306, 0],
        #["D) Tophat, Direct, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 20, -20, 20, "tophat", "tophat", "tophat", "tophat", 0, -0.3490658503988659, 1.2217304763960306, 0],
        #["E) Tophat, Gradient, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 50, 20, -20, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, -0.3490658503988659, 1.2217304763960306, 0],
        #["F) Exponential, $\\mu_{\\rho\\rho}>\\mu_{cc}>0$", 0.25, 0.25, 1, 0.5, 0.5, 0.5, 0.5, 50, 20, -20, 20, "exponential", "exponential", "exponential", "exponential", 0, -0.3490658503988659, 1.2217304763960306, 0],
        #["G)22 Tophat, Gradient, $\\mu_{cc}>\\mu_{\\rho\\rho}>0$", 0.25, 0.25, 1, 1, 1, 1, 1, 8, 8, -8, 20, "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", "tophat_gradient_model", 0, -0.3490658503988659, 1.2217304763960306, 0],   
        ]
    
    
    test_set = [
        # [U, V, D, xi_matrix[0,0], xi_matrix[1,0], xi_matrix[0,1], xi_matrix[1,1], mu_matrix[0,0], mu_matrix[1,0], mu_matrix[0,1], mu_matrix[1,1], kernel_name_matrix[0,0], kernel_name_matrix[1,0], kernel_name_matrix[0,1], kernel_name_matrix[1,1], angle_matrix[0,0], angle_matrix[1,0], angle_matrix[0,1], angle_matrix[1,1]]
        
        ["$\\alpha = 0^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["$\\alpha = 30^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 30*np.pi/180, 0],
        ["$\\alpha = 45^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 45*np.pi/180, 0],
        ["$\\alpha = 50^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 50*np.pi/180, 0],
        ["$\\alpha = 55^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 55*np.pi/180, 0],
        ["$\\alpha = 60^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 60*np.pi/180, 0],
        ]

    test_set_2 = [
        # [U, V, D, xi_matrix[0,0], xi_matrix[1,0], xi_matrix[0,1], xi_matrix[1,1], mu_matrix[0,0], mu_matrix[1,0], mu_matrix[0,1], mu_matrix[1,1], kernel_name_matrix[0,0], kernel_name_matrix[1,0], kernel_name_matrix[0,1], kernel_name_matrix[1,1], angle_matrix[0,0], angle_matrix[1,0], angle_matrix[0,1], angle_matrix[1,1]]
        
        ["$\\alpha = 0^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50, "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["$\\alpha = 30^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50*np.cos(30*np.pi/180), "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["$\\alpha = 45^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50*np.cos(45*np.pi/180), "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["$\\alpha = 50^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50*np.cos(50*np.pi/180), "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["$\\alpha = 55^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50*np.cos(55*np.pi/180), "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ["$\\alpha = 60^{\\circ}$", 0.25, 0.25, 1, 1, 1, 1, 1, 20, 20, -20, 50*np.cos(60*np.pi/180), "tophat", "tophat", "tophat", "tophat", 0, 0, 0, 0],
        ]
    
    
    # Parse parameter sets
    #parameter_sets = [parse_params(params) for params in chiral_parameter_sets]
    #parameter_sets = [parse_params(params) for params in other_parameter_sets]
    #parameter_sets = [parse_params(params) for params in parallel_parameter_sets]
    #parameter_sets = [parse_params(params) for params in parallel_complex_sets]
    #parameter_sets = [parse_params(params) for params in chiral_complex_sets]
    parameter_sets = [parse_params(params) for params in test_set_2]

    
    
    theory_k_points = np.linspace(0.01, 5, 1000)
    
    # Prepare figures
    fig_real, ax_real = plt.subplots(figsize=(10, 8))
    fig_imag, ax_imag = plt.subplots(figsize=(10, 8))
    fig_complex, ax_complex = plt.subplots(figsize=(10, 8))
    
    markers = ['o', 's', 'D', '^', 'v', '*', 'P', 'X']  # List of marker styles
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (4, 4, 1, 4))]  # Line styles for variety
    colors = list(plt.cm.tab10.colors) + ["#800080", "#008080"]  # Use a colormap for consistent coloring
    
    previous_count = 6
    
    # Plot each parameter set
    for i, params in enumerate(parameter_sets):
        label = params["id"]
        dispersion_calculator = dispersion(
            theory_k_points,
            params["angle_matrix"],
            params["U"],
            params["V"],
            params["D"],
            params["mu_matrix"],
            params["xi_matrix"],
            params["kernel_name_matrix"]
        )
        lambda_plus = dispersion_calculator.lambda_plus()
        lambda_minus = dispersion_calculator.lambda_minus()
        
        # Cycle through markers, linestyles, and colors
        marker = markers[(i + previous_count) % len(markers)]
        linestyle = linestyles[(i + previous_count) % len(linestyles)]
        color = colors[(i + previous_count) % len(colors)]
        marker_offset = (i)*10
        
        ax_real.plot(theory_k_points, lambda_plus.real, label=label, alpha=0.7,
                     marker=marker, markevery=(marker_offset, 50),
                     color=color, linestyle=linestyle)
        #ax_real.plot(theory_k_points, lambda_minus.real, linestyle='dotted', label=rf"Re($\lambda_-$) {label_base}")
        ax_imag.plot(theory_k_points, lambda_plus.imag, label=label, alpha=0.7,
                     marker=marker, markevery=(marker_offset, 50), color=color, linestyle=linestyle)
        #ax_imag.plot(theory_k_points, lambda_minus.imag, linestyle='dotted', label=rf"Im($\lambda_-$) {label_base}")
        
        ax_complex.plot(lambda_plus.real, lambda_plus.imag, label=label, alpha=0.7,
                     marker=marker, markevery=(marker_offset, 50), color=color, linestyle=linestyle)
    
    
    
    line1 = "Parallel Chase-and-Run"
    line2 = "$\\alpha_{c \\rho}=\\alpha_{\\rho c}=0$"
    #line1 = "Chiral Chase-and-Run"
    #line2 = "$\\alpha_{c \\rho}=-20^{\\circ},\,\\alpha_{\\rho c}=70^{\\circ}$"
    
    max_length = max(len(line1), len(line2))
    line1 = line1.center(max_length)
    line2 = line2.center(max_length)
    title = f"{line1}\n{line2}"
    # Customize plots
    for ax, fig, xlabel, ylabel in zip(
        [ax_real, ax_imag, ax_complex], 
        [fig_real, fig_imag, fig_complex],
        ["k", "k", r"Re($\lambda_+$)"],
        [r"Re($\lambda_+$)", r"Im($\lambda_+$)", r"Im($\lambda_+$)"], 
    ):
        ax.axhline(y=0, color='black', linestyle='solid')
        ax.set_xlabel(xlabel, fontsize=50)
        ax.set_ylabel(ylabel, fontsize=50)
        ax.legend(fontsize=15, ncol=2, loc="upper right") # fontisze=30 is better for final graph
        ax.set_xlim(0, 5)
        #ax.set_xlim(0, 3)
        #ax.set_ylim(-3, 3)
        ax.set_ylim(-8, 6)
        #ax.set_ylim(-10,10)
        #ax.set_title(title, fontsize=50, pad=30)
        #fig.text(0.5, 1, title, ha='center', fontsize=50)
    
    ax_real.set_title(title, fontsize=50, pad=30)
    #ax_real.set_ylim(-3,3)
    #ax_imag.set_ylim(-3, 3)
    ax_complex.axvline(x=0,color='black', linestyle='solid')
    
    
    # create legend separately
    fig_legend, ax_legend = plt.subplots(figsize=(50, 50))
    legend = fig_legend.legend(*ax_real.get_legend_handles_labels(), loc="center", ncol=2)
    fig_legend.canvas.draw()
    
    # Show or save figures
    if not use_PGF:
        plt.show()
        
    if save_figures:
        fig_real.savefig(f"hreal_part_lambda.{file_format}", bbox_inches="tight", pad_inches=0.3)
        fig_imag.savefig(f"imaginary_part_lambda.{file_format}", bbox_inches="tight", pad_inches=0.3)
        fig_legend.savefig(f"legend.{file_format}", bbox_inches="tight", pad_inches=0.3)

    