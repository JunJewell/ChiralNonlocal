#For the 2 species, 2D, nonlocal reaction-advection-diffusion system
#Class takes input of solution data and system parameters
# and has methods that plot and save graphs or animations of density heatmaps,

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import zoom

import matplotlib
from matplotlib_style import matplotlib_style

    

class plotter():
    
    def __init__(self, L_x, L_y, mesh_points_x, mesh_points_y, solution_y, solution_t, 
                  folder_tag ,file_tag, use_PGF=True, save=True, dispersion_relation=None):
        
        self.mesh_points_x = mesh_points_x
        self.mesh_points_y = mesh_points_y
        

        self.u_v_over_time = solution_y.reshape(2, self.mesh_points_x, self.mesh_points_y, solution_t.size)
        if (self.u_v_over_time[0,...] + self.u_v_over_time[1,...] > 1).any():
            print("WARNING: Some points exceed density carry capacity of 1.")
        if (self.u_v_over_time < 0).any():
            print("WARNING: Some densities are negative.")
            
        self.solution_t = solution_t
        
        self.x_points = np.linspace(0, L_x, self.mesh_points_x)
        self.y_points = np.linspace(0, L_y, self.mesh_points_y)
        
        
        plt.rcParams.update(matplotlib_style(**{"text.usetex":True})) 
        self.use_PGF = use_PGF
        if (self.use_PGF):
            matplotlib.use('pgf')
        
        

        self.save = save
        #string to add to the start of the saved graph file, to distinguish it
        self.file_tag = file_tag
        #name of folder to save to
        self.folder_tag = folder_tag
                
        #dispersion relation object which can be used for plotting, if supplied
        self.dispersion_relation = dispersion_relation
        
        self.colourmap = self.create_colourmap()

    
    def create_colourmap(self):
        N=200 
        hue_start=60
        hue_end=180

        p = 100
        # 8 is good
        #num_bins = 8
        num_bins = 8 #debug
        saturation_bin_vals = np.linspace(0, 1, num_bins+2)[1:-1]
        hue_bin_vals = np.linspace(hue_start, hue_end, num_bins+2)[1:-1]/360

        # Generate coordinate grid with origin at (0,0)
        y, x = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

        # Compute angle (theta) and radius (r) from (0,0)
        theta = np.arctan2(y, x)  # Angle in radians
        r = np.sqrt(x**2 + y**2)  # Radial distance
        r = np.clip(r, 0, 1)  # Ensure it stays in [0,1] for saturation

        # Map theta to Hue range correctly
        theta = np.degrees(theta)%360 # Convert to degrees
        theta_mapped = hue_start + (hue_end - hue_start) * (theta / 90)
        theta_mapped = np.clip(theta_mapped, hue_start, hue_end) / 360  # Normalize to [0,1] range

        h = theta_mapped
        s = r
        v = np.ones_like(r)

        s_nonlinear = saturation_bin_vals.size
        for i in saturation_bin_vals:
            s_nonlinear += np.tanh(p*(s-i))
        s_nonlinear /= 2*saturation_bin_vals.size

        h_nonlinear = hue_bin_vals.size
        for i in hue_bin_vals:
            h_nonlinear += np.tanh(p*(h-i))
        h_nonlinear /= 2*hue_bin_vals.size
        h_nonlinear = (hue_start + h_nonlinear*(hue_end-hue_start))/360


        #hsv_stack = np.stack((h_nonlinear, s_nonlinear, v), axis=-1)
        #hsv_stack = np.stack((h_nonlinear, v, s_nonlinear), axis=-1)  # (0,0) black
        hsv_stack = np.stack((h_nonlinear, s_nonlinear, 1-s_nonlinear/3), axis=-1) # (0,0) white

        return matplotlib.colors.hsv_to_rgb(hsv_stack)
    
    def densities_to_colours(self, time_index, normalisation_method=0, print_output=True):
    
        # Takes 2xNxN array of densities for species u and v in NxN 2D space
        # Takes 2D RGBA colourmap defined by a MxMx4 array, where M is resolution
        # Outputs NxNx4 array corresponding to an RGBA colour for 
        # each point in NxN space
        
        h, w, _ = self.colourmap.shape
        
        u = self.u_v_over_time[0,:,:,time_index].T.reshape(-1)
        v = self.u_v_over_time[1,:,:,time_index].T.reshape(-1)
        
        # normalisation method 0, 1, or 2, or length 4 list of (u_min, u_max, v_min, v_max)
        # 0 = no normalisation, take raw u and v values, but clipped to [0,1]
        # 1 = each species normalised between max and min of that species
        # 2 = both species normalised between max and min of their sums
        
        if ( type(normalisation_method) == np.ndarray or type(normalisation_method) == list):
            normalisation_method = np.array(normalisation_method)
            if (normalisation_method.shape == (4,)):
                u_min, u_max = normalisation_method[0], normalisation_method[1]
                v_min, v_max = normalisation_method[2], normalisation_method[3]
                u = (u - u_min) / (u_max - u_min)
                v = (v - v_min) / (v_max - v_min)
                
            else:
                raise ValueError("ERROR: Please only input 0, 1, 2," 
                                 "or a list of length 4 for normalisation_method")
                
        elif (normalisation_method == 0):
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)        
            
        elif (normalisation_method == 1):
            u_min, u_max = u.min(), u.max()
            v_min, v_max = v.min(), v.max()
            u = (u - u_min) / (u_max - u_min +1e-12)
            v = (v - v_min) / (v_max - v_min +1e-12)
            if print_output:
                print("u_min = %.3f, u_max = %.3f, v_min = %.3f, v_max = %.3f" 
                      %(u_min, u_max, v_min, v_max))
        
        elif (normalisation_method == 2):
            u_v_min = np.min([u,v])
            u_v_max = (u+v).max()
            u = (u - u_v_min) / (u_v_max - u_v_min +1e-12)
            v = (v - u_v_min) / (u_v_max - u_v_min +1e-12)
            
        else:
            raise ValueError("ERROR: Please only input 0, 1, 2," 
                             "or a list of length 4 for normalisation_method")
            
        
        # Scale to pixel indices
        u_idx = (u * (w - 1)).astype(int)
        v_idx = (v * (h - 1)).astype(int)
        
        # Fetch colors using advanced indexing
        colours = self.colourmap[u_idx, v_idx]
        
        # i.e. 3 for rgb, 4 for rgba
        colour_dimensions = colours.size//(self.mesh_points_x*self.mesh_points_y)
        
        return colours.reshape(self.mesh_points_x, self.mesh_points_y, colour_dimensions)    
 
    #Create a heatmap of the density at the a specific time for either u(x,t) or v(x,t)
    #Default is at the final time. u_not_v=True to plot u, and u_not_v=False to plot v.
    def heatmap(self, u_not_v = True, time_index=-1, raster=True, num_ticks=3):
        
        if (u_not_v):
            species_index = 0
            species_name = "u"
        else:
            species_index = 1
            species_name = "v"
        
        fig, ax = plt.subplots()
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        pcm = ax.pcolormesh(self.x_points, self.y_points,
                            self.u_v_over_time[species_index,:,:,time_index].T,
                            cmap="magma", rasterized=raster)
        cb = fig.colorbar(pcm, cax=cax)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MultipleLocator(self.x_points.max()/(num_ticks-1)))
        ax.yaxis.set_major_locator(MultipleLocator(self.y_points.max()/(num_ticks-1)))
        
        if (self.save):
            fig.savefig("%s/%s_heatmap_%s_t=%.2f.pdf" %(self.folder_tag, self.file_tag,
                                                        species_name,
                                                  self.solution_t[time_index]))
            
    

    #plots a graph with a curve for u or v at each (x,y) against time
    def evolution_with_time(self, u_not_v = True, plot_every=1):
        
        if (u_not_v):
            species_index = 0
            species_name = "u"
        else:
            species_index = 1
            species_name = "v"  
        
        fig, ax = plt.subplots()
        counter=0
        for x in self.u_v_over_time[species_index]:
            for y in x:
                if counter%plot_every==0:
                    ax.plot(self.solution_t, y)
                counter+=1
                
        ax.set_xlabel("$t$")
        ax.set_ylabel("Each $%s(x,y)$" %species_name)
        ax.axhline(y=0, c='black', linestyle='dashed')
        
        if (self.save):
            fig.savefig("%s/%s_%s_evolution.png" %(self.folder_tag, self.file_tag, species_name))
        
    
    #plots a graph with a curve for du/dt at each (x,y) against time
    #Note this is only an estimate using finite difference of u solution
    def time_derivative(self, u_not_v = True):
        
        if (u_not_v):
            species_index = 0
            species_name = "u"
        else:
            species_index = 1
            species_name = "v"  
        
        derivative = (self.u_v_over_time[species_index, :,:,1:]-self.u_v_over_time[species_index, :,:,0:-1]) / (self.solution_t[1:]-self.solution_t[0:-1])
        
        fig, ax = plt.subplots()
        for x in derivative:
            for y in x:
                ax.plot(self.solution_t[1:], y)
        ax.set_xlabel("$t$")
        ax.set_ylabel("Each $\partial %s(x,y)/ \partial t$" %species_name)
        ax.axhline(y=0, c='black', linestyle='dashed')
        
        if (self.save):
            fig.savefig("%s/%s_%s_derivative.png" %(self.folder_tag, self.file_tag, species_name))
        

    

    def dispersion_plot(self, imaginary=False):
        
        if self.dispersion_relation is None:
            print("No dispersion relation supplied so cannot plot.")
            return -1
        
        fig_disp, ax_disp = plt.subplots(1,1, figsize=(10,8))
        
        fontsize = 30
        
        if (imaginary == False):
            ax_disp.plot(self.dispersion_relation.k, self.dispersion_relation.lambda_plus().real, label="Re($\lambda$+)")
            ax_disp.plot(self.dispersion_relation.k,self.dispersion_relation.lambda_minus().real, label="Re($\lambda$-)", linestyle='dotted')
            ax_disp.set_ylabel("Re($\lambda$+)", fontsize=fontsize)
        if (imaginary == True):
            ax_disp.plot(self.dispersion_relation.k, self.dispersion_relation.lambda_plus().imag, label="Im($\lambda$+)")
            ax_disp.plot(self.dispersion_relation.k,self.dispersion_relation.lambda_minus().imag, label="Im($\lambda$-)", linestyle='dotted')      
            ax_disp.set_ylabel("Im($\lambda$+)", fontsize=fontsize)
        
        ax_disp.axhline(y=0, c='black', linestyle='solid')
        ax_disp.set_xlabel("k")
        ax_disp.legend(loc='upper left', fontsize=fontsize)
        
        #setting useful y and x limits using heuristics
        max_k_index = np.where(np.sign(self.dispersion_relation.lambda_plus().real[:-1]) - np.sign(self.dispersion_relation.lambda_plus().real[1:]) == 2)[0] + 1
        if(max_k_index.size == 0):
            max_k_index=20
        else:
            max_k_index = max_k_index[0] 
        ax_disp.set_xlim(-0.1, 1.5*self.dispersion_relation.k[max_k_index])
        
        if (imaginary == False):
            max_y = np.max(self.dispersion_relation.lambda_plus().real)
        if (imaginary == True):
            max_y = np.max(self.dispersion_relation.lambda_plus().imag)
            if (max_y==0):
                max_y=1
            
        max_y = abs(max_y) + 0.25*abs(max_y)
        ax_disp.set_ylim(-max_y, max_y)
        
        #Plotting C(k) and D(k)
        
        if (imaginary == False):
        
            ax2 = ax_disp.twinx()
            ax2.set_ylabel("$\mathcal{C}(k)$", fontsize=fontsize)
            ax2.plot(self.dispersion_relation.k, self.dispersion_relation.C_k, label="$\mathcal{C}(k)$", color='green',linestyle='dashed')
            
            max_y = np.max(abs(self.dispersion_relation.C_k[:max_k_index]))
            ax2.set_ylim(-1.1*max_y, 1.1*max_y)
            
            ax2.legend(loc="upper right", fontsize=fontsize)
            
            ax2.ticklabel_format(axis='y', style='sci', scilimits=[-3,3])
            
            ax3 = ax_disp.twinx()
            ax3.set_ylabel("$\mathcal{D}(k)$", fontsize=fontsize)
            ax3.plot(self.dispersion_relation.k, self.dispersion_relation.D_k, label="$\mathcal{D}(k)$", color='red',linestyle='dashed')
            
            max_y = np.max(abs(self.dispersion_relation.D_k[:max_k_index]))
            ax3.set_ylim(-1.1*max_y, 1.1*max_y)
            
            ax3.legend(loc="lower right", fontsize=fontsize)
            
            ax3.ticklabel_format(axis='y', style='sci', scilimits=[-3,3])
            
            plt.ticklabel_format(style='plain')
        
    
        if (self.save):
            f = "%s/%s_dispersion2.pdf" %(self.folder_tag, self.file_tag)
            if (imaginary == True):
                f+= "_imag"
            f += ".pdf"
            fig_disp.savefig(f)
            

    
    def heatmap_both_species(self, time_index=-1, num_ticks=3, raster=True,
                             normalisation_method=0, file_type="pdf"):
    
        rgba_image = self.densities_to_colours(time_index, normalisation_method=normalisation_method)
    
        # Create figure and plot heatmap
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(rgba_image, origin="lower", interpolation="nearest",
                  rasterized=raster, 
                  extent=[self.x_points.min(), self.x_points.max(), self.y_points.min(), self.y_points.max()])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MultipleLocator(self.x_points.max()/(num_ticks-1)))
        ax.yaxis.set_major_locator(MultipleLocator(self.y_points.max()/(num_ticks-1)))

        if self.save:
            filename = f"{self.folder_tag}/{self.file_tag}_heatmap_RGB_t={self.solution_t[time_index]:.2f}." + file_type
            fig.savefig(filename, bbox_inches="tight", pad_inches=0.3, transparent=True)
            
    
    
    
    def animation_both_species(self, sample_rate=10, num_ticks=3,
                             normalisation_method=0):
        """
        Creates an animation where species u and v are displayed as a single RGB colormap.
        Red is proportional to species u, Blue is proportional to species v, Green is always zero.
        
        Parameters:
            sample_rate (int): How often to take a frame.
            num_ticks (int): Number of major ticks on axes.
        """
        
        fig, ax = plt.subplots(figsize=(8, 8))
    
        # Frame indices based on sample rate
        frame_indices = np.arange(0, self.solution_t.size-1, sample_rate, dtype=int)
    
        # Function to update the frame
        def update_frame(i):
            ax.clear()
            
            rgba_image = self.densities_to_colours(i,
                                     normalisation_method=normalisation_method,
                                     print_output=False)
    
            # Display image
            ax.imshow(rgba_image, origin="lower", interpolation="nearest", 
                      extent=[self.x_points.min(), self.x_points.max(), self.y_points.min(), self.y_points.max()])
    
            ax.set_title(f"Time = {self.solution_t[i]:.3f}")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_aspect('equal', adjustable='box')
    
            # Set axis ticks
            ax.xaxis.set_major_locator(MultipleLocator(self.x_points.max() / (num_ticks - 1)))
            ax.yaxis.set_major_locator(MultipleLocator(self.y_points.max() / (num_ticks - 1)))
    
        # Create animation
        anim = animation.FuncAnimation(fig, update_frame, frames=frame_indices, repeat=True, repeat_delay=5000)
    
        # Save animation if required
        if self.save:
            filename = f"{self.folder_tag}/{self.file_tag}_anim_RGB.mp4"
            anim.save(filename, writer=animation.FFMpegWriter(fps=10))
    
    
    

            
    # plots the trajectory of a single point in space in the u, v plane
    # useful for understanding oscillations
    def single_point_trajectory_plot(self, x_index=0, y_index=0, file_type="pdf"):
        
        fig, ax = plt.subplots()

        ax.plot(self.u_v_over_time[0, x_index, y_index, :],
                self.u_v_over_time[1, x_index, y_index, :],
                alpha=0.3)
        
        
        ax.set_xlabel("$c$")
        ax.set_ylabel("$\\rho$")
                
        x = self.x_points[x_index]
        y = self.y_points[y_index]
        ax.set_title("x=%.1f y=%.1f" %(x, y))
        
        
        if self.save:
            filename = f"{self.folder_tag}/{self.file_tag}_single_point_trajectory_x={x:.1f}_y=={y:.1f}." + file_type
            fig.savefig(filename, bbox_inches="tight", pad_inches=0.3, transparent=True)
    
    
    # Creates an animation showing the trajectory of a single point in space in the u, v plane.
    #  The trajectory is drawn progressively over time.
    def single_point_trajectory_animation(self, x_index=0, y_index=0, file_type="mp4", fps=20, frame_step=5):

        fig, ax = plt.subplots()
        ax.set_xlabel("$c$")
        ax.set_ylabel("$\\rho$")
        x = self.x_points[x_index]
        y = self.y_points[y_index]
        title = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center")
        
        # Data points
        u_vals = self.u_v_over_time[0, x_index, y_index, :]
        v_vals = self.u_v_over_time[1, x_index, y_index, :]
        time_vals = self.solution_t
        
        # Set axis limits to fit data
        ax.set_xlim(u_vals.min() - 0.0001, u_vals.max() + 0.0001)
        ax.set_ylim(v_vals.min() - 0.0001, v_vals.max() + 0.0001)
        
        line, = ax.plot([], [], lw=2, color='blue')
        
        def init():
            line.set_data([], [])
            title.set_text("")
            return line,
        
        def update(frame):
            idx = frame * frame_step
            if idx >= len(time_vals):
                idx = len(time_vals) - 1  # Ensure index is within bounds
            
            line.set_data(u_vals[:idx], v_vals[:idx])
            title.set_text(f"Trajectory at x={x:.1f}, y={y:.1f}, Time = {time_vals[idx]:.3f}")
        
        total_frames = len(time_vals) // frame_step
        anim = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, interval=50)
        
        if self.save:
            filename = f"{self.folder_tag}/{self.file_tag}_single_point_trajectory_x={x:.1f}_y={y:.1f}.{file_type}"
            anim.save(filename, writer=animation.FFMpegWriter(fps=fps))
            
    
    
    # Doesn't work so well because of the periodic boundary conditions
    # e.g. if an aggregate is split across the boundary, 
    # then the CoM would be placed in the middle of the domain
    def centre_of_mass_trajectory(self, time_groups_species=None, arrow_spacing=3):
        """
        Plots the center of mass trajectory for both species over time in x-y space.
        
        Parameters:
            time_groups_species (dict or None): Dictionary with keys 0 and 1, where each key maps to a list of lists.
                                                Each inner list contains time indices for a group of points.
            arrow_spacing (int): The interval at which to place arrows along the trajectory.
        """
        if time_groups_species is None:
            time_groups_species = {
                0: [list(np.arange(0, self.solution_t.size, max(1, self.solution_t.size // 20)))],
                1: [list(np.arange(0, self.solution_t.size, max(1, self.solution_t.size // 20)))],
            }

        colours = {0: 'darkgoldenrod', 1: 'turquoise'}
        labels = {0: 'Species 1 (u)', 1: 'Species 2 (v)'}
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for species_index in [0, 1]:
            if species_index not in time_groups_species:
                continue
            
            for group in time_groups_species[species_index]:
                com_x = []
                com_y = []
                
                for t in group:
                    density = self.u_v_over_time[species_index, :, :, t]
                    total_mass = np.sum(density)
                    
                    if total_mass > 0:
                        x_com = np.sum(self.x_points[:, None] * density) / total_mass
                        y_com = np.sum(self.y_points[None, :] * density) / total_mass
                        com_x.append(x_com)
                        com_y.append(y_com)
                        ax.text(x_com, y_com, f't={self.solution_t[t]:.1f}', fontsize=8, color=colours[species_index])
                    
                ax.plot(com_x, com_y, marker='o', color=colours[species_index], label=labels[species_index] if group is time_groups_species[species_index][0] else "")
                
                # Add arrows along the trajectory within the group
                for i in range(0, len(com_x) - 1, arrow_spacing):
                    dx = com_x[i + 1] - com_x[i]
                    dy = com_y[i + 1] - com_y[i]
                    ax.add_patch(matplotlib.patches.FancyArrow(com_x[i], com_y[i], dx, dy, color=colours[species_index], 
                                            head_width=0.1, head_length=0.1, length_includes_head=True))
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Center of Mass Trajectory")
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        
        if self.save:
            fig.savefig(f"{self.folder_tag}/{self.file_tag}_center_of_mass_trajectory.pdf", bbox_inches="tight")


    
    def animation_three_panels(self, sample_rate=10, num_ticks=3):
        """
        Creates a 3-panel animation:
        Left: RGB combined view (u + v)
        Middle: Species u (with v = 0)
        Right: Species v (with u = 0)
        All use the same RGB colormap logic.
        Time is shown once at the top.
        
        Note: this implementation is quite hacky, because I temporarily alter the
        class data in order to do something easily.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        ax_rgb, ax_u, ax_v = axes
    
        frame_indices = np.arange(0, self.solution_t.size-1, sample_rate, dtype=int)
    
        def update_frame(i):
            for ax in axes:
                ax.clear()
    
            # Panel 1: Combined u and v
            rgba_rgb = self.densities_to_colours(i, normalisation_method=0, print_output=False)
            ax_rgb.imshow(rgba_rgb, origin="lower", interpolation="nearest", 
                          extent=[self.x_points.min(), self.x_points.max(), self.y_points.min(), self.y_points.max()])
            ax_rgb.set_title("Both Species", fontsize=20)
            ax_rgb.set_aspect('equal')
            ax_rgb.xaxis.set_major_locator(MultipleLocator(self.x_points.max() / (num_ticks - 1)))
            ax_rgb.yaxis.set_major_locator(MultipleLocator(self.y_points.max() / (num_ticks - 1)))
            ax_rgb.set_xlabel("$x$")
            ax_rgb.set_ylabel("$y$")
    
            # Panel 2: Species u only (v = 0)
            original_v = self.u_v_over_time[1, :, :, i].copy()
            self.u_v_over_time[1, :, :, i] = 0  # Temporarily set v = 0
            rgba_u = self.densities_to_colours(i, normalisation_method=1, print_output=False)
            self.u_v_over_time[1, :, :, i] = original_v  # Restore v
            ax_u.imshow(rgba_u, origin="lower", interpolation="nearest",
                        extent=[self.x_points.min(), self.x_points.max(), self.y_points.min(), self.y_points.max()])
            ax_u.set_title("$c$, colour normalised for visiblity", fontsize=20)
            ax_u.set_aspect('equal')
            ax_u.xaxis.set_major_locator(MultipleLocator(self.x_points.max() / (num_ticks - 1)))
            ax_u.yaxis.set_major_locator(MultipleLocator(self.y_points.max() / (num_ticks - 1)))
            ax_u.set_xlabel("$x$")
            ax_u.set_ylabel("$y$")
    
            # Panel 3: Species v only (u = 0)
            original_u = self.u_v_over_time[0, :, :, i].copy()
            self.u_v_over_time[0, :, :, i] = 0  # Temporarily set u = 0
            rgba_v = self.densities_to_colours(i, normalisation_method=1, print_output=False)
            self.u_v_over_time[0, :, :, i] = original_u  # Restore u
            ax_v.imshow(rgba_v, origin="lower", interpolation="nearest",
                        extent=[self.x_points.min(), self.x_points.max(), self.y_points.min(), self.y_points.max()])
            ax_v.set_title("$\\rho$, colour normalised for visiblity", fontsize=20)
            ax_v.set_aspect('equal')
            ax_v.xaxis.set_major_locator(MultipleLocator(self.x_points.max() / (num_ticks - 1)))
            ax_v.yaxis.set_major_locator(MultipleLocator(self.y_points.max() / (num_ticks - 1)))
            ax_v.set_xlabel("$x$")
            ax_v.set_ylabel("$y$")
    
            # Shared time title
            fig.suptitle(f"Time = {self.solution_t[i]:.3f}", fontsize=32)
    
        anim = animation.FuncAnimation(fig, update_frame, frames=frame_indices, repeat=True, repeat_delay=5000)

        if self.save:
            filename = f"{self.folder_tag}/{self.file_tag}_anim_three_panels_rgbsplit.mp4"
            anim.save(filename, writer=animation.FFMpegWriter(fps=10))
