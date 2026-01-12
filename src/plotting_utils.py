import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_simulation_results(sim_data, output_dir, run_id="sim"):
    """
    Plots the 2D potential field and the corrosion profile from the simulation data.
    
    Args:
        sim_data (dict): Dictionary returned by PhysicsBridge.run_sim() loaded from .mat
        output_dir (str): Directory to save plots
        run_id (str): Identifier for naming files
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Extract data
    phi = sim_data['phi']
    
    # Extract data
    phi = sim_data['phi']
    
    # -------------------------------------------------------------
    # Robust Coordinate Handling
    # -------------------------------------------------------------
    # Goal: Always end up with valid X, Y meshgrids matching Phi (or Phi.T)
    
    xpos_raw = sim_data.get('xpos')
    ypos_raw = sim_data.get('ypos')
    
    using_fallback = True
    
    if xpos_raw is not None and ypos_raw is not None:
        # Strategy: Flatten everything to 1D unique vectors first
        x_vec = xpos_raw.flatten()
        y_vec = ypos_raw.flatten()
        
        # Check if they look like Axes (length matches dimensions of Phi)
        # Phi is (M, N) or (N, M)
        # x_vec size should match one dim, y_vec size the other.
        
        dim_match = False
        
        # Check alignment with Phi
        if (x_vec.size == phi.shape[0] and y_vec.size == phi.shape[1]) or \
           (x_vec.size == phi.shape[1] and y_vec.size == phi.shape[0]):
           
           # They are likely Axes.
           # Determine orientation. Matplotlib meshgrid(x, y) produces (len(y), len(x))
           
           # If Phi is (len(x), len(y)) -> we need Phi.T to match (len(y), len(x))
           # If Phi is (len(y), len(x)) -> it matches directly.
           
           # Let's just create the meshgrid and verify
           X, Y = np.meshgrid(x_vec, y_vec) # Shape: (len(y), len(x))
           
           if X.shape == phi.shape:
               Phi_Plot = phi
               using_fallback = False
           elif X.shape == phi.T.shape:
               Phi_Plot = phi.T
               using_fallback = False
           else:
               # Swapped inputs?
               X_alt, Y_alt = np.meshgrid(y_vec, x_vec)
               if X_alt.shape == phi.shape:
                   X, Y = X_alt, Y_alt
                   Phi_Plot = phi
                   using_fallback = False
                   
        # Check if they are already flattened Grids (size == phi.size)
        elif x_vec.size == phi.size and y_vec.size == phi.size:
            # Reshape them to match Phi
            X = x_vec.reshape(phi.shape)
            Y = y_vec.reshape(phi.shape)
            Phi_Plot = phi
            using_fallback = False
            
    if using_fallback:
        print("Using Fallback Grid Generation (Dimensions inferred from Phi shape)")
        # If we failed to align, generate synthetic grid
        # Assume Rows = Y (Depth), Cols = X (Length)
        rows, cols = phi.shape
        # Standard Domain: -3m to 3m length, 0 to 1m depth (inverted Y usually)
        x_range = np.linspace(-3000, 3000, cols) # mm
        y_range = np.linspace(0, 1000, rows) # mm
        
        # If rows > cols, maybe transposed?
        if rows > cols:
             x_range = np.linspace(-3000, 3000, rows)
             y_range = np.linspace(0, 1000, cols)
             X, Y = np.meshgrid(y_range, x_range) # Potentially swapped naming
             Phi_Plot = phi
             # This is a heuristic guess, might be rotated.
        else:
             X, Y = np.meshgrid(x_range, y_range)
             Phi_Plot = phi

    # Final Verification
    if X.shape != Phi_Plot.shape:
        print(f"CRITICAL PLOTTING ERROR: Shape mismatch persists. X:{X.shape}, Phi:{Phi_Plot.shape}")
        # Force Transpose
        if X.shape == Phi_Plot.T.shape:
            Phi_Plot = Phi_Plot.T

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Contour Plot
    levels = np.linspace(np.min(Phi_Plot), np.max(Phi_Plot), 25)
    cp = ax.contourf(X, Y, Phi_Plot, levels=levels, cmap='viridis')
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label('Potential ($V_{SCE}$)')
    
    # Overlay Metal Labels
    
    ax.set_title(f"Potential Distribution (Run {run_id})")
    ax.set_xlabel('Position (mm)') # Changed to mm based on orchestrator ranges
    ax.set_ylabel('Electrolyte Depth (mm)')
    ax.set_ylim([np.min(Y), np.max(Y)])
    
    # Add markers for metals
    ax.plot([-3, 0], [0, 0], 'r-', linewidth=4, label='Cathode (I625)')
    ax.plot([0, 3], [0, 0], 'b-', linewidth=4, label='Anode (CuNi)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_id}_potential_field.png"), dpi=150)
    plt.close(fig)
    
    # ---------------------------------------------------------
    # 2. Plot Corrosion Rate Profile along Anode
    # ---------------------------------------------------------
    
    # Extract corrosion specific data if available
    # These usage safety checks prevent NameErrors
    current_density = sim_data.get('currentDensityProfile', None)
    anode_indices = sim_data.get('anodeIndices', None)

    if current_density is not None and anode_indices is not None:
        current_density = current_density.flatten()
        anode_indices = anode_indices.flatten().astype(int) - 1 # 0-based
        
        # Need to extract 1D x-axis for profile plot
        # If X is a meshgrid, we can take the first row (assuming variability is along x)
        if X.ndim == 2:
            x_axis_1d = X[0, :] # Assuming X varies across columns
        else:
            x_axis_1d = X.flatten()

        # Handle Indices range safety
        valid_indices = anode_indices[anode_indices < len(x_axis_1d)]
        
        if len(valid_indices) > 0:
            x_anode = x_axis_1d[valid_indices]
            # Ensure current density matches length
            if len(current_density) > len(valid_indices):
                 current_density_plot = current_density[valid_indices]
            else:
                 current_density_plot = current_density
                 x_anode = x_anode[:len(current_density_plot)]
            
            # Flip sign (Anodic current is typically positive in electrochemistry conventions, but solver might output negative)
            # User requested flip.
            current_density_plot = -1.0 * current_density_plot
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_anode, current_density_plot, 'b-o', markersize=3, label='Corrosion Current')
            
            ax.set_title(f"Corrosion Rate Profile along Anode (Run {run_id})")
            ax.set_xlabel('Position along Pipe (mm)')
            ax.set_ylabel('Current Density ($A/m^2$)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{run_id}_corrosion_profile.png"), dpi=150)
            plt.close(fig)
        else:
            print("Warning: Could not align Anode indices with X-axis for profile plotting.")
    else:
        print("Note: Skipping Corrosion Profile Plot (Missing Data in sim_data)")
    
    print(f"Plots saved to {output_dir}")
