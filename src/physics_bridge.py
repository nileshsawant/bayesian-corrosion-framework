import os
import subprocess
import re
import time
import select
import numpy as np

class PhysicsBridge:
    def __init__(self, dependency_root):
        """
        dependency_root: Path to 'corrosion-modeling-applications'
        """
        self.dep_path = os.path.abspath(dependency_root)
        # Assumes wrapper is in the same directory as this file
        self.wrapper_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'physics_wrapper.m')
        
    def run_sim(self, params):
        """
        Run simulation via octave-cli subprocess.
        params: list [NaCl, T, pH, v]
        """
        # Ensure params are simple values
        # octave-cli src/physics_wrapper.m <dep_path> <c> <T> <pH> <v>
        
        # Use absolute path to octave-cli
        octave_bin = '/projects/hpcapps/nsawant/corrosion/env/bin/octave-cli'
        if not os.path.exists(octave_bin):
            # Fallback to searching in PATH
            octave_bin = 'octave-cli'

        cmd = [
            octave_bin,
            '--no-gui',
            '--no-window-system',
            '--no-history',
            '--no-init-file',
            '--no-site-file',
            '--silent',
            self.wrapper_path,
            self.dep_path,
            str(params[0]),
            str(params[1]),
            str(params[2]),
            str(params[3])
        ]
        
        print(f"DEBUG: Executing: {cmd}")
        full_output = []
        start_time = time.time()
        timeout = 3600  # 1 hour timeout
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            while True:
                # Check for timeout
                if time.time() - start_time > timeout:
                    process.kill()
                    print("Error: Octave execution timed out.")
                    return None
                
                # Non-blocking read check
                reads = [process.stdout.fileno()]
                ret = select.select(reads, [], [], 1.0) # 1 sec wait
                
                if ret[0]:
                    line = process.stdout.readline()
                    if line:
                        print(line, end='') # Stream output to user
                        full_output.append(line)
                    else:
                        # EOF
                        break
                
                if process.poll() is not None:
                    # Process finished, consume remaining output
                    for line in process.stdout:
                        print(line, end='')
                        full_output.append(line)
                    break
                    
            out = "".join(full_output)
            
            # Parse outputs
            result_data = {}
            
            # 1. Scalar Rate
            match_scalar = re.search(r'RESULT_SCALAR:\s*([0-9\.eE\-\+]+)', out)
            if match_scalar:
                result_data['corrosionRate'] = float(match_scalar.group(1))
            
            # 2. Dimensions
            match_dims = re.search(r'PHI_DIMS:(\d+),(\d+)', out)
            rows, cols = 0, 0
            if match_dims:
                rows = int(match_dims.group(1))
                cols = int(match_dims.group(2))
            
            # 3. Phi Field
            pattern_block = r'PHI_START\s+(.*?)\s+PHI_END'
            match_block = re.search(pattern_block, out, re.DOTALL)
            
            if match_block and rows > 0 and cols > 0:
                try:
                    raw_vals = match_block.group(1).strip().split()
                    # Convert to float array
                    data_vec = np.array([float(x) for x in raw_vals])
                    
                    if len(data_vec) == rows * cols:
                        # Reshape using Fortran order (Column-Major) to match Octave
                        phi = data_vec.reshape((rows, cols), order='F')
                        result_data['phi'] = phi
                    else:
                        print(f"Error: Mismatch in data length. Expected {rows*cols}, got {len(data_vec)}")
                except Exception as e:
                    print(f"Error parsing phi block: {e}")
            
            # 4. Current Density Profile
            current_density_match = re.search(r'CURRENT_DENSITY_START\s+(.*?)\s+CURRENT_DENSITY_END', out, re.DOTALL)
            current_density_len_match = re.search(r'CURRENT_DENSITY_LENGTH:(\d+)', out)
            
            if current_density_match and current_density_len_match:
                try:
                    cd_length = int(current_density_len_match.group(1))
                    cd_vals = current_density_match.group(1).strip().split()
                    cd_array = np.array([float(x) for x in cd_vals])
                    
                    if len(cd_array) == cd_length:
                        result_data['currentDensity'] = cd_array
                    else:
                        print(f"Warning: Current density length mismatch. Expected {cd_length}, got {len(cd_array)}")
                except Exception as e:
                    print(f"Error parsing current density: {e}")
            
            # 5. X and Y coordinates
            xpos_match = re.search(r'XPOS_START\s+(.*?)\s+XPOS_END', out, re.DOTALL)
            ypos_match = re.search(r'YPOS_START\s+(.*?)\s+YPOS_END', out, re.DOTALL)
            
            if xpos_match:
                try:
                    xpos_vals = xpos_match.group(1).strip().split()
                    result_data['xpos'] = np.array([float(x) for x in xpos_vals])
                except Exception as e:
                    print(f"Error parsing xpos: {e}")
                    
            if ypos_match:
                try:
                    ypos_vals = ypos_match.group(1).strip().split()
                    result_data['ypos'] = np.array([float(x) for x in ypos_vals])
                except Exception as e:
                    print(f"Error parsing ypos: {e}")
            
            # Check Return Code
            if process.returncode is not None and process.returncode != 0:
                print(f"Octave Execution Warning: Process returned code {process.returncode}")
                # If we have valid data, we proceed. If not, we fail.
                if not ('phi' in result_data or 'corrosionRate' in result_data):
                     return None
            
            if 'phi' in result_data:
                # Add scalar alias for backward compatibility
                if 'corrosionRate' in result_data:
                    result_data['scalar'] = result_data['corrosionRate']
                return result_data
            elif 'corrosionRate' in result_data:
                 print("Warning: returning only scalar rate, no field found.")
                 return result_data
            else:
                 print("Error: No valid result found in Octave output.")
                 # Try legacy fallback
                 match = re.search(r'RESULT:\s*([0-9\.eE\-\+]+)', out)
                 if match:
                     # If legacy matched, it's just a scalar, but orchestrator might expect dict
                     return {'corrosionRate': float(match.group(1))}
                 return None

        except Exception as e:
            print(f"Error executing Octave: {e}")
            return None
