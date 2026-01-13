function simOut = run_physics(conc, temp, ph, flow)
    % run_physics.m - Octave entry point for the Bayesian Framework
    % Inputs: 
    %   conc (Molar), temp (C), ph, flow (m/s)
    % Returns: 
    %   simOut - Struct containing scalar rate, field data, and geometry
    
    % 1. Setup Geometry (Standardized for the Digital Twin)
    cathodeWidth = 3.0; % meters
    anodeWidth = 3.0; % meters
    widthTotal = cathodeWidth + anodeWidth;     
    deltaDistance = 5.0e-2; % Mesh size
    
    aNodes = round(anodeWidth/deltaDistance); 
    cNodes = round(cathodeWidth/deltaDistance);
    totNodes = aNodes + cNodes;
    
    electrolyteNodes = 20; 
    electrolyteHeight = electrolyteNodes * deltaDistance; 
    
    % 2. Material & Boundary Configuration
    edgesBC = {'neumann','neumann','neumann'}; 
    vApp = 0.0; % Short circuit
    aReactName = 'cuni'; 
    cReactName = 'i625'; 
    
    % 3. Environment Vector
    % IMPORTANT: galvCorrSim expects [cCl, T, pH, velocity] for butlerVolmer
    % This is DIFFERENT from polCurveMain.m which uses [T, pH, cCl, velocity]
    currentEnv = [conc, temp, ph, flow];
    fprintf('DEBUG: run_physics calling galvCorrSim with env=[cCl=%.2f, T=%.2f, pH=%.2f, v=%.2f]...\n', conc, temp, ph, flow); 
    fflush(stdout);
    
    try
        % Instantiate the Simulation Object (from legacy repo)
        % Note: galvanicCorrosion constructor calls aTafelPolCurve internally
        gC = galvanicCorrosion(edgesBC, widthTotal, electrolyteHeight, ...
                               deltaDistance, electrolyteNodes, totNodes, ...
                               aNodes, cNodes, aReactName, cReactName, ...
                               vApp, 12, currentEnv);
        
        % 4. Solve Physics (Laplace Equation)
        gC.aSim.phi = galvanicCorrosion.JacobiSolver(gC.aSim);
        
        % 5. Post-Process: Calculate Anodic Current Density
        % J = -kappa * dPhi/dy (Current flowing INTO electrolyte from metal is positive corrosion current)
        
        % Identify Anode indices (Bottom boundary)
        % From galvanicCorrosion.m: NBA = (aSim.NXc+1):(aSim.NXa+aSim.NXc)
        startIdx = gC.aSim.NXc + 1;
        endIdx = gC.aSim.NXc + gC.aSim.NXa;
        anodeIndices = startIdx:endIdx;
        
        % Potential Gradient at Anode Surface (Bottom row, index 1 in MATLAB usually, let's verify orientation)
        % JacobiSolver uses Top/Bottom logic. 
        % Check galvanicCorrosion.m line 69: Ebottom = EBL(i);
        % It seems row 1 is bottom. 
        % dPhi/dy = (phi(2, :) - phi(1, :)) / dy
        
        % The error says "gC(_,121): out of bound 21 (dimensions are 121x21)"
        % This means size(gC.aSim.phi) is likely [NX, NY] = [121, 21].
        % If we are accessing phi(1, ...), that is invalid if indices are (col, row). 
        % In MATLAB/Octave, size(A) = [rows, cols]. 
        % If size is [121, 21], then rows=121 (NX), cols=21 (NY).
        % So phi(x, y). 
        % The Anode is at the "bottom". In the meshgrid geometry, y=0 usually corresponds to index 1 of the Y dimension.
        % If A(x, y), then we want all X indices for Y=1 and Y=2.
        
        phi_surface = gC.aSim.phi(anodeIndices, 1);
        phi_above   = gC.aSim.phi(anodeIndices, 2);
        
        dy = gC.aSim.dy;
        kappa = 5.0; % Seawater conductivity approx (S/m). Should be in gC.aSim.conductivity if set.
        try
            if isfield(gC.aSim, 'conductivity') && ~isempty(gC.aSim.conductivity)
                 kappa = gC.aSim.conductivity;
            end
        catch
            % Use default kappa if conductivity not available
        end
        
        % Flux J = -kappa * grad(phi). 
        % Current INTO domain is positive.
        % Node 1 is boundary. Node 2 is interior.
        % J = -kappa * (phi(2) - phi(1))/dy ?
        % If phi(1) is Anode (Anodic), it should be higher than phi(2). 
        % Current flows High -> Low. Anode -> Electrolyte.
        % So J = (phi(1) - phi(2))/dy * kappa
        
        currentDensityProfile = kappa * (phi_surface - phi_above) / dy;
        
        % Average Current Density (A/m^2)
        % Use sum/length instead of mean() for Octave compatibility
        corrosionRate = sum(currentDensityProfile(:)) / numel(currentDensityProfile);
        
        fprintf('DEBUG: Calculated Corrosion Rate from Flux: %.6e A/m^2\n', corrosionRate);
        fflush(stdout);

        % Fallback if something is wrong (e.g. 0)
        if corrosionRate == 0
             fprintf('DEBUG: Warning, computed rate is 0. Using theoretical uncoupled value.\n');
             if ~isempty(gC.aSim.corrosionCurrentAnodic)
                 % Use the value from constructor (max absolute current)
                 [~, idx] = max(abs(gC.aSim.corrosionCurrentAnodic));
                 corrosionRate = gC.aSim.corrosionCurrentAnodic(idx) / anodeWidth;
             end
        end
        
        % Construct Output Struct
        simOut.corrosionRate = double(corrosionRate);
        simOut.phi = double(gC.aSim.phi);
        simOut.xpos = double(gC.aSim.xpos);
        simOut.ypos = double(gC.aSim.ypos);
        simOut.anodeIndices = double(anodeIndices);
        simOut.currentDensityProfile = double(currentDensityProfile);
        simOut.conductivity = double(kappa);
        
    catch ME
        rethrow(ME);
    end
end
