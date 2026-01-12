args = argv();
fprintf('DEBUG: Wrapper started. Args: %d\n', length(args)); fflush(stdout);
% Expect: dep_path, conc, temp, ph, flow (5 args)
if length(args) < 5
    error('Usage: octave-cli physics_wrapper.m <dep_path> <conc> <temp> <ph> <flow>');
end

dep_path = args{1};
conc = str2double(args{2});
temp = str2double(args{3});
ph   = str2double(args{4});
flow = str2double(args{5});

% Add paths (manually concatenate instead of using fullfile)
addpath([dep_path '/pipe-spool-model/supportingClasses']);
addpath([dep_path '/pipe-spool-model/materials']);
addpath([dep_path '/pipe-spool-model/plotFunctions']);
addpath([dep_path '/polarization-curve-modeling']);

try
   result = run_physics(conc, temp, ph, flow);
   
   % Print Scalar Result
   fprintf('RESULT_SCALAR:%.6f\n', result.corrosionRate);

   % Print Phi Field
   % Linearize column-by-column (Standard Octave/Matlab behavior)
   fprintf('PHI_START\n');
   fprintf('%.6e\n', result.phi(:));
   fprintf('PHI_END\n');
   
   % Print Dimensions
   [r, c] = size(result.phi);
   fprintf('PHI_DIMS:%d,%d\n', r, c);
   
   % Print Current Density Profile (1D array along anode)
   fprintf('CURRENT_DENSITY_START\n');
   fprintf('%.6e\n', result.currentDensityProfile(:));
   fprintf('CURRENT_DENSITY_END\n');
   fprintf('CURRENT_DENSITY_LENGTH:%d\n', length(result.currentDensityProfile));
   
   % Print X and Y coordinate vectors
   fprintf('XPOS_START\n');
   fprintf('%.6e\n', result.xpos(:));
   fprintf('XPOS_END\n');
   
   fprintf('YPOS_START\n');
   fprintf('%.6e\n', result.ypos(:));
   fprintf('YPOS_END\n');

   fflush(stdout);
   exit(0);
catch e
   fprintf('ERROR:%s\n', e.message);
   exit(1);
end
