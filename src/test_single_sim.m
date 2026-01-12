% Test a single simulation with two different flow velocities
% to debug why outputs are identical

clear all;

dep_path = '../../corrosion-modeling-applications';
addpath(fullfile(dep_path, 'pipe-spool-model', 'supportingClasses'));
addpath(fullfile(dep_path, 'pipe-spool-model', 'materials'));
addpath(fullfile(dep_path, 'polarization-curve-modeling'));

% Test parameters
conc = 0.1;  % M
temp = 278.0; % K
ph = 6.0;
flow1 = 0.1;  % m/s
flow2 = 3.0;  % m/s

fprintf('Running two simulations with different flow velocities...\n');
fprintf('Parameters: NaCl=%.2f M, T=%.1f K, pH=%.1f\n', conc, temp, ph);
fprintf('Flow 1: %.2f m/s\n', flow1);
fprintf('Flow 2: %.2f m/s\n\n', flow2);

% Run simulation 1
result1 = run_physics(conc, temp, ph, flow1);
fprintf('Simulation 1 complete. Corrosion rate: %.6f\n', result1.corrosionRate);
fprintf('  Phi min: %.6e, max: %.6e, mean: %.6e\n', min(result1.phi(:)), max(result1.phi(:)), mean(result1.phi(:)));

% Run simulation 2
result2 = run_physics(conc, temp, ph, flow2);
fprintf('\nSimulation 2 complete. Corrosion rate: %.6f\n', result2.corrosionRate);
fprintf('  Phi min: %.6e, max: %.6e, mean: %.6e\n', min(result2.phi(:)), max(result2.phi(:)), mean(result2.phi(:)));

% Compare
diff = result2.phi - result1.phi;
fprintf('\nDifference between simulations:\n');
fprintf('  Mean: %.6e\n', mean(diff(:)));
fprintf('  Std:  %.6e\n', std(diff(:)));
fprintf('  Max:  %.6e\n', max(abs(diff(:))));

if max(abs(diff(:))) < 1e-10
    fprintf('\n⚠️  WARNING: Outputs are essentially identical!\n');
    fprintf('  This suggests flow velocity is not affecting the simulation.\n');
else
    fprintf('\n✓ Outputs differ - flow velocity is working!\n');
end
