addpath('/projects/hpcapps/nsawant/corrosion/corrosion-modeling-applications/pipe-spool-model/supportingClasses');
try
    x = galvanicCorrosion();
    disp('Class loaded');
catch e
    disp(e.message);
end
