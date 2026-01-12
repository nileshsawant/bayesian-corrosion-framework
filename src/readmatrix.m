function M = readmatrix(filename)
    % READMATRIX Compatibility wrapper for Octave
    % Reads numeric matrix from CSV, searching in path if needed.
    
    % Try checking if file exists directly
    if exist(filename, 'file')
        M = dlmread(filename, ',');
        return;
    end
    
    % Try finding in path
    fullpath = file_in_path(path, filename);
    if ~isempty(fullpath)
        M = dlmread(fullpath, ',');
    else
        % Fallback, maybe dlmread finds it
        try
             M = dlmread(filename, ',');
        catch
             error(['readmatrix: File not found: ' filename]);
        end
    end
end
