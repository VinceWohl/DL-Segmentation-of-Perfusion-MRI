%% zero_mask_files_subp023.m
% Zero out both mask files in-place (keeps header/orientation).
% Requires SPM on your MATLAB path.

files = {
'D:\Data\anon_DATA_250919\DATA_patients\First_visit\output\sub-p023\task-AIR\ASL\ssLICA\PerfTerrMask\mask_LICA_manual_Corrected.nii'
'D:\Data\anon_DATA_250919\DATA_patients\Second_visit\output\sub-p023\task-AIR\ASL\ssLICA\PerfTerrMask\mask_LICA_manual_Corrected.nii'
};

% If needed:
% addpath('D:\Code\DLSegPerf\spm');

for i = 1:numel(files)
    fn = files{i};
    if exist(fn,'file') ~= 2
        warning('Missing file: %s', fn);
        continue;
    end

    V = spm_vol(fn);
    X = spm_read_vols(V);

    X(:) = 0;                 % set all voxels to zero
    V.dt    = [2 0];          % uint8 datatype
    V.pinfo = [1;0;0];        % identity scaling (no slope/offset)

    spm_write_vol(V, uint8(X));
    fprintf('Zeroed: %s\n', fn);
end

disp('Done.');
