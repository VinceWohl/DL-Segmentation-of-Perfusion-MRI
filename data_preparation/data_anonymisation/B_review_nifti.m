% author: VW
% Script to review all anonymised NIfTI files and their headers using SPM viewer (Linux-compatible)

%% Define base folder to search for anonymised NIfTI files
%anon_root = '/home/BRAIN/vwohlfarth/Documents/anon_Data';
anon_root = 'C:\Users\Vincent Wohlfarth\Data\anon_Data';

 %    Optionally set SPM path
%spm_path = '/home/BRAIN/vwohlfarth/Documents/MATLAB/spm';
spm_path = 'C:\Users\Vincent Wohlfarth\MatlabProjects\spm';
if exist(fullfile(spm_path, 'spm.m'), 'file')
    addpath(spm_path);
else
    warning('SPM not found at %s. Make sure it''s in your MATLAB path.', spm_path);
end

% Find all 'anon_*.nii' files recursively
anon_files = dir(fullfile(anon_root, '**', 'anon_*.nii'));
 
if isempty(anon_files)
    error('No anonymised NIfTI files found in %s', anon_root);
end

fprintf('Found %d anonymised files.\n', numel(anon_files));

% Loop over all anonymised files
for i = 1:length(anon_files)
    file_path = fullfile(anon_files(i).folder, anon_files(i).name);
    fprintf('\n\n=== [%d/%d] %s ===\n', i, length(anon_files), file_path);

    % Try to get header with SPM, fallback to niftiinfo
    try
        V = spm_vol(file_path);
        disp('--- NIfTI Header Information (SPM) ---');
        disp(V);
    catch
        warning('SPM error or not found, using MATLAB niftiinfo.');
        try
            V = niftiinfo(file_path);
            disp('--- NIfTI Header Information (MATLAB) ---');
            disp(V);
        catch ME
            warning('Could not read header: %s', ME.message);
            continue;
        end
    end

    % Display the image using SPM viewer
    try
        spm_image('Display', file_path);
    catch ME
        warning('Could not open viewer for %s: %s\nSkipping...', file_path, ME.message);
        continue;
    end

    % Wait for review input
    fprintf('Press any key to continue to next file...\n');
    pause;
    close all;
end

disp('--- Review complete ---');