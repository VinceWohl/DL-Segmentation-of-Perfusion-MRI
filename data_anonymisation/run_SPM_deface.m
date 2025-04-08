% author: VW
% Matlab needs to know where SPM folder is located


%% Define root directory
root_dir = 'C:\Users\Vincent Wohlfarth\Data\ssASL-Project-anonymisation';

%% Recursively search for .nii files
nii_files = dir(fullfile(root_dir, '**', '*.nii'));
image_paths = {};
for i = 1:length(nii_files)
    name = nii_files(i).name;
    % Include only files that contain 'T1w' or 'FLAIR', and exclude those with 'deface' or 'anon'
    if (contains(name, 'T1w', 'IgnoreCase', true) || contains(name, 'FLAIR', 'IgnoreCase', true)) && ...
       ~contains(name, 'deface', 'IgnoreCase', true) && ...
       ~contains(name, 'anon', 'IgnoreCase', true)
        image_paths{end+1} = fullfile(nii_files(i).folder, name); %#ok<SAGROW>
    end
end

%% Set up the SPM batch
matlabbatch = {};
matlabbatch{1}.spm.util.deface.images = image_paths';

%% Initialize SPM and run the batch
spm('defaults', 'FMRI');
spm_jobman('initcfg');
spm_jobman('run', matlabbatch);