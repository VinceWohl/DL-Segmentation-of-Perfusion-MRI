%% Script to remap the intensity values in the label masks to integers
% author: Vw

%% Setup paths
inputFolder = 'C:\Users\Vincent Wohlfarth\Data\nnUNet_raw\Dataset001_PerfusionTerritories\labelsTr';
outputFolder = fullfile(inputFolder, 'converted_spm');
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

niiFiles = dir(fullfile(inputFolder, '*.nii'));

%% Run remapping of label values
for k = 1:length(niiFiles)
    filePath = fullfile(inputFolder, niiFiles(k).name);
    V = spm_vol(filePath);

    % Load and scale image
    raw_data = spm_read_vols(V);  % scaled values
    data = round(raw_data);       % round for robust matching

    % Get sorted unique values (excluding 0)
    uvals = unique(data(:));
    uvals(uvals == 0) = [];
    uvals = sort(uvals);

    % Initialize new image
    mapped = zeros(size(data), 'int64');

    if numel(uvals) >= 3
        mapped(data == uvals(1)) = 1;
        mapped(data == uvals(2)) = 2;
        mapped(data == uvals(3)) = 3;
    else
        warning('%s has less than 3 unique non-zero labels.', niiFiles(k).name);
    end

    % Save new NIfTI file (same name, different folder)
    V_out = V;
    V_out.fname = fullfile(outputFolder, niiFiles(k).name);
    V_out.dt = [64 0];  % 64 = double; no true int64 support in SPM write
    spm_write_vol(V_out, double(mapped));
end

disp('✅ All label masks remapped and saved.');