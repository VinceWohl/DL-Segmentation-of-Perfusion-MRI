% author: VW
% Script to prepare the input for the nnUNet model in an adequate format

% Directory containing the NIfTI files
dataDir = 'C:\Users\Vincent Wohlfarth\Data\nnUNet_raw\Dataset001_PerfusionTerritories\labelsTr';
niiFiles = dir(fullfile(dataDir, '*.nii'));

% Loop through each NIfTI file
for i = 1:length(niiFiles)
    filePath = fullfile(dataDir, niiFiles(i).name);
    nii = niftiread(filePath);
    niiInfo = niftiinfo(filePath);
    
    % Initialize list of empty slice indices
    emptySlices = [];

    % Check each 2D slice along the 3rd dimension
    for z = 1:size(nii, 3)
        if all(nii(:, :, z) == 0, 'all')
            emptySlices(end+1) = z;
        end
    end

    % Display results
    if isempty(emptySlices)
        fprintf('File %s: No empty slices\n', niiFiles(i).name);
    else
        fprintf('File %s: %d empty slices at indices: %s\n', ...
            niiFiles(i).name, length(emptySlices), mat2str(emptySlices));
    end
end
