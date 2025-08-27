%% A_crop_to_ROI.m
% Delete first & last slice for all 16-slice NIfTI volumes under rootDir.
% Keeps slices 2..15 (-> 14 slices) and overwrites the original file.
% Requires SPM on the MATLAB path.

%% -------------------- SETTINGS --------------------
rootDir = 'C:\Users\Vincent Wohlfarth\Data\anon_Data_250808';
dryRun  = false;   % true = print only, no writes
%% --------------------------------------------------

if exist('spm_vol','file') ~= 2
    warning('SPM not found on path. Add it (addpath(...)) before running.');
end

fprintf('\n=== Crop to ROI (remove slices 1 & 16 for 16-slice volumes) ===\n');
fprintf('Root: %s | Dry-run: %d\n\n', rootDir, dryRun);

niiList = dir(fullfile(rootDir, '**', '*.nii'));   % recursive (R2016b+)
nFound=0; nCropped=0; nSkip=0; nErr=0;

for i = 1:numel(niiList)
    fn = fullfile(niiList(i).folder, niiList(i).name);
    try
        V = spm_vol(fn);
        if numel(V) ~= 1    % skip 4D or multi-header files
            nSkip = nSkip + 1; continue;
        end
        if V.dim(3) ~= 16
            nSkip = nSkip + 1; continue;
        end
        nFound = nFound + 1;

        if dryRun
            fprintf('[DRY] Would crop: %s (16 -> 14)\n', fn);
            continue;
        end

        img  = spm_read_vols(V);     % double
        imgC = img(:,:,2:15);        % keep slices 2..15

        Vout       = V;
        Vout.fname = fn;             % overwrite
        Vout.dim(3)= size(imgC,3);
        Vout.descrip = sprintf('%s | Cropped to slices 2..15', V.descrip);

        % Shift origin by +1 slice so world coordinates remain consistent
        % (new slice #1 equals old slice #2)
        try
            Vout.mat(:,4) = V.mat(:,4) + V.mat(:,3);
        catch
            % If unexpected mat, carry on without shift
        end

        % Preserve datatype if present
        if isfield(V,'dt'), Vout.dt = V.dt; end

        spm_write_vol(Vout, imgC);
        fprintf('CROPPED: %s (16 -> 14 slices)\n', fn);
        nCropped = nCropped + 1;

    catch ME
        nErr = nErr + 1;
        fprintf(2,'ERROR: %s\n   -> %s\n', fn, ME.message);
    end
end

fprintf('\n=== Summary ===\nFound 16-slice: %d | Cropped: %d | Skipped: %d | Errors: %d\nDone.\n', ...
    nFound, nCropped, nSkip, nErr);
