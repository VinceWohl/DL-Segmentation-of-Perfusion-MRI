% Author: VW
% Prepare Dataset001_PerfusionTerritories (nnU-Net raw) with TWO channels:
%   _0000 = CBF (CBF_3_BRmsk_CSF.nii)
%   _0001 = FLAIR (prefer hemisphere-specific FLAIR_coreg; fallback to native FLAIR)
% Labels: PerfTerrMask
% NOTE: Does NOT rebuild dataset.json

clear; clc;

%% -------- SETTINGS --------
prefer_coreg_FLAIR = true;   % prefer hemisphere-specific FLAIR_coreg
verbose = true;

%% -------- PATHS --------
src_root = 'C:\Users\Vincent Wohlfarth\Data\anon_Data_250808';
dst_root = 'C:\Users\Vincent Wohlfarth\Data\nnUNet_raw\Dataset001_PerfusionTerritories';

% Target subfolders
dst_imagesTr = fullfile(dst_root, 'imagesTr');
dst_labelsTr = fullfile(dst_root, 'labelsTr');
dst_imagesTs = fullfile(dst_root, 'imagesTs');

cellfun(@(p) ~exist(p,'dir') && mkdir(p), {dst_imagesTr, dst_labelsTr, dst_imagesTs});

%% -------- SPLIT TABLE --------
xls_path = fullfile('C:\Users\Vincent Wohlfarth\CodingProjects\DLSegPerf', ...
    'data_preparation','Quality','final_quality_check', ...
    'data_completeness_report_20250820_171756.xlsx');

% Keep the exact header names from Excel
T = readtable(xls_path, 'Sheet', 'Data Completeness', 'VariableNamingRule', 'preserve');

% Find the split column robustly
splitColIdx = find(contains(T.Properties.VariableNames, 'Training', 'IgnoreCase', true) & ...
                   contains(T.Properties.VariableNames, 'Split',    'IgnoreCase', true), 1);
if isempty(splitColIdx)
    error('Could not find the "Training/Validation + Testing Split" column in the Excel.');
end
splitVarName = T.Properties.VariableNames{splitColIdx};

%% -------- VISIT MAP --------
visit_map = containers.Map( ...
    {'First_visit','Second_visit','Third_visit'}, ...
    {'v1','v2','v3'});

%% -------- HEMISPHERES (source folder names & tags) --------
hemi_src  = {'ssLICA','ssRICA'};   % left / right source subfolders
hemi_tag  = {'L','R'};             % Z = L or R
maskName  = {'mask_LICA_manual_Corrected.nii','mask_RICA_manual_Corrected.nii'};

%% -------- PROCESS --------
nTr = 0; nTs = 0; nSkipped = 0;

for i = 1:height(T)
    % Force char to keep isfile() scalar-logical
    grp   = char(strtrim(string(T.("Group"){i})));      % 'DATA_HC' or 'DATA_patients'
    subjS = strtrim(string(T.("Subject"){i}));          % e.g., "sub-p001"
    visit = char(strtrim(string(T.("Visit"){i})));      % 'First_visit', etc.
    splitVal = strtrim(string(T.(splitVarName)(i)));

    % skip if split is 'x' or empty
    if strlength(splitVal) == 0 || strcmpi(splitVal, "x")
        nSkipped = nSkipped + 1;
        if verbose, fprintf('[SKIP row %d] %s | %s | %s (split=%s)\n', i, grp, subjS, visit, splitVal); end
        continue;
    end
    if ~isKey(visit_map, visit)
        nSkipped = nSkipped + 1;
        if verbose, fprintf('[SKIP row %d] Unknown visit tag: %s | %s | %s\n', i, grp, subjS, visit); end
        continue;
    end

    subj     = char(subjS);                 % 'sub-p001'
    subj_num = extractAfter(subj, 'sub-p'); % '001'
    vtag     = visit_map(visit);

    % Base source paths
    subj_base = fullfile(src_root, grp, visit, 'output', subj);
    asl_base  = fullfile(subj_base, 'task-AIR', 'ASL');

    for h = 1:2
        hemi = hemi_src{h};
        Z    = hemi_tag{h};

        % ---- inputs/label ----
        cbf_src  = fullfile(asl_base, hemi, 'CBF_nativeSpace', 'CBF_3_BRmsk_CSF.nii');
        mask_src = fullfile(asl_base, hemi, 'PerfTerrMask',    maskName{h});

        % FLAIR candidates (coreg first if preferred)
        flair_coreg_name  = sprintf('anon_r%s_FLAIR.nii', subj);  % anon_rsub-p001_FLAIR.nii
        flair_native_name = sprintf('anon_%s_FLAIR.nii',   subj); % anon_sub-p001_FLAIR.nii
        flair_coreg_path  = fullfile(asl_base, hemi, 'FLAIR_coreg', flair_coreg_name);
        flair_native_path = fullfile(subj_base, 'FLAIR', flair_native_name);

        used_fallback = false;
        if prefer_coreg_FLAIR
            if isfile(flair_coreg_path)
                flair_src = flair_coreg_path;
            elseif isfile(flair_native_path)
                flair_src = flair_native_path; used_fallback = true;
            else
                flair_src = '';
            end
        else
            if isfile(flair_native_path)
                flair_src = flair_native_path;
            elseif isfile(flair_coreg_path)
                flair_src = flair_coreg_path; used_fallback = true;
            else
                flair_src = '';
            end
        end

        % ---- check all required inputs exist ----
        if ~isfile(cbf_src) || ~isfile(mask_src) || isempty(flair_src)
            nSkipped = nSkipped + 1;
            if verbose
                fprintf('[MISS] %s | %s | %s | %s  (cbf:%d, flair:%d, mask:%d)\n', ...
                    grp, subj, visit, hemi, isfile(cbf_src), ~isempty(flair_src), isfile(mask_src));
            end
            continue;
        end
        if verbose && used_fallback
            fprintf('[WARN Fallback FLAIR] Using native FLAIR for %s %s %s %s\n', grp, subj, visit, hemi);
        end

        % ---- destinations ----
        base    = sprintf('PerfTerr%s-%s-%s', subj_num, vtag, Z);
        img000  = [base '_0000.nii'];    % CBF
        img001  = [base '_0001.nii'];    % FLAIR
        lbl_out = [base '.nii'];         % label

        switch char(splitVal)
            case 'Training/Validation'
                copyfile(cbf_src,   fullfile(dst_imagesTr, img000), 'f');
                copyfile(flair_src, fullfile(dst_imagesTr, img001), 'f');
                copyfile(mask_src,  fullfile(dst_labelsTr, lbl_out), 'f');
                nTr = nTr + 1;
                if verbose, fprintf('[TR]  %s %s %s %s -> %s\n', grp, subj, visit, hemi, base); end

            case 'Testing'
                copyfile(cbf_src,   fullfile(dst_imagesTs, img000), 'f');
                copyfile(flair_src, fullfile(dst_imagesTs, img001), 'f');
                % no labels for test set
                nTs = nTs + 1;
                if verbose, fprintf('[TS]  %s %s %s %s -> %s\n', grp, subj, visit, hemi, base); end

            otherwise
                nSkipped = nSkipped + 1;
                if verbose, fprintf('[SKIP split] %s | %s | %s | %s (split=%s)\n', ...
                                    grp, subj, visit, hemi, splitVal); end
        end
    end
end

fprintf('\nDone.\n Copied TR cases: %d (pairs image+label)\n Copied TS cases: %d (images only)\n Skipped: %d\n', ...
    nTr, nTs, nSkipped);
