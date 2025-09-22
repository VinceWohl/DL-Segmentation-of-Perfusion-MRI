% Author: VW
% Prepare Dataset001_PerfusionTerritories (nnU-Net raw) with TWO channels:
%   _0000 = CBF (CBF_3_BRmsk_CSF.nii)
%   _0001 = T1w (prefer hemisphere-specific T1w_coreg; fallback to native T1w)
% Labels: PerfTerrMask
% NOTE: Does NOT rebuild dataset.json

clear; clc;

%% -------- SETTINGS --------
prefer_coreg_T1w = true;   % prefer hemisphere-specific T1w_coreg (recommended)
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

% Preserve headers (so names match exactly what you saw in Excel)
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
    % Use char everywhere to avoid string-array gotchas
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

    subj   = char(subjS);                  % char, e.g. 'sub-p001'
    subj_num = extractAfter(subj, 'sub-p');% '001'
    vtag     = visit_map(visit);

    % Base source paths
    subj_base = fullfile(src_root, grp, visit, 'output', subj);   % char
    asl_base  = fullfile(subj_base, 'task-AIR', 'ASL');           % char

    for h = 1:2
        hemi = hemi_src{h};
        Z    = hemi_tag{h};

        % ---- inputs/label ----
        cbf_src  = fullfile(asl_base, hemi, 'CBF_nativeSpace', 'CBF_3_BRmsk_CSF.nii');
        mask_src = fullfile(asl_base, hemi, 'PerfTerrMask',    maskName{h});

        % T1w candidates (coreg first if preferred)
        t1_coreg_name  = sprintf('anon_r%s_T1w.nii', subj);   % anon_rsub-p001_T1w.nii
        t1_native_name = sprintf('anon_%s_T1w.nii',   subj);  % anon_sub-p001_T1w.nii
        t1_coreg_path  = fullfile(asl_base, hemi, 'T1w_coreg', t1_coreg_name);
        t1_native_path = fullfile(subj_base, 'T1w', t1_native_name);

        if prefer_coreg_T1w
            if isfile(t1_coreg_path)
                t1_src = t1_coreg_path;
                used_fallback = false;
            elseif isfile(t1_native_path)
                t1_src = t1_native_path;
                used_fallback = true;
            else
                t1_src = '';
            end
        else
            if isfile(t1_native_path)
                t1_src = t1_native_path;
                used_fallback = false;
            elseif isfile(t1_coreg_path)
                t1_src = t1_coreg_path;
                used_fallback = true;
            else
                t1_src = '';
            end
        end

        % ---- check all required inputs exist ----
        if ~isfile(cbf_src) || ~isfile(mask_src) || isempty(t1_src)
            nSkipped = nSkipped + 1;
            if verbose
                fprintf('[MISS] %s | %s | %s | %s  (cbf:%d, t1w:%d, mask:%d)\n', ...
                    grp, subj, visit, hemi, isfile(cbf_src), ~isempty(t1_src), isfile(mask_src));
            end
            continue;
        end
        if verbose && exist('used_fallback','var') && used_fallback
            fprintf('[WARN Fallback T1w] Using native T1w for %s %s %s %s\n', grp, subj, visit, hemi);
        end

        % ---- destinations ----
        base    = sprintf('PerfTerr%s-%s-%s', subj_num, vtag, Z);
        img000  = [base '_0000.nii'];    % CBF
        img001  = [base '_0001.nii'];    % T1w
        lbl_out = [base '.nii'];         % label

        switch char(splitVal)
            case 'Training/Validation'
                copyfile(cbf_src, fullfile(dst_imagesTr, img000), 'f');
                copyfile(t1_src,  fullfile(dst_imagesTr, img001), 'f');
                copyfile(mask_src,fullfile(dst_labelsTr, lbl_out), 'f');
                nTr = nTr + 1;
                if verbose, fprintf('[TR]  %s %s %s %s -> %s\n', grp, subj, visit, hemi, base); end

            case 'Testing'
                copyfile(cbf_src, fullfile(dst_imagesTs, img000), 'f');
                copyfile(t1_src,  fullfile(dst_imagesTs, img001), 'f');
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