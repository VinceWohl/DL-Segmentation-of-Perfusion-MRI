% Author: VW
% Prepare Dataset001_PerfusionTerritories (nnU-Net raw)
% Inputs: CBF map per hemisphere as _0000, PerfTerrMask as label
% Splits taken from the Excel in final_quality_check

clear; clc;

%% -------- PATHS --------
src_root = 'D:\Data\anon_DATA_250919';
dst_root = 'D:\Data\nnUNet_raw\Dataset001_PerfusionTerritories';

% Target subfolders
dst_imagesTr = fullfile(dst_root, 'imagesTr');
dst_labelsTr = fullfile(dst_root, 'labelsTr');
dst_imagesTs = fullfile(dst_root, 'imagesTs');

cellfun(@(p) ~exist(p,'dir') && mkdir(p), {dst_imagesTr, dst_labelsTr, dst_imagesTs});

%% -------- SPLIT TABLE --------
xls_path = fullfile('D:\Code\DLSegPerf\data_preparation\Quality\data_completeness_report_20250820_171756.xlsx');

T = readtable(xls_path, 'Sheet', 'Data Completeness');

% Find the split column robustly (it contains this phrase in your file)
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
hemi_src = {'ssLICA','ssRICA'};   % left / right source subfolders
hemi_tag = {'L','R'};             % Z = L or R
maskName = {'mask_LICA_manual_Corrected.nii','mask_RICA_manual_Corrected.nii'};

%% -------- PROCESS --------
nCopiedTr = 0; nCopiedTs = 0; nSkipped = 0;

for i = 1:height(T)
    grp   = strtrim(string(T.Group{i}));          % 'DATA_HC' or 'DATA_patients'
    subj  = strtrim(string(T.Subject{i}));        % 'sub-p001'
    visit = strtrim(string(T.Visit{i}));          % 'First_visit', etc.

    % skip if split is 'x' or empty
    splitVal = string(T.(splitVarName)(i));
    if isempty(splitVal) || strcmpi(splitVal, "x")
        nSkipped = nSkipped + 1;
        fprintf('[SKIP row %d] %s | %s | %s (split=%s)\n', i, grp, subj, visit, splitVal);
        continue;
    end

    if ~isKey(visit_map, visit)
        nSkipped = nSkipped + 1;
        fprintf('[SKIP row %d] Unknown visit tag: %s | %s | %s\n', i, grp, subj, visit);
        continue;
    end

    subj_num = extractAfter(subj, 'sub-p');  % '001'
    vtag     = visit_map(visit);

    % Base source path to ASL
    asl_base = fullfile(src_root, grp, visit, 'output', subj, 'task-AIR', 'ASL');

    for h = 1:2
        hemi = hemi_src{h};
        Z    = hemi_tag{h};

        cbf_src  = fullfile(asl_base, hemi, 'CBF_nativeSpace', 'CBF_3_BRmsk_CSF.nii');
        mask_src = fullfile(asl_base, hemi, 'PerfTerrMask', maskName{h});

        if ~isfile(cbf_src) || ~isfile(mask_src)
            nSkipped = nSkipped + 1;
            fprintf('[MISS] %s | %s | %s | %s  (cbf:%d, mask:%d)\n', ...
                grp, subj, visit, hemi, isfile(cbf_src), isfile(mask_src));
            continue;
        end

        base = sprintf('PerfTerr%s-%s-%s', subj_num, vtag, Z);
        img_out = [base '_0000.nii'];
        lbl_out = [base '.nii'];

        switch string(splitVal)
            case "Training/Validation"
                copyfile(cbf_src,  fullfile(dst_imagesTr, img_out), 'f');
                copyfile(mask_src, fullfile(dst_labelsTr, lbl_out),  'f');
                nCopiedTr = nCopiedTr + 1;
                fprintf('[TR]  %s %s %s %s -> %s\n', grp, subj, visit, hemi, base);

            case "Testing"
                copyfile(cbf_src, fullfile(dst_imagesTs, img_out), 'f');
                % nnU-Net raw: do not place labels for test set
                nCopiedTs = nCopiedTs + 1;
                fprintf('[TS]  %s %s %s %s -> %s\n', grp, subj, visit, hemi, base);

            otherwise
                % Unknown split value -> skip safely
                nSkipped = nSkipped + 1;
                fprintf('[SKIP split] %s | %s | %s | %s (split=%s)\n', ...
                    grp, subj, visit, hemi, splitVal);
        end
    end
end

fprintf('\nDone.\n Copied TR pairs: %d\n Copied TS images: %d\n Skipped: %d\n', ...
    nCopiedTr, nCopiedTs, nSkipped);
