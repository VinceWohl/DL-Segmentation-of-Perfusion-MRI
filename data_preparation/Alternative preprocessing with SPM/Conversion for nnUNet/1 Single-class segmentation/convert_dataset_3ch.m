% Author: VW
% Prepare Dataset001_PerfusionTerritories (nnU-Net raw) with THREE channels:
%   _0000 = CBF (CBF_3_BRmsk_CSF.nii)
%   _0001 = T1w (prefer hemisphere-specific T1w_coreg; fallback to native T1w)
%   _0002 = FLAIR (prefer hemisphere-specific FLAIR_coreg; fallback to native FLAIR)
% Labels: PerfTerrMask
% NOTE: This script does NOT rebuild dataset.json.

clear; clc;

%% -------- SETTINGS --------
prefer_coreg_T1w   = true;     % prefer hemisphere-specific coreg T1
prefer_coreg_FLAIR = true;     % prefer hemisphere-specific coreg FLAIR
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

% keep original column headers
T = readtable(xls_path, 'Sheet', 'Data Completeness', 'VariableNamingRule', 'preserve');

% find split column robustly
splitColIdx = find(contains(T.Properties.VariableNames,'Training','IgnoreCase',true) & ...
                   contains(T.Properties.VariableNames,'Split',   'IgnoreCase',true), 1);
if isempty(splitColIdx)
    error('Could not find the "Training/Validation + Testing Split" column in the Excel.');
end
splitVarName = T.Properties.VariableNames{splitColIdx};

%% -------- VISIT MAP --------
visit_map = containers.Map({'First_visit','Second_visit','Third_visit'}, {'v1','v2','v3'});

%% -------- HEMISPHERES (source folder names & tags) --------
hemi_src  = {'ssLICA','ssRICA'};                  % left / right source subfolders
hemi_tag  = {'L','R'};                            % Z = L or R
maskName  = {'mask_LICA_manual_Corrected.nii','mask_RICA_manual_Corrected.nii'};

%% -------- PROCESS --------
nTr = 0; nTs = 0; nSkipped = 0;

for i = 1:height(T)
    grp     = char(strtrim(string(T.("Group"){i})));     % 'DATA_HC' or 'DATA_patients'
    subjStr = strtrim(string(T.("Subject"){i}));         % 'sub-p001'
    visit   = char(strtrim(string(T.("Visit"){i})));     % 'First_visit', etc.
    splitVal= strtrim(string(T.(splitVarName)(i)));

    if strlength(splitVal)==0 || strcmpi(splitVal,"x")
        nSkipped = nSkipped + 1;
        if verbose, fprintf('[SKIP row %d] %s | %s | %s (split=%s)\n', i, grp, subjStr, visit, splitVal); end
        continue;
    end
    if ~isKey(visit_map, visit)
        nSkipped = nSkipped + 1;
        if verbose, fprintf('[SKIP row %d] Unknown visit tag: %s | %s | %s\n', i, grp, subjStr, visit); end
        continue;
    end

    subj     = char(subjStr);                % 'sub-p001'
    subj_num = extractAfter(subj,'sub-p');   % '001'
    vtag     = visit_map(visit);

    subj_base = fullfile(src_root, grp, visit, 'output', subj);
    asl_base  = fullfile(subj_base, 'task-AIR', 'ASL');

    for h = 1:2
        hemi = hemi_src{h};
        Z    = hemi_tag{h};

        % ----- Channel 0: CBF -----
        cbf_src  = fullfile(asl_base, hemi, 'CBF_nativeSpace', 'CBF_3_BRmsk_CSF.nii');

        % ----- Label -----
        mask_src = fullfile(asl_base, hemi, 'PerfTerrMask', maskName{h});

        % ----- Channel 1: T1w (coreg -> native fallback) -----
        t1_coreg = fullfile(asl_base, hemi, 'T1w_coreg',  sprintf('anon_r%s_T1w.nii',  subj));
        t1_native= fullfile(subj_base,           'T1w',    sprintf('anon_%s_T1w.nii',  subj));
        used_fb_t1 = false;
        if prefer_coreg_T1w
            if isfile(t1_coreg),  t1_src = t1_coreg;
            elseif isfile(t1_native), t1_src = t1_native; used_fb_t1 = true;
            else, t1_src = ''; end
        else
            if isfile(t1_native), t1_src = t1_native;
            elseif isfile(t1_coreg), t1_src = t1_coreg; used_fb_t1 = true;
            else, t1_src = ''; end
        end

        % ----- Channel 2: FLAIR (coreg -> native fallback) -----
        fl_coreg = fullfile(asl_base, hemi, 'FLAIR_coreg', sprintf('anon_r%s_FLAIR.nii', subj));
        fl_native= fullfile(subj_base,          'FLAIR',    sprintf('anon_%s_FLAIR.nii', subj));
        used_fb_fl = false;
        if prefer_coreg_FLAIR
            if isfile(fl_coreg),  fl_src = fl_coreg;
            elseif isfile(fl_native), fl_src = fl_native; used_fb_fl = true;
            else, fl_src = ''; end
        else
            if isfile(fl_native), fl_src = fl_native;
            elseif isfile(fl_coreg), fl_src = fl_coreg; used_fb_fl = true;
            else, fl_src = ''; end
        end

        % ----- sanity checks -----
        if ~isfile(cbf_src) || isempty(t1_src) || isempty(fl_src) || ~isfile(mask_src)
            nSkipped = nSkipped + 1;
            if verbose
                fprintf('[MISS] %s | %s | %s | %s (cbf:%d, t1:%d, flair:%d, mask:%d)\n', ...
                    grp, subj, visit, hemi, isfile(cbf_src), ~isempty(t1_src), ~isempty(fl_src), isfile(mask_src));
            end
            continue;
        end
        if verbose
            if used_fb_t1, fprintf('[WARN] Using native T1w fallback: %s %s %s %s\n', grp, subj, visit, hemi); end
            if used_fb_fl, fprintf('[WARN] Using native FLAIR fallback: %s %s %s %s\n', grp, subj, visit, hemi); end
        end

        % ----- destinations -----
        base   = sprintf('PerfTerr%s-%s-%s', subj_num, vtag, Z);
        img000 = [base '_0000.nii'];   % CBF
        img001 = [base '_0001.nii'];   % T1w
        img002 = [base '_0002.nii'];   % FLAIR
        lbl    = [base '.nii'];

        switch char(splitVal)
            case 'Training/Validation'
                copyfile(cbf_src, fullfile(dst_imagesTr, img000), 'f');
                copyfile(t1_src,  fullfile(dst_imagesTr, img001), 'f');
                copyfile(fl_src,  fullfile(dst_imagesTr, img002), 'f');
                copyfile(mask_src,fullfile(dst_labelsTr, lbl),    'f');
                nTr = nTr + 1;
                if verbose, fprintf('[TR] %s %s %s %s -> %s\n', grp, subj, visit, hemi, base); end

            case 'Testing'
                copyfile(cbf_src, fullfile(dst_imagesTs, img000), 'f');
                copyfile(t1_src,  fullfile(dst_imagesTs, img001), 'f');
                copyfile(fl_src,  fullfile(dst_imagesTs, img002), 'f');
                % no labels for test set
                nTs = nTs + 1;
                if verbose, fprintf('[TS] %s %s %s %s -> %s\n', grp, subj, visit, hemi, base); end

            otherwise
                nSkipped = nSkipped + 1;
                if verbose, fprintf('[SKIP split] %s | %s | %s | %s (split=%s)\n', grp, subj, visit, hemi, splitVal); end
        end
    end
end

fprintf('\nDone.\nCopied TR cases: %d (image triplets + label)\nCopied TS cases: %d (image triplets only)\nSkipped: %d\n', ...
    nTr, nTs, nSkipped);
