% Author: VW
% Halve Dataset001_PerfusionTerritories by KEEPING the hemisphere listed
% in your table for each subject/visit and DELETING the opposite one.
% Works with ANY number of image channels (_0000, _0001, _0002, ...).
%
% Affects: imagesTr and labelsTr
% File patterns assumed:
%   images: PerfTerr###-v#-[L|R]_<channel>.nii   (e.g., _0000.nii, _0001.nii, _0002.nii)
%   labels: PerfTerr###-v#-[L|R].nii             (single label per hemisphere)
%
% Tip: set dryRun=true to preview.

clear; clc;

% ---- PATHS ----
basePath = 'C:\Users\Vincent Wohlfarth\Data\nnUNet_raw\Dataset001_PerfusionTerritories\';
imgPath  = fullfile(basePath, 'imagesTr');
labPath  = fullfile(basePath, 'labelsTr');

% Safety toggle (preview only)
dryRun = false;

% ---- KEEP PLAN from your table (ssLICA = 'L', ssRICA = 'R') ----
keepPairs = { ...
    'PerfTerr001-v1','R';  'PerfTerr001-v2','L';  'PerfTerr001-v3','R'; ...
    'PerfTerr002-v1','L';  'PerfTerr002-v2','R';  'PerfTerr002-v3','L'; ...
    'PerfTerr003-v1','R';  'PerfTerr003-v2','L';  'PerfTerr003-v3','R'; ...
    'PerfTerr004-v1','L';  'PerfTerr004-v2','R';  'PerfTerr004-v3','L'; ...
    'PerfTerr005-v1','R';  'PerfTerr005-v2','L';  'PerfTerr005-v3','R'; ...
    'PerfTerr006-v1','L';  'PerfTerr006-v2','R';  'PerfTerr006-v3','L'; ...
    'PerfTerr007-v1','R';  'PerfTerr007-v3','L'; ...  % v2 missing
    'PerfTerr008-v1','R';  'PerfTerr008-v2','L';  'PerfTerr008-v3','R'; ...
    'PerfTerr009-v1','L';  'PerfTerr009-v2','R';  'PerfTerr009-v3','L'; ...
    'PerfTerr010-v1','R';  'PerfTerr010-v2','L';  'PerfTerr010-v3','R'; ...
    'PerfTerr011-v1','L';  'PerfTerr011-v2','R';  'PerfTerr011-v3','L'; ...
    'PerfTerr012-v1','R';  'PerfTerr012-v2','L';  'PerfTerr012-v3','R'; ...
    'PerfTerr013-v1','L';  'PerfTerr013-v2','R';  'PerfTerr013-v3','L'  ...
};

% ---- PROCESS ----
deleted = {};
kept    = {};
for k = 1:size(keepPairs,1)
    prefix   = keepPairs{k,1};            % e.g., 'PerfTerr001-v1'
    keepSide = keepPairs{k,2};            % 'L' or 'R'
    delSide  = ternary(keepSide=='L', 'R', 'L');

    % Delete ALL channels of the opposite side (channel-agnostic via wildcard)
    delImgs = dir(fullfile(imgPath, sprintf('%s-%s_*.nii', prefix, delSide)));
    delLabs = dir(fullfile(labPath, sprintf('%s-%s.nii',  prefix, delSide)));

    % (Optional) log channels that will be kept (useful sanity check)
    keepImgs = dir(fullfile(imgPath, sprintf('%s-%s_*.nii', prefix, keepSide)));
    kept{end+1,1} = struct('case',prefix,'side',keepSide,'nChannels',numel(keepImgs)); %#ok<SAGROW>

    for d = 1:numel(delImgs)
        f = fullfile(delImgs(d).folder, delImgs(d).name);
        if dryRun, fprintf('[dry-run] delete %s\n', f);
        else, delete(f); fprintf('Deleted: %s\n', f);
        end
        deleted{end+1,1} = f; %#ok<AGROW>
    end
    for d = 1:numel(delLabs)
        f = fullfile(delLabs(d).folder, delLabs(d).name);
        if dryRun, fprintf('[dry-run] delete %s\n', f);
        else, delete(f); fprintf('Deleted: %s\n', f);
        end
        deleted{end+1,1} = f; %#ok<AGROW>
    end

    if isempty(delImgs) && isempty(delLabs)
        fprintf('Note: nothing to delete for %s (maybe already pruned / missing).\n', prefix);
    else
        fprintf('Kept %sICA for %s (%d ch), deleted %sICA.\n', ...
            keepSide, prefix, numel(keepImgs), delSide);
    end
end

% ---- SUMMARY ----
fprintf('\nSummary:\n');
fprintf('Cases processed: %d\n', size(keepPairs,1));
fprintf('Total files deleted: %d\n', numel(deleted));

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
