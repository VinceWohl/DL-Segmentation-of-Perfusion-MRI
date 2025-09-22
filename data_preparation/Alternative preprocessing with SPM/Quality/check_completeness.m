function [T, outfile] = check_completeness(rootDir, outfile)

% F_check_completeness  Build completeness table and write Excel.
%
% Usage:
%   [T, outfile] = check_completeness('D:\Data\anon_DATA_250919');  
%   [T, outfile] = check_completeness('D:\Data\anon_DATA', 'C:\path\to\file.xlsx');
%
% Outputs:
%   T        : table that gets written to Excel
%   outfile  : full path to the written Excel file

% -------- Args / defaults --------
if nargin < 1 || isempty(rootDir)
    error('Please provide rootDir, e.g., D:\Data\anon_DATA');
end
if nargin < 2 || isempty(outfile)
    [thisPath,~,~] = fileparts(mfilename('fullpath'));
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    outfile = fullfile(thisPath, sprintf('completeness_check_%s.xlsx', timestamp));
end

% -------- Config --------
hcSubjects = arrayfun(@(x) sprintf('sub-p%03d', x), 1:15, 'UniformOutput', false);
ptSubjects = arrayfun(@(x) sprintf('sub-p%03d', x), 16:23, 'UniformOutput', false);

groupInfo = struct();
groupInfo.HC.folder           = 'DATA_HC';
groupInfo.HC.subjects         = hcSubjects;
groupInfo.HC.visits           = {'First_visit','Second_visit','Third_visit'};
groupInfo.patients.folder     = 'DATA_patients';
groupInfo.patients.subjects   = ptSubjects;
groupInfo.patients.visits     = {'First_visit','Second_visit'};

hemis  = {'ssLICA','ssRICA'};  % left, right

baseCols = {'Group','Subject','Visit','Hemisphere'};
tags = {'ASL_meano','T1w_coreg','FLAIR_coreg','CBF_native','Mask_manual'};
fileCols = {};
for k = 1:numel(tags)
    tag = tags{k};
    fileCols = [fileCols, {sprintf('%s_DimX',tag), sprintf('%s_DimY',tag), sprintf('%s_DimZ',tag), ...
                           sprintf('%s_Min',tag),  sprintf('%s_Max',tag)}]; %#ok<AGROW>
end
headers = [baseCols, fileCols];

% -------- Build rows --------
rows = {};  % cell array

for gname = fieldnames(groupInfo)'
    G = gname{1};
    g = groupInfo.(G);
    for s = 1:numel(g.subjects)
        subj  = g.subjects{s};
        for v = 1:numel(g.visits)
            visit = g.visits{v};
            for h = 1:numel(hemis)
                hemi = hemis{h};   % 'ssLICA' or 'ssRICA'

                row = {G, subj, visit, hemi};
                baseFolder = fullfile(rootDir, g.folder, visit, 'output', subj);
                relpaths = expected_paths_for_hemi(hemi, subj);

                for e = 1:numel(relpaths)
                    fpath = fullfile(baseFolder, relpaths{e});
                    if isfile(fpath)
                        [dx,dy,dz,vmin,vmax] = get_nifti_stats(fpath);
                        row = [row, {dx,dy,dz,vmin,vmax}]; %#ok<AGROW>
                    else
                        row = [row, {'MISSING','MISSING','MISSING','MISSING','MISSING'}]; %#ok<AGROW>
                    end
                end

                rows = [rows; row]; %#ok<AGROW>
            end
        end
    end
end

% -------- Sort rows (Group, Subject, Visit, Hemisphere) --------
grpOrder = @(g) find(strcmp(g, {'HC','patients'}),1);
visOrder = @(v) find(strcmp(v, {'First_visit','Second_visit','Third_visit'}),1);
[~, idxSort] = sortrows( ...
    [ cellfun(grpOrder,  rows(:,1)), ...
      cellfun(@(s) sscanf(s,'sub-p%03d'), rows(:,2)), ...
      cellfun(visOrder,  rows(:,3)), ...
      cellfun(@(h) find(strcmp(h,hemis)), rows(:,4)) ] );
rows = rows(idxSort,:);

% -------- Write & format Excel --------
T = cell2table(rows, 'VariableNames', headers);
writetable(T, outfile, 'FileType','spreadsheet');
fprintf('Wrote completeness sheet: %s\n', outfile);

% Conditional formatting via Excel (optional)
try
    Excel = actxserver('Excel.Application');
    Excel.Visible = false;
    WB = Excel.Workbooks.Open(outfile);
    WS = WB.Worksheets.Item(1);

    UsedRange = WS.UsedRange;
    UsedRange.FormatConditions.Delete;

    xlCellValue = 1;   % XlFormatConditionType
    xlEqual     = 3;   % XlFormatConditionOperator
    fc = UsedRange.FormatConditions.Add(xlCellValue, xlEqual, '="MISSING"');
    fc.Interior.Color   = 255;   % red
    fc.Font.ColorIndex  = 2;     % white

    WB.Save;
    WB.Close(false);
    Excel.Quit;
    delete(Excel);
    fprintf('Applied red highlighting for missing cells.\n');
catch ME
    warning('Excel formatting step skipped: %s', ME.message);
end
end

% ===== Helper functions =====
function relpaths = expected_paths_for_hemi(hemi, subj)
    % Return relative paths (ordered as tags) for given hemisphere.
    % hemi is 'ssLICA' or 'ssRICA'
    if strcmpi(hemi,'ssLICA')
        maskName = 'LICA';
        labelStr = 'ssLICA';
    else
        maskName = 'RICA';
        labelStr = 'ssRICA';
    end
    relpaths = {
        sprintf('task-AIR/ASL/%s/ASL_fullcoreg/anon_meano_%s_task-AIR_acq-epi_label-%s_asl_002_1.nii', hemi, subj, labelStr)
        sprintf('task-AIR/ASL/%s/T1w_coreg/anon_r%s_T1w.nii',     hemi, subj)
        sprintf('task-AIR/ASL/%s/FLAIR_coreg/anon_r%s_FLAIR.nii', hemi, subj)
        sprintf('task-AIR/ASL/%s/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii', hemi)
        sprintf('task-AIR/ASL/%s/PerfTerrMask/mask_%s_manual_Corrected.nii', hemi, maskName)
    };
end

function [dimx, dimy, dimz, vmin, vmax] = get_nifti_stats(fpath)
    nii = niftiread(fpath);
    if ~isfloat(nii); nii = double(nii); end
    sz = size(nii); sz = [sz, ones(1, 3-numel(sz))];
    dimx = sz(1); dimy = sz(2); dimz = sz(3);
    vmin = min(nii(:));
    vmax = max(nii(:));
end
