function logTable = F_coregister_FLAIR_to_ASLfullcoreg(rootDir, opts)
% F_coregister_FLAIR_to_ASLfullcoreg
% Coregister the First_visit FLAIR of each subject to each visit's
% ssLICA/ASL_fullcoreg and ssRICA/ASL_fullcoreg, saving results as:
%   task-AIR/ASL/ssLICA/FLAIR_coreg/anon_rsub-pXXX_FLAIR.nii
%   task-AIR/ASL/ssRICA/FLAIR_coreg/anon_rsub-pXXX_FLAIR.nii
%
% Usage:
%   logTable = F_coregister_FLAIR_to_ASLfullcoreg('D:\Data\anon_DATA_250919', ...
%               struct('spmPath','D:\Code\DLSegPerf\spm'));
%
% Options (all optional):
%   opts.groups = {'DATA_HC','DATA_patients'}               % default
%   opts.visits = {'First_visit','Second_visit','Third_visit'}  % default
%   opts.spmPath = 'D:\Code\DLSegPerf\spm'                   % if SPM not on path
%
% Notes:
% - SPM defaults are used: cost_fun=nmi, sep=[4 2], tol=default, fwhm=[7 7],
%   interp=4 (B-spline), wrap=[0 0 0], mask=0, prefix='r'.
% - The SPM-produced r* file is moved/renamed to the target FLAIR_coreg folder.

if nargin < 1 || isempty(rootDir)
    error('Please provide rootDir, e.g. D:\Data\anon_DATA_250919');
end
if nargin < 2, opts = struct; end
if ~isfield(opts,'groups'), opts.groups = {'DATA_HC','DATA_patients'}; end
if ~isfield(opts,'visits'), opts.visits = {'First_visit','Second_visit','Third_visit'}; end
if ~isfield(opts,'spmPath'), opts.spmPath = ''; end

% --- SPM setup
if ~isempty(opts.spmPath) && exist(fullfile(opts.spmPath,'spm.m'),'file')
    addpath(opts.spmPath);
end
if ~exist('spm','file')
    error('SPM not found on path. Set opts.spmPath or add SPM to MATLAB path.');
end
spm('defaults','FMRI'); spm_jobman('initcfg');

rows = {};

for g = 1:numel(opts.groups)
    groupName = opts.groups{g};
    groupDir  = fullfile(rootDir, groupName);

    % Subjects enumerated from First_visit (moving FLAIR lives here)
    firstVisitDir = fullfile(groupDir, 'First_visit', 'output');
    if ~isfolder(firstVisitDir), continue; end

    dSubs = dir(fullfile(firstVisitDir,'sub-p*'));
    for s = 1:numel(dSubs)
        if ~dSubs(s).isdir, continue; end
        subj = dSubs(s).name;

        % Moving image (always First_visit FLAIR)
        moving = fullfile(firstVisitDir, subj, 'FLAIR', ['anon_' subj '_FLAIR.nii']);
        if exist(moving,'file') ~= 2
            rows{end+1} = mkRow(groupName, subj, 'First_visit', 'LICA', 'skipped', 'Missing First_visit FLAIR', ''); %#ok<AGROW>
            rows{end+1} = mkRow(groupName, subj, 'First_visit', 'RICA', 'skipped', 'Missing First_visit FLAIR', '');
            continue;
        end
        [movingPath, movingName, movingExt] = fileparts(moving);
        rtemp = fullfile(movingPath, ['r' movingName movingExt]);  % SPM writes here

        % Process each visit that exists
        for v = 1:numel(opts.visits)
            visitName = opts.visits{v};
            visitSubDir = fullfile(groupDir, visitName, 'output', subj);
            if ~isfolder(visitSubDir), continue; end

            % ---- LICA target
            fixedLICA = fullfile(visitSubDir, 'task-AIR','ASL','ssLICA','ASL_fullcoreg', ...
                sprintf('anon_meano_%s_task-AIR_acq-epi_label-ssLICA_asl_002_1.nii', subj));
            destLICAFolder = fullfile(visitSubDir, 'task-AIR','ASL','ssLICA','FLAIR_coreg');
            destLICA = fullfile(destLICAFolder, ['anon_r' subj '_FLAIR.nii']);

            if exist(fixedLICA,'file')==2
                try
                    coreg_and_move(moving, fixedLICA, rtemp, destLICAFolder, destLICA);
                    rows{end+1} = mkRow(groupName, subj, visitName, 'LICA', 'ok', '', destLICA); %#ok<AGROW>
                catch ME
                    rows{end+1} = mkRow(groupName, subj, visitName, 'LICA', 'error', ME.message, ''); %#ok<AGROW>
                end
            else
                rows{end+1} = mkRow(groupName, subj, visitName, 'LICA', 'skipped', 'Fixed LICA ASL_fullcoreg missing', '');
            end

            % ---- RICA target
            fixedRICA = fullfile(visitSubDir, 'task-AIR','ASL','ssRICA','ASL_fullcoreg', ...
                sprintf('anon_meano_%s_task-AIR_acq-epi_label-ssRICA_asl_002_1.nii', subj));
            destRICAFolder = fullfile(visitSubDir, 'task-AIR','ASL','ssRICA','FLAIR_coreg');
            destRICA = fullfile(destRICAFolder, ['anon_r' subj '_FLAIR.nii']);

            if exist(fixedRICA,'file')==2
                try
                    coreg_and_move(moving, fixedRICA, rtemp, destRICAFolder, destRICA);
                    rows{end+1} = mkRow(groupName, subj, visitName, 'RICA', 'ok', '', destRICA); %#ok<AGROW>
                catch ME
                    rows{end+1} = mkRow(groupName, subj, visitName, 'RICA', 'error', ME.message, ''); %#ok<AGROW>
                end
            else
                rows{end+1} = mkRow(groupName, subj, visitName, 'RICA', 'skipped', 'Fixed RICA ASL_fullcoreg missing', '');
            end
        end
    end
end

% Assemble log table
if isempty(rows), logTable = table; else, logTable = struct2table([rows{:}]'); end
end

% ---------- helpers ----------
function coreg_and_move(moving, fixed, rtemp, destFolder, destFile)
% Run SPM Coregister: Estimate & Reslice (moving -> fixed), then move/rename result
matlabbatch = {};
matlabbatch{1}.spm.spatial.coreg.estwrite.ref    = {fixed};
matlabbatch{1}.spm.spatial.coreg.estwrite.source = {moving};
matlabbatch{1}.spm.spatial.coreg.estwrite.other  = {''};

% Defaults
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep      = [4 2];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol      = ...
  [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm     = [7 7];

matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = 4;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap   = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask   = 0;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

% Run job
spm_jobman('run', matlabbatch);

% Move/rename SPM output
if ~exist(rtemp,'file')
    error('Expected resliced file not found: %s', rtemp);
end
if ~exist(destFolder,'dir'), mkdir(destFolder); end
if exist(destFile,'file'), delete(destFile); end
movefile(rtemp, destFile);
end

function row = mkRow(groupName, subj, visitName, hemi, status, msg, dest)
row = struct( ...
    Group=string(groupName), ...
    Subject=string(subj), ...
    Visit=string(visitName), ...
    Target=string(hemi), ...
    Status=string(status), ...
    Message=string(msg), ...
    Output=string(dest) );
end
