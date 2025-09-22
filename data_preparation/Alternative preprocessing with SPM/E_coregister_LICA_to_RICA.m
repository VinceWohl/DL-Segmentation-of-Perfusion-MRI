function [logTable, errList] = E_coregister_LICA_to_RICA(rootDir, opts)
% E_coregister_LICA_to_RICA
% Coregister ssLICA -> ssRICA per subject/visit and apply the transform
% to other LICA images (T1w_coreg, CBF_nativeSpace, PerfTerrMask).
%
% USAGE:
%   [logTable, errList] = E_coregister_LICA_to_RICA('D:\Data\anon_DATA_250919');
%
% INPUTS
%   rootDir  : string, root of your dataset (e.g. 'D:\Data\anon_DATA_250919')
%   opts     : optional struct with fields:
%              .groups   = {'DATA_HC','DATA_patients'}             % default
%              .visits   = {'First_visit','Second_visit','Third_visit'} % default
%              .spmPath  = 'D:\Code\DLSegPerf\spm'                  % if SPM not on path
%              .writeCSV = true                                     % write log CSV
%              .maskNN   = true                                     % NN for masks (binary)
%
% OUTPUTS
%   logTable : table with one row per attempted coregistration
%   errList  : cellstr of error strings (if any)
%
% Notes:
% - Estimation/Reslice use SPM defaults (cost_fun=nmi, sep=[4 2], tol=SPM default,
%   fwhm=[7 7], interp=4, wrap=[0 0 0], mask=0) for continuous images.
% - Masks are resliced with nearest-neighbour and rewritten as binary uint8.
% - CBF_nativeSpace is clamped to >= 0 (float32).
% - Result files end with the SAME filenames as before (no 'r' prefix).
%
% Fixed/Moved/Other patterns under each subject/visit/output:
%   FIXED  : task-AIR/ASL/ssRICA/ASL_fullcoreg/anon_meano_*RICA*_002_1.nii
%   MOVED  : task-AIR/ASL/ssLICA/ASL_fullcoreg/anon_meano_*LICA*_002_1.nii
%   OTHERs :   task-AIR/ASL/ssLICA/T1w_coreg/anon_rsub-XXX_T1w.nii
%              task-AIR/ASL/ssLICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii
%              task-AIR/ASL/ssLICA/PerfTerrMask/mask_LICA_manual_Corrected.nii

if nargin < 1 || isempty(rootDir)
    error('Please provide rootDir, e.g. D:\Data\anon_DATA_250919');
end
if nargin < 2, opts = struct; end
if ~isfield(opts,'groups'),   opts.groups   = {'DATA_HC','DATA_patients'}; end
if ~isfield(opts,'visits'),   opts.visits   = {'First_visit','Second_visit','Third_visit'}; end
if ~isfield(opts,'spmPath'),  opts.spmPath  = ''; end
if ~isfield(opts,'writeCSV'), opts.writeCSV = true; end
if ~isfield(opts,'maskNN'),   opts.maskNN   = true; end   % <- default: keep masks binary

% --- SPM setup
if ~isempty(opts.spmPath) && exist(fullfile(opts.spmPath,'spm.m'),'file')
    addpath(opts.spmPath);
end
if ~exist('spm','file')
    error('SPM not found on path. Set opts.spmPath or add SPM to MATLAB path.');
end
spm('defaults','FMRI');
spm_jobman('initcfg');

rows = {};
errList = {};

for g = 1:numel(opts.groups)
    groupName = opts.groups{g};
    groupDir  = fullfile(rootDir, groupName);
    if ~isfolder(groupDir), continue; end

    for v = 1:numel(opts.visits)
        visitName = opts.visits{v};
        visitDir  = fullfile(groupDir, visitName, 'output');
        if ~isfolder(visitDir), continue; end

        dSubs = dir(fullfile(visitDir,'sub-p*'));
        for s = 1:numel(dSubs)
            if ~dSubs(s).isdir, continue; end
            subDir = fullfile(visitDir, dSubs(s).name);

            % Build expected paths
            fixed = fullfile(subDir, 'task-AIR','ASL','ssRICA','ASL_fullcoreg', ...
                sprintf('anon_meano_%s_task-AIR_acq-epi_label-ssRICA_asl_002_1.nii', dSubs(s).name));

            moved = fullfile(subDir, 'task-AIR','ASL','ssLICA','ASL_fullcoreg', ...
                sprintf('anon_meano_%s_task-AIR_acq-epi_label-ssLICA_asl_002_1.nii', dSubs(s).name));

            otherCandidates = {
                fullfile(subDir,'task-AIR','ASL','ssLICA','T1w_coreg', ['anon_r' dSubs(s).name '_T1w.nii'])
                fullfile(subDir,'task-AIR','ASL','ssLICA','CBF_nativeSpace','CBF_3_BRmsk_CSF.nii')
                fullfile(subDir,'task-AIR','ASL','ssLICA','PerfTerrMask','mask_LICA_manual_Corrected.nii')
            };
            other = otherCandidates(cellfun(@(p) exist(p,'file')==2, otherCandidates));

            row = struct( ...
                Group=string(groupName), ...
                Visit=string(visitName), ...
                Subject=string(dSubs(s).name), ...
                FixedExists=exist(fixed,'file')==2, ...
                MovedExists=exist(moved,'file')==2, ...
                N_Other=numel(other), ...
                Status="pending", ...
                Message="" );

            try
                if ~(exist(fixed,'file')==2 && exist(moved,'file')==2)
                    row.Status  = "skipped";
                    row.Message = "Fixed or moved image missing";
                    rows{end+1} = row; %#ok<AGROW>
                    continue;
                end

                % --- Build SPM batch (estimate & reslice)
                matlabbatch = {};

                % Split mask vs non-mask if requested
                isMask = false(size(other));
                for i=1:numel(other)
                    oi = lower(other{i});
                    isMask(i) = contains(oi,'perfterrmask') || contains(oi,'mask_lica_manual');
                end
                other_nonmask = other(~isMask);
                other_mask    = other(isMask);

                % Job 1: moved + nonmask with default interp=4
                matlabbatch{1}.spm.spatial.coreg.estwrite.ref    = {fixed};
                matlabbatch{1}.spm.spatial.coreg.estwrite.source = {moved};
                matlabbatch{1}.spm.spatial.coreg.estwrite.other  = reshape(other_nonmask,[],1);

                matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
                matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep      = [4 2];
                matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol      = ...
                    [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
                matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm     = [7 7];

                matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = 4;   % B-spline (default)
                matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap   = [0 0 0];
                matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask   = 0;
                matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

                % Job 2: ONLY masks with NN (if present and NN desired)
                if opts.maskNN && ~isempty(other_mask)
                    matlabbatch{2} = matlabbatch{1};
                    matlabbatch{2}.spm.spatial.coreg.estwrite.other = reshape(other_mask,[],1);
                    matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.interp = 0; % nearest-neighbour
                end

                % Run
                spm_jobman('run', matlabbatch);

                % --- Replace originals with 'r*' files (no prefix end state)
                repl = [ {moved}; other(:) ];
                for k = 1:numel(repl)
                    orig = repl{k};
                    [p,f,e] = fileparts(orig);
                    rfile = fullfile(p, ['r' f e]);
                    if exist(rfile,'file')==2
                        delete(orig);
                        movefile(rfile, orig);
                    end
                end

                % --- Post-fix: enforce mask binary/uint8 and clamp CBF >= 0
                for k = 1:numel(repl)
                    fn = repl{k};
                    L  = lower(fn);

                    % PerfTerrMask -> binary uint8
                    if contains(L,'perfterrmask') || contains(L,'mask_lica_manual')
                        V = spm_vol(fn);
                        X = spm_read_vols(V);
                        Xb = X > 0.5;           % strict binary
                        V.dt    = [2 0];        % uint8
                        V.pinfo = [1;0;0];      % no scaling
                        spm_write_vol(V, uint8(Xb));
                    end

                    % CBF_nativeSpace -> clamp negatives to zero
                    if contains(L, fullfile('sslica','cbf_nativespace')) && contains(L,'cbf_3_brmsk_csf.nii')
                        V = spm_vol(fn);
                        X = spm_read_vols(V);
                        X(X < 0) = 0;
                        V.dt    = [16 0];       % float32
                        V.pinfo = [1;0;0];
                        spm_write_vol(V, single(X));
                    end
                end

                row.Status  = "ok";
                row.Message = "Coreg + replace + post-fix done";
            catch ME
                row.Status  = "error";
                row.Message = ME.message;
                errList{end+1} = sprintf('%s | %s | %s | %s', groupName, visitName, dSubs(s).name, ME.message); %#ok<AGROW>
            end

            rows{end+1} = row; %#ok<AGROW>
        end
    end
end

% Build table
if isempty(rows)
    logTable = table;
else
    logTable = struct2table([rows{:}]');
end

% Optional CSV
if opts.writeCSV
    ts = datestr(now,'yyyymmdd_HHMMSS');
    [fnPath,~,~] = fileparts(mfilename('fullpath'));
    outCSV = fullfile(fnPath, ['E_coregister_LICA_to_RICA_log_' ts '.csv']);
    try
        writetable(logTable, outCSV);
        fprintf('Log written: %s\n', outCSV);
    catch
        warning('Could not write CSV log.');
    end
end
end
