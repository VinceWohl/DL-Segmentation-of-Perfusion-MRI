function [logTable, errList] = E_coregister_LICA_to_RICA(rootDir, opts)
% E_coregister_LICA_to_RICA
% 1) Reset origins of ssLICA/ASL_fullcoreg and ssRICA/ASL_fullcoreg
%    and apply the *same per-hemisphere shift* to their other task-AIR images.
% 2) Coregister LICA -> RICA (Estimate & Reslice), apply transform to LICA
%    other images.
% 3) Keep PerfTerrMask binary (NN + uint8). Clamp CBF_nativeSpace >= 0.
% 4) Preserve original filenames (no 'r' prefix in final state).
%
% USAGE:
%   [logTable, errList] = E_coregister_LICA_to_RICA('D:\Data\anon_DATA_250919');
%
% OPTIONS (all optional):
%   opts.groups   = {'DATA_HC','DATA_patients'}
%   opts.visits   = {'First_visit','Second_visit','Third_visit'}
%   opts.spmPath  = 'D:\Code\DLSegPerf\spm'
%   opts.writeCSV = true
%   opts.maskNN   = true     % NN for masks (binary); default true

if nargin < 1 || isempty(rootDir)
    error('Please provide rootDir, e.g. D:\Data\anon_DATA_250919');
end
if nargin < 2, opts = struct; end
if ~isfield(opts,'groups'),   opts.groups   = {'DATA_HC','DATA_patients'}; end
if ~isfield(opts,'visits'),   opts.visits   = {'First_visit','Second_visit','Third_visit'}; end
if ~isfield(opts,'spmPath'),  opts.spmPath  = ''; end
if ~isfield(opts,'writeCSV'), opts.writeCSV = true; end
if ~isfield(opts,'maskNN'),   opts.maskNN   = true; end   % keep masks binary by default

% --- Check SPM and reset_origin
if ~isempty(opts.spmPath) && exist(fullfile(opts.spmPath,'spm.m'),'file'), addpath(opts.spmPath); end
if ~exist('spm','file'), error('SPM not found on path. Set opts.spmPath or add SPM to MATLAB path.'); end
if ~exist('reset_origin','file'), error('reset_origin.m not found on path. Please add it.'); end
spm('defaults','FMRI'); spm_jobman('initcfg');

rows = {}; errList = {};

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

            % --- Build expected paths (LICA/RICA)
            fixed_RICA = fullfile(subDir,'task-AIR','ASL','ssRICA','ASL_fullcoreg', ...
                sprintf('anon_meano_%s_task-AIR_acq-epi_label-ssRICA_asl_002_1.nii', dSubs(s).name));
            moved_LICA = fullfile(subDir,'task-AIR','ASL','ssLICA','ASL_fullcoreg', ...
                sprintf('anon_meano_%s_task-AIR_acq-epi_label-ssLICA_asl_002_1.nii', dSubs(s).name));

            LICA_others = {
                fullfile(subDir,'task-AIR','ASL','ssLICA','T1w_coreg', ['anon_r' dSubs(s).name '_T1w.nii'])
                fullfile(subDir,'task-AIR','ASL','ssLICA','CBF_nativeSpace','CBF_3_BRmsk_CSF.nii')
                fullfile(subDir,'task-AIR','ASL','ssLICA','PerfTerrMask','mask_LICA_manual_Corrected.nii')
            };
            LICA_others = LICA_others(cellfun(@(p) exist(p,'file')==2, LICA_others));

            RICA_others = {
                fullfile(subDir,'task-AIR','ASL','ssRICA','T1w_coreg', ['anon_r' dSubs(s).name '_T1w.nii'])
                fullfile(subDir,'task-AIR','ASL','ssRICA','CBF_nativeSpace','CBF_3_BRmsk_CSF.nii')
                fullfile(subDir,'task-AIR','ASL','ssRICA','PerfTerrMask','mask_RICA_manual_Corrected.nii')
            };
            RICA_others = RICA_others(cellfun(@(p) exist(p,'file')==2, RICA_others));

            row = struct( ...
                Group=string(groupName), Visit=string(visitName), Subject=string(dSubs(s).name), ...
                FixedExists=exist(fixed_RICA,'file')==2, MovedExists=exist(moved_LICA,'file')==2, ...
                N_Other_LICA=numel(LICA_others), N_Other_RICA=numel(RICA_others), ...
                Status="pending", Message="" );

            try
                if ~(exist(fixed_RICA,'file')==2 && exist(moved_LICA,'file')==2)
                    row.Status="skipped"; row.Message="Fixed or moved image missing";
                    rows{end+1} = row; %#ok<AGROW>
                    continue;
                end

                % ==========================================================
                % 1) RESET ORIGINS (and apply same shifts to other images)
                % ==========================================================
                % LICA: compute shift from LICA ASL_fullcoreg, apply to LICA ASL + LICA others
                shift_LICA = local_get_shift(moved_LICA);
                local_reset_one(moved_LICA, shift_LICA);  % ASL_fullcoreg LICA
                cellfun(@(p) local_reset_one(p, shift_LICA), LICA_others);

                % RICA: compute shift from RICA ASL_fullcoreg, apply to RICA ASL + RICA others
                shift_RICA = local_get_shift(fixed_RICA);
                local_reset_one(fixed_RICA, shift_RICA);  % ASL_fullcoreg RICA
                cellfun(@(p) local_reset_one(p, shift_RICA), RICA_others);

                % ==========================================================
                % 2) COREGISTER (Estimate & Reslice): LICA -> RICA
                %    apply transform to LICA OTHERS only (as before)
                % ==========================================================
                other = LICA_others;  % transform is for LICA branch
                isMask = false(size(other));
                for i=1:numel(other)
                    oi = lower(other{i});
                    isMask(i) = contains(oi,'perfterrmask') || contains(oi,'mask_lica_manual');
                end
                other_nonmask = other(~isMask);
                other_mask    = other(isMask);

                matlabbatch = {};
                % Job 1: moved + non-mask (default B-spline)
                matlabbatch{1}.spm.spatial.coreg.estwrite.ref    = {fixed_RICA};
                matlabbatch{1}.spm.spatial.coreg.estwrite.source = {moved_LICA};
                matlabbatch{1}.spm.spatial.coreg.estwrite.other  = reshape(other_nonmask,[],1);

                matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
                matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep      = [4 2];
                matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol      = ...
                    [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
                matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm     = [7 7];

                matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp = 4;
                matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap   = [0 0 0];
                matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask   = 0;
                matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

                % Job 2: masks with NN (if any)
                if opts.maskNN && ~isempty(other_mask)
                    matlabbatch{2} = matlabbatch{1};
                    matlabbatch{2}.spm.spatial.coreg.estwrite.other = reshape(other_mask,[],1);
                    matlabbatch{2}.spm.spatial.coreg.estwrite.roptions.interp = 0; % NN
                end

                spm_jobman('run', matlabbatch);

                % ==========================================================
                % 3) Replace originals with 'r*' files (no prefix end state)
                % ==========================================================
                repl = [ {moved_LICA}; other(:) ];
                for k = 1:numel(repl)
                    orig = repl{k};
                    [p,f,e] = fileparts(orig);
                    rfile = fullfile(p, ['r' f e]);
                    if exist(rfile,'file')==2
                        delete(orig);
                        movefile(rfile, orig);
                    end
                end

                % ==========================================================
                % 4) Post-fix: keep masks binary/uint8, clamp CBF >= 0
                % ==========================================================
                for k = 1:numel(repl)
                    fn = repl{k}; L = lower(fn);

                    % PerfTerrMask -> binary uint8
                    if contains(L,'perfterrmask') || contains(L,'mask_lica_manual')
                        V = spm_vol(fn); X = spm_read_vols(V);
                        Xb = X > 0.5;  V.dt = [2 0]; V.pinfo = [1;0;0];
                        spm_write_vol(V, uint8(Xb));
                    end

                    % CBF_nativeSpace -> clamp negatives (LICA branch)
                    if contains(L, fullfile('sslica','cbf_nativespace')) && contains(L,'cbf_3_brmsk_csf.nii')
                        V = spm_vol(fn); X = spm_read_vols(V);
                        X(X < 0) = 0;  V.dt = [16 0]; V.pinfo = [1;0;0];
                        spm_write_vol(V, single(X));
                    end
                end

                row.Status="ok"; row.Message="Reset origins + coreg + post-fix done";
            catch ME
                row.Status="error"; row.Message=ME.message;
                errList{end+1} = sprintf('%s | %s | %s | %s', groupName, visitName, dSubs(s).name, ME.message); %#ok<AGROW>
            end

            rows{end+1} = row; %#ok<AGROW>
        end
    end
end

% Build table
if isempty(rows), logTable = table; else, logTable = struct2table([rows{:}]'); end

% Optional CSV
if opts.writeCSV
    ts = datestr(now,'yyyymmdd_HHMMSS');
    [fnPath,~,~] = fileparts(mfilename('fullpath'));
    outCSV = fullfile(fnPath, ['E_coregister_LICA_to_RICA_log_' ts '.csv']);
    try, writetable(logTable, outCSV); fprintf('Log written: %s\n', outCSV);
    catch, warning('Could not write CSV log.'); end
end
end

% ===== Helpers =============================================================
function shiftOrigin = local_get_shift(niiPath)
% replicate reset_origin's get_shiftOrigin_auto for a single file
V = spm_vol(niiPath);
T = V.mat; dim = V.dim;
vCenter = (dim + 1) / 2;
wCenter = T * [vCenter, 1]';
% Special-case upward shift for large-FOV anatomicals (kept as in your helper)
if contains(niiPath,'FLAIR') || contains(niiPath,'T1w')
    wCenter(3) = wCenter(3) + 25;
end
shiftOrigin.right   = -wCenter(1);
shiftOrigin.forward = -wCenter(2);
shiftOrigin.up      = -wCenter(3);
end

function local_reset_one(niiPath, shiftOrigin)
% call reset_origin(filepath, filename, shiftOrigin)
[p, f, e] = fileparts(niiPath);
reset_origin(p, [f e], shiftOrigin);   % uses prefix='' and reslices in place
end
