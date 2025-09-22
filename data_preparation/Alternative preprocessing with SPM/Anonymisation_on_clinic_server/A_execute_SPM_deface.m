function A_execute_SPM_deface(source_root, target_root, spm_path)
% A_execute_SPM_deface
% Copy selected MRI files from source_root to target_root.
% For files flagged as "deface", run SPM's defacing on the copied file.
% No further modifications (no origin reset, no trimming, no affine edits).

% Defaults
if nargin < 1 || isempty(source_root), source_root = '/data_feivel/Repro_Study'; end
if nargin < 2 || isempty(target_root), target_root = '/data_feivel/Vincent/anon_DATA'; end
if nargin < 3 || isempty(spm_path),    spm_path    = '/home/BRAIN/vwohlfarth/Documents/MATLAB/DLSegPerf/spm'; end

% Ensure SPM is available
if exist(fullfile(spm_path,'spm.m'),'file')
    addpath(spm_path);
else
    error('SPM folder not found at: %s', spm_path);
end

% Ensure target root exists
if ~exist(target_root,'dir'), mkdir(target_root); end

% Log file
timestamp = datestr(now,'yyyymmdd_HHMMSS');
log_file  = fullfile(pwd, sprintf('defacing_log_%s.txt', timestamp));
log_fid   = fopen(log_file,'w');
fprintf(log_fid, 'SPM DEFACING LOG — %s\n\n', datestr(now));

% Cohorts & visits
groups = {
    'DATA_HC',        1, 15;
    'DATA_patients', 16, 23;
};
visits = {'First_visit','Second_visit','Third_visit'};

% Files to handle: {relative_path_template, deface?}
file_info = {
    'output/{sub}/FLAIR/{sub}_FLAIR.nii', true;
    'output/{sub}/T1w/{sub}_T1w.nii',     true;

    'output/{sub}/task-AIR/ASL/ssLICA/T1w_coreg/r{sub}_T1w.nii', true;
    'output/{sub}/task-AIR/ASL/ssRICA/T1w_coreg/r{sub}_T1w.nii', true;

    'output/{sub}/task-AIR/ASL/ssLICA/ASL_fullcoreg/meano_{sub}_task-AIR_acq-epi_label-ssLICA_asl_002_1.nii', true;
    'output/{sub}/task-AIR/ASL/ssRICA/ASL_fullcoreg/meano_{sub}_task-AIR_acq-epi_label-ssRICA_asl_002_1.nii', true;

    'output/{sub}/task-AIR/ASL/ssLICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii', false;
    'output/{sub}/task-AIR/ASL/ssRICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii', false;
    'output/{sub}/task-AIR/ASL/ssLICA/PerfTerrMask/mask_LICA_manual_Corrected.nii', false;
    'output/{sub}/task-AIR/ASL/ssRICA/PerfTerrMask/mask_RICA_manual_Corrected.nii', false;
};

deface_inputs   = {};   % copied files to feed into SPM deface
defaced_expected = {};  % expected anon_* outputs
orig_copies     = {};   % copied originals to delete if anon_* appears

% -------- Copy phase (and queue defacing) --------
for g = 1:size(groups,1)
    group_name    = groups{g,1};
    subject_range = groups{g,2}:groups{g,3};

    for s = subject_range
        sub = sprintf('sub-p%03d', s);
        for v = 1:numel(visits)
            visit = visits{v};

            for f = 1:size(file_info,1)
                rel_path  = strrep(file_info{f,1}, '{sub}', sub);
                do_deface = file_info{f,2};

                src_file = fullfile(source_root, group_name, visit, rel_path);
                tgt_file = fullfile(target_root, group_name, visit, rel_path);

                if isfile(src_file)
                    tgt_folder = fileparts(tgt_file);
                    if ~exist(tgt_folder,'dir'), mkdir(tgt_folder); end
                    copyfile(src_file, tgt_file);

                    [tgt_folder_only, tgt_base, tgt_ext] = fileparts(tgt_file);

                    if do_deface
                        deface_inputs{end+1} = tgt_file; %#ok<AGROW>
                        defaced_expected{end+1} = fullfile(tgt_folder_only, ['anon_' tgt_base tgt_ext]); %#ok<AGROW>
                        orig_copies{end+1} = tgt_file; %#ok<AGROW>
                        msg = sprintf('✓ Copied for defacing: %s\n', tgt_file);
                    else
                        msg = sprintf('✓ Copied: %s\n', tgt_file);
                    end
                else
                    msg = sprintf('⨉ Missing: %s\n', src_file);
                end

                fprintf(msg);
                fprintf(log_fid, msg);
            end
        end
    end
end

% -------- Defacing phase --------
if ~isempty(deface_inputs)
    try
        matlabbatch = {};
        matlabbatch{1}.spm.util.deface.images = deface_inputs';
        spm('defaults','FMRI');
        spm_jobman('initcfg');
        spm_jobman('run', matlabbatch);

        % After defacing, keep anon_* and remove the copied non-anon if anon_* exists
        for i = 1:numel(defaced_expected)
            anon_path = defaced_expected{i};
            orig_path = orig_copies{i};
            if isfile(anon_path)
                try
                    if isfile(orig_path), delete(orig_path); end
                    fprintf(log_fid, '✓ Defaced -> kept: %s | removed original copy: %s\n', anon_path, orig_path);
                    fprintf('✓ Defaced: %s\n', anon_path);
                catch ME
                    fprintf(log_fid, '⚠ Could not delete original copy %s — %s\n', orig_path, ME.message);
                end
            else
                fprintf(log_fid, '⚠ Expected defaced file not found: %s (kept original copy: %s)\n', anon_path, orig_path);
                warning('Expected defaced file not found: %s', anon_path);
            end
        end
    catch ME
        fprintf(log_fid, '⨉ SPM defacing FAILED — %s\n', ME.message);
        warning('SPM defacing failed: %s', ME.message);
    end
else
    fprintf(log_fid, 'No files queued for defacing.\n');
end

% -------- Close log --------
fprintf(log_fid, '\n--- Completed at %s ---\n', datestr(now));
fclose(log_fid);
fprintf('Log file saved to: %s\n', log_file);
disp('--- Copy/Deface complete ---');
end
