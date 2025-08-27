% A_execute_SPM_deface_updated.m
% One-step script to extract, copy, and deface selected MRI files into anon_DATA

%% Setup root paths
source_root = '/data_feivel/Repro_Study';
target_root = '/data_feivel/Vincent/anon_DATA';

% SPM path (adjust if needed)
spm_path = '/home/BRAIN/vwohlfarth/Documents/MATLAB/DLSegPerf/spm';
if exist(fullfile(spm_path, 'spm.m'), 'file')
    addpath(spm_path);
else
    error('SPM folder not found at: %s', spm_path);
end

% Create target root if not existing
if ~exist(target_root, 'dir')
    mkdir(target_root);
end

%% Create log file
timestamp = datestr(now,'yyyymmdd_HHMMSS');
log_file = fullfile(pwd, sprintf('defacing_log_%s.txt', timestamp));
log_fid = fopen(log_file, 'w');
fprintf(log_fid, 'SPM DEFACING LOG — %s\n\n', datestr(now));

%% Define subject groups, subject ranges, and visit names
groups = {
    'DATA_HC', 1, 15;
    'DATA_patients', 16, 23;
};
visits = {'First_visit', 'Second_visit', 'Third_visit'};

%% Define which files to deface and which to copy only
file_info = {
    'output/{sub}/FLAIR/{sub}_FLAIR.nii', true;
    'output/{sub}/T1w/{sub}_T1w.nii', true;
    'output/{sub}/task-AIR/ASL/ssLICA/T1w_coreg/r{sub}_T1w.nii', true;
    'output/{sub}/task-AIR/ASL/ssRICA/T1w_coreg/r{sub}_T1w.nii', true;
    'output/{sub}/task-AIR/ASL/ssLICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii', false;
    'output/{sub}/task-AIR/ASL/ssRICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii', false;
    'output/{sub}/task-AIR/ASL/ssLICA/PerfTerrMask/mask_LICA_manual_Corrected.nii', false;
    'output/{sub}/task-AIR/ASL/ssRICA/PerfTerrMask/mask_RICA_manual_Corrected.nii', false;
};

deface_paths = {};
copied_originals = {};

%% Traverse all subjects, visits and files
for g = 1:size(groups,1)
    group_name = groups{g,1};
    subject_range = groups{g,2}:groups{g,3};

    for s = subject_range
        sub = sprintf('sub-p%03d', s);

        for v = 1:length(visits)
            visit = visits{v};

            for f = 1:size(file_info,1)
                rel_path_template = file_info{f,1};
                do_deface = file_info{f,2};

                rel_path = strrep(rel_path_template, '{sub}', sub);
                src_file = fullfile(source_root, group_name, visit, rel_path);
                tgt_file = fullfile(target_root, group_name, visit, rel_path);

                if isfile(src_file)
                    tgt_folder = fileparts(tgt_file);
                    if ~exist(tgt_folder, 'dir')
                        mkdir(tgt_folder);
                    end
                    copyfile(src_file, tgt_file);

                    if do_deface
                        deface_paths{end+1} = tgt_file; %#ok<*SAGROW>
                        copied_originals{end+1} = tgt_file;
                        msg = sprintf('✓ Copied for defacing: %s\n', tgt_file);
                    else
                        msg = sprintf('✓ Copied (no defacing): %s\n', tgt_file);
                    end
                else
                    msg = sprintf('⨉ Missing: %s\n', src_file);
                end

                % Write to console and log
                fprintf(msg);
                fprintf(log_fid, msg);
            end
        end
    end
end

%% Run SPM defacing if applicable
if ~isempty(deface_paths)
    matlabbatch = {};
    matlabbatch{1}.spm.util.deface.images = deface_paths';

    spm('defaults', 'FMRI');
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);

    % Delete the copied originals after defacing
    for i = 1:length(copied_originals)
        try
            if isfile(copied_originals{i})
                delete(copied_originals{i});
            end
        catch ME
            warning('Failed to delete %s: %s', copied_originals{i}, ME.message);
            fprintf(log_fid, '⚠ Warning: Could not delete %s — %s\n', copied_originals{i}, ME.message);
        end
    end
else
    disp('No files found for defacing.');
    fprintf(log_fid, 'No files found for defacing.\n');
end

%% Close log
fprintf(log_fid, '\n--- Completed at %s ---\n', datestr(now));
fclose(log_fid);
fprintf('Log file saved to: %s\n', log_file);
disp('--- Data extraction and SPM defacing complete ---');