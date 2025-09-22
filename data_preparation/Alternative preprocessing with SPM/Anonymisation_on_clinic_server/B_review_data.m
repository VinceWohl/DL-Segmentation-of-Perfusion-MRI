function [csv_file, existing_files] = B_review_data(anon_root, spm_path, do_interactive)
% B_review_data
% Scan anonymised NIfTI files under anon_root and write a CSV presence table.
% Optionally open each anonymised file in SPM's viewer for manual review.
%
% USAGE:
%   B_review_data();                                  % use defaults, interactive review
%   B_review_data('/path/to/anon_DATA');              % custom root
%   B_review_data('/root', '/path/to/spm', false);    % non-interactive
%
% OUTPUTS:
%   csv_file       - full path to generated CSV summary
%   existing_files - cell array of anonymised files found (for review)

% Defaults
if nargin < 1 || isempty(anon_root)
    anon_root = '/data_feivel/Vincent/anon_DATA';
end
if nargin < 2 || isempty(spm_path)
    spm_path = '/home/BRAIN/vwohlfarth/Documents/MATLAB/DLSegPerf/spm';
end
if nargin < 3 || isempty(do_interactive)
    do_interactive = true;
end

% SPM (optional)
if exist(fullfile(spm_path, 'spm.m'), 'file')
    addpath(spm_path);
else
    warning('SPM not found at %s. Header + viewer steps will fallback where possible.', spm_path);
end

% Cohorts & visits
groups = {
    'DATA_HC',      1, 15;
    'DATA_patients',16, 23;
};
visits = {'First_visit','Second_visit','Third_visit'};

% Expected files (template, label)
expected_files = {
    'output/{sub}/FLAIR/anon_{sub}_FLAIR.nii',                                  'anon_FLAIR';
    'output/{sub}/T1w/anon_{sub}_T1w.nii',                                      'anon_T1w';
    'output/{sub}/task-AIR/ASL/ssLICA/T1w_coreg/anon_r{sub}_T1w.nii',           'L_anon_T1w_coreg';
    'output/{sub}/task-AIR/ASL/ssRICA/T1w_coreg/anon_r{sub}_T1w.nii',           'R_anon_T1w_coreg';
    'output/{sub}/task-AIR/ASL/ssLICA/ASL_fullcoreg/anon_meano_{sub}_task-AIR_acq-epi_label-ssLICA_asl_002_1.nii', 'L_anon_ASLfull_mean';
    'output/{sub}/task-AIR/ASL/ssRICA/ASL_fullcoreg/anon_meano_{sub}_task-AIR_acq-epi_label-ssRICA_asl_002_1.nii', 'R_anon_ASLfull_mean';
    'output/{sub}/task-AIR/ASL/ssLICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii',     'L_CBF';
    'output/{sub}/task-AIR/ASL/ssRICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii',     'R_CBF';
    'output/{sub}/task-AIR/ASL/ssLICA/PerfTerrMask/mask_LICA_manual_Corrected.nii', 'L_mask';
    'output/{sub}/task-AIR/ASL/ssRICA/PerfTerrMask/mask_RICA_manual_Corrected.nii', 'R_mask';
};

% CSV
csv_file = fullfile(pwd, sprintf('file_check_summary_%s.csv', datestr(now,'yyyymmdd_HHMMSS')));
csv_fid  = fopen(csv_file, 'w');
fprintf(csv_fid, 'Group,Subject,Visit');
for f = 1:size(expected_files,1)
    fprintf(csv_fid, ',%s', expected_files{f,2});
end
fprintf(csv_fid, '\n');

% Scan
existing_files = {};
for g = 1:size(groups,1)
    group = groups{g,1};
    sub_range = groups{g,2}:groups{g,3};
    for s = sub_range
        sub = sprintf('sub-p%03d', s);
        for v = 1:numel(visits)
            visit = visits{v};
            row = sprintf('%s,%s,%s', group, sub, visit);
            for f = 1:size(expected_files,1)
                rel_path = strrep(expected_files{f,1}, '{sub}', sub);
                full_path = fullfile(anon_root, group, visit, rel_path);
                if isfile(full_path)
                    row = [row ',+']; %#ok<AGROW>
                    if contains(rel_path,'anon_') && endsWith(rel_path,'.nii')
                        existing_files{end+1} = full_path; %#ok<AGROW>
                    end
                else
                    row = [row ',-']; %#ok<AGROW>
                end
            end
            fprintf(csv_fid, '%s\n', row);
        end
    end
end
fclose(csv_fid);
fprintf('✅ Summary CSV written to: %s\n', csv_file);

% Interactive review (optional)
if ~do_interactive
    return;
end
if isempty(existing_files)
    error('No anonymised files found for review.');
end

fprintf('Found %d anonymised files for review.\n', numel(existing_files));
for i = 1:numel(existing_files)
    file_path = existing_files{i};
    fprintf('\n\n=== [%d/%d] %s ===\n', i, numel(existing_files), file_path);

    try
        V = spm_vol(file_path); %#ok<NASGU>
        disp('--- NIfTI header via SPM (spm_vol) ---');
        disp(V);
    catch
        warning('SPM spm_vol failed, trying niftiinfo...');
        try
            V = niftiinfo(file_path);
            disp('--- NIfTI header via MATLAB (niftiinfo) ---');
            disp(V);
        catch ME
            warning('Could not read header: %s', ME.message);
            continue;
        end
    end

    try
        spm_image('Display', file_path);
    catch ME
        warning('Viewer failed for %s: %s. Skipping...', file_path, ME.message);
        continue;
    end
    fprintf('Press any key to continue...\n');
    pause;
    close all;
end
disp('--- Review complete ---');
end
