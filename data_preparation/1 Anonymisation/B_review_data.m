% B_review_data.m
% Review anonymised NIfTI files and create detailed CSV summary

%% Setup
anon_root = '/data_feivel/Vincent/anon_DATA';
spm_path = '/home/BRAIN/vwohlfarth/Documents/MATLAB/DLSegPerf/spm'; % adjust as needed

if exist(fullfile(spm_path, 'spm.m'), 'file')
    addpath(spm_path);
else
    warning('SPM not found at %s. Make sure it''s in your MATLAB path.', spm_path);
end

groups = {
    'DATA_HC', 1, 15;
    'DATA_patients', 16, 23;
};
visits = {'First_visit', 'Second_visit', 'Third_visit'};

expected_files = {
    'output/{sub}/FLAIR/anon_{sub}_FLAIR.nii',            'anon_FLAIR';
    'output/{sub}/T1w/anon_{sub}_T1w.nii',                'anon_T1w';
    'output/{sub}/task-AIR/ASL/ssLICA/T1w_coreg/anon_r{sub}_T1w.nii', 'L_anon_T1w_coreg';
    'output/{sub}/task-AIR/ASL/ssRICA/T1w_coreg/anon_r{sub}_T1w.nii', 'R_anon_T1w_coreg';
    'output/{sub}/task-AIR/ASL/ssLICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii', 'L_CBF';
    'output/{sub}/task-AIR/ASL/ssRICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii', 'R_CBF';
    'output/{sub}/task-AIR/ASL/ssLICA/PerfTerrMask/mask_LICA_manual_Corrected.nii', 'L_mask';
    'output/{sub}/task-AIR/ASL/ssRICA/PerfTerrMask/mask_RICA_manual_Corrected.nii', 'R_mask';
};

% CSV summary setup
csv_file = fullfile(pwd, sprintf('file_check_summary_%s.csv', datestr(now,'yyyymmdd_HHMMSS')));
csv_fid = fopen(csv_file, 'w');
fprintf(csv_fid, 'Group,Subject,Visit');
for f = 1:size(expected_files,1)
    fprintf(csv_fid, ',%s', expected_files{f,2});
end
fprintf(csv_fid, '\n');

%% Scan and collect
existing_files = {}; % for review

for g = 1:size(groups,1)
    group = groups{g,1};
    sub_range = groups{g,2}:groups{g,3};

    for s = sub_range
        sub = sprintf('sub-p%03d', s);

        for v = 1:numel(visits)
            visit = visits{v};

            row_entry = sprintf('%s,%s,%s', group, sub, visit);

            for f = 1:size(expected_files,1)
                rel_path = strrep(expected_files{f,1}, '{sub}', sub);
                full_path = fullfile(anon_root, group, visit, rel_path);

                if isfile(full_path)
                    row_entry = strcat(row_entry, ',+');

                    % Add to review list only if it's an anonymised file
                    if contains(rel_path, 'anon_') && endsWith(rel_path, '.nii')
                        existing_files{end+1} = full_path;
                    end
                else
                    row_entry = strcat(row_entry, ',-');
                end
            end

            fprintf(csv_fid, '%s\n', row_entry);
        end
    end
end

fclose(csv_fid);
fprintf('✅ Summary CSV written to: %s\n', csv_file);

%% Review loop
if isempty(existing_files)
    error('No anonymised files found for review.');
end

fprintf('Found %d anonymised files for review.\n', numel(existing_files));

for i = 1:length(existing_files)
    file_path = existing_files{i};
    fprintf('\n\n=== [%d/%d] %s ===\n', i, length(existing_files), file_path);

    try
        V = spm_vol(file_path);
        disp('--- NIfTI Header Information (SPM) ---');
        disp(V);
    catch
        warning('SPM error, using MATLAB niftiinfo.');
        try
            V = niftiinfo(file_path);
            disp('--- NIfTI Header Information (MATLAB) ---');
            disp(V);
        catch ME
            warning('Could not read header: %s', ME.message);
            continue;
        end
    end

    try
        spm_image('Display', file_path);
    catch ME
        warning('Could not open viewer for %s: %s\nSkipping...', file_path, ME.message);
        continue;
    end

    fprintf('Press any key to continue...\n');
    pause;
    close all;
end

disp('--- Review complete ---');
