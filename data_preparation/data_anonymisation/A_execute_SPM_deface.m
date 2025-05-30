% author: VW
% One-step script to extract, copy, and deface MRI files into anon_Data

%% Setup source and target root directories
source_root = '/data_feivel/Repro_Study/DATA';
target_root = '/home/BRAIN/vwohlfarth/Documents/anon_Data';

% Add SPM to MATLAB path
spm_path = '/home/BRAIN/vwohlfarth/Documents/MATLAB/spm';
if exist(fullfile(spm_path, 'spm.m'), 'file')
    addpath(spm_path);
else
    error('SPM folder not found at: %s', spm_path);
end

if ~exist(target_root, 'dir')
    mkdir(target_root);
end

%% Define visits and subjects
visits = {'First_visit', 'Second_visit', 'Third_visit'};
subjects = arrayfun(@(i) sprintf('sub-p%03d', i), 1:15, 'UniformOutput', false);

%% Define file patterns and defacing status
file_info = {
    'output/{sub}/FLAIR/{sub}_FLAIR.nii', true;
    'output/{sub}/T1w/{sub}_T1w.nii', true;
    'output/{sub}/task-AIR/ASL/ssLICA/T1w_coreg/r{sub}_T1w.nii', true;
    'output/{sub}/task-AIR/ASL/ssRICA/T1w_coreg/r{sub}_T1w.nii', true;
    'output/{sub}/task-AIR/ASL/ssLICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii', false;
    'output/{sub}/task-AIR/ASL/ssRICA/CBF_nativeSpace/CBF_3_BRmsk_CSF.nii', false;
    'output/{sub}/task-AIR/ASL/ssLICA/PerfTerrMask/mask_LICA_manual.nii', false;
    'output/{sub}/task-AIR/ASL/ssRICA/PerfTerrMask/mask_RICA_manual.nii', false;
};

deface_paths = {};
copied_originals = {};

%% Copy and prepare for defacing
for v = 1:length(visits)
    visit = visits{v};
    for s = 1:length(subjects)
        sub = subjects{s};
        for f = 1:size(file_info,1)
            rel_path_template = file_info{f,1};
            do_deface = file_info{f,2};

            rel_path = strrep(rel_path_template, '{sub}', sub);
            src_file = fullfile(source_root, visit, rel_path);
            tgt_file = fullfile(target_root, visit, rel_path);

            if isfile(src_file)
                tgt_folder = fileparts(tgt_file);
                if ~exist(tgt_folder, 'dir')
                    mkdir(tgt_folder);
                end
                copyfile(src_file, tgt_file);

                if do_deface
                    deface_paths{end+1} = tgt_file; %#ok<*SAGROW>
                    copied_originals{end+1} = tgt_file;
                    fprintf('✓ Copied for defacing: %s\n', rel_path);
                else
                    fprintf('✓ Copied (no defacing): %s\n', rel_path);
                end
            else
                fprintf('⨉ Missing: %s\n', src_file);
            end
        end
    end
end

%% Run SPM defacing on copied files
if ~isempty(deface_paths)
    matlabbatch = {};
    matlabbatch{1}.spm.util.deface.images = deface_paths';

    spm('defaults', 'FMRI');
    spm_jobman('initcfg');
    spm_jobman('run', matlabbatch);

    % Delete original copies after defacing (keep only anon_*)
    for i = 1:length(copied_originals)
        try
            if isfile(copied_originals{i})
                delete(copied_originals{i});
            end
        catch ME
            warning('Failed to delete %s: %s', copied_originals{i}, ME.message);
        end
    end
else
    disp('No files found for defacing.');
end

disp('--- Data extraction and SPM defacing complete ---');