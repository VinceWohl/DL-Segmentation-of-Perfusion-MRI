% C_renumerate_subjects.m
% Renumerate subject folders and filenames based on renumeration.csv
% Skips missing folders/files and avoids folder name collisions

%% Setup
source_root = '/data_feivel/Vincent/anon_DATA';
mapping_csv = 'renumeration.csv';  % Path to the mapping file

groups = {'DATA_HC', 'DATA_patients'};
visits = {'First_visit', 'Second_visit', 'Third_visit'};

%% Load CSV mapping
if ~isfile(mapping_csv)
    error('Mapping file not found: %s', mapping_csv);
end

map_table = readtable(mapping_csv, 'TextType', 'string');
orig_ids = map_table.OriginalID;
new_ids = map_table.NewID;

fprintf('✅ Loaded mapping for %d subjects from: %s\n\n', height(map_table), mapping_csv);

%% Renaming process
for g = 1:length(groups)
    group_path = fullfile(source_root, groups{g});

    for v = 1:length(visits)
        visit = visits{v};
        visit_path = fullfile(group_path, visit, 'output');
        if ~isfolder(visit_path), continue; end

        % Step 1: Temporary renaming to avoid name collisions
        for i = 1:height(map_table)
            orig_id = orig_ids(i);
            old_folder = fullfile(visit_path, orig_id);
            if isfolder(old_folder)
                tmp_folder = fullfile(visit_path, ['tmp_' + orig_id]);
                movefile(char(old_folder), char(tmp_folder));
            end
        end

        % Step 2: Final rename and NIfTI renaming
        for i = 1:height(map_table)
            orig_id = orig_ids(i);
            new_id = new_ids(i);

            tmp_folder = fullfile(visit_path, ['tmp_' + orig_id]);
            new_folder = fullfile(visit_path, new_id);

            if isfolder(tmp_folder)
                movefile(char(tmp_folder), char(new_folder));

                % Rename all files that contain the original subject ID
                nii_files = dir(fullfile(new_folder, '**', sprintf('*%s*.nii', orig_id)));
                for k = 1:length(nii_files)
                    old_file = fullfile(nii_files(k).folder, nii_files(k).name);
                    new_name = strrep(nii_files(k).name, orig_id, new_id);
                    new_file = fullfile(nii_files(k).folder, new_name);
                    movefile(char(old_file), char(new_file));
                    fprintf('  → Renamed file: %s\n', new_file);
                end

                fprintf('✓ [%s - %s] %s → %s\n', groups{g}, visit, orig_id, new_id);
            else
                fprintf('⚠ [%s - %s] %s: Folder not found, skipping.\n', groups{g}, visit, orig_id);
            end
        end
    end
end

fprintf('\n🎉 Renumeration complete using mapping file: %s\n', mapping_csv);
