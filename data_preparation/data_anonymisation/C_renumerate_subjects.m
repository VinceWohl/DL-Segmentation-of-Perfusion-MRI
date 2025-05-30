% author: VW
% Script to randomly renumerate subject folder and file names consistently across visits.
% Only folders and files that contain "sub-p..." subject IDs are renamed.
% Mapping of original to new subject IDs is saved as CSV.

%% Setup paths
source_root = '/home/BRAIN/vwohlfarth/Documents/anon_Data';
map_output = '/home/BRAIN/vwohlfarth/Documents/renumeration.csv';
visits = {'First_visit', 'Second_visit', 'Third_visit'};

%% Step 1: Collect all unique subject IDs
all_subs = {};
for i = 1:length(visits)
    visit_path = fullfile(source_root, visits{i}, 'output');
    if ~isfolder(visit_path), continue; end
    subs = dir(fullfile(visit_path, 'sub-p*'));
    sub_names = {subs([subs.isdir]).name};
    all_subs = union(all_subs, sub_names);
end

%% Step 2: Generate randomized mapping
numSubs = numel(all_subs);
shuffled_idx = randperm(numSubs);
new_ids = arrayfun(@(i) sprintf('sub-p%03d', i), shuffled_idx, 'UniformOutput', false);
mapping_table = table(all_subs(:), new_ids(:), 'VariableNames', {'OriginalID', 'NewID'});

%% Step 3: Apply renaming across visits
for v = 1:length(visits)
    visit = visits{v};
    out_path = fullfile(source_root, visit, 'output');
    if ~isfolder(out_path), continue; end

    % Use temp renaming to avoid name collisions
    for i = 1:numSubs
        orig_id = mapping_table.OriginalID{i};
        old_folder = fullfile(out_path, orig_id);
        if isfolder(old_folder)
            temp_folder = fullfile(out_path, ['tmp_' orig_id]);
            movefile(old_folder, temp_folder);
        end
    end

    for i = 1:numSubs
        orig_id = mapping_table.OriginalID{i};
        new_id = mapping_table.NewID{i};

        tmp_folder = fullfile(out_path, ['tmp_' orig_id]);
        final_folder = fullfile(out_path, new_id);

        if isfolder(tmp_folder)
            movefile(tmp_folder, final_folder);

            % Rename NIfTI files within the folder tree
            nii_files = dir(fullfile(final_folder, '**', sprintf('*%s*.nii', orig_id)));
            for k = 1:length(nii_files)
                old_file = fullfile(nii_files(k).folder, nii_files(k).name);
                new_name = strrep(nii_files(k).name, orig_id, new_id);
                new_file = fullfile(nii_files(k).folder, new_name);
                movefile(old_file, new_file);
                fprintf('  → Renamed file: %s\n', new_file);
            end

            fprintf('✓ Renamed folder: %s -> %s\n', orig_id, new_id);
        end
    end
end

%% Step 4: Save the mapping
writetable(mapping_table, map_output);

fprintf('\n✅ Subject renumeration complete. Mapping saved to: %s\n', map_output);