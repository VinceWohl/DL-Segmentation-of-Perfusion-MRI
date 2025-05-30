% author: VW
% Script to reverse subject folder and file name renumeration using saved mapping.

%% Setup paths
source_root = '/home/BRAIN/vwohlfarth/Documents/anon_Data';
mapping_file = '/home/BRAIN/vwohlfarth/Documents/renumeration.csv';
visits = {'First_visit', 'Second_visit', 'Third_visit'};

%% Load renumeration mapping
if ~isfile(mapping_file)
    error('Mapping file not found: %s', mapping_file);
end

map = readtable(mapping_file);
original_ids = map.OriginalID;
new_ids = map.NewID;
numSubs = height(map);

%% Reverse renaming across visits
for v = 1:length(visits)
    visit = visits{v};
    out_path = fullfile(source_root, visit, 'output');
    if ~isfolder(out_path), continue; end

    for i = 1:numSubs
        orig_id = original_ids{i};
        new_id = new_ids{i};

        current_folder = fullfile(out_path, new_id);
        target_folder = fullfile(out_path, orig_id);

        if isfolder(current_folder)
            movefile(current_folder, target_folder);

            % Rename NIfTI files within the folder tree
            nii_files = dir(fullfile(target_folder, '**', sprintf('*%s*.nii', new_id)));
            for k = 1:length(nii_files)
                old_file = fullfile(nii_files(k).folder, nii_files(k).name);
                new_name = strrep(nii_files(k).name, new_id, orig_id);
                new_file = fullfile(nii_files(k).folder, new_name);
                movefile(old_file, new_file);
                fprintf('  → Reverted file: %s\n', new_file);
            end

            fprintf('✓ Reverted folder: %s -> %s\n', new_id, orig_id);
        end
    end
end

fprintf('\n✅ Subject renumeration successfully reversed using mapping: %s\n', mapping_file);