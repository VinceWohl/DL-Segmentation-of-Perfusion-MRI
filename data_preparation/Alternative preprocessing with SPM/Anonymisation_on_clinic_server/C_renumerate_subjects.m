function C_renumerate_subjects(source_root, mapping_csv)
% C_renumerate_subjects
% Renumber subject folders and files according to a 2-column CSV mapping:
%   OriginalID,NewID   (e.g., sub-p001,sub-pA01)
% Works across HC/patient groups and all visits. Handles .nii and .nii.gz.
%
% USAGE:
%   C_renumerate_subjects();                              % use defaults
%   C_renumerate_subjects('/path/to/anon_DATA','map.csv');

% Defaults
if nargin < 1 || isempty(source_root)
    source_root = '/data_feivel/Vincent/anon_DATA';
end
if nargin < 2 || isempty(mapping_csv)
    mapping_csv = 'renumeration.csv';
end

% Validate mapping
if ~isfile(mapping_csv)
    error('Mapping file not found: %s', mapping_csv);
end
map_table = readtable(mapping_csv, 'TextType','string');
if ~all(ismember({'OriginalID','NewID'}, map_table.Properties.VariableNames))
    error('Mapping CSV must contain columns: OriginalID, NewID');
end
orig_ids = string(map_table.OriginalID);
new_ids  = string(map_table.NewID);
fprintf('✅ Loaded mapping for %d subjects from: %s\n\n', height(map_table), mapping_csv);

% Scope
groups = {'DATA_HC','DATA_patients'};
visits = {'First_visit','Second_visit','Third_visit'};

% Process
for g = 1:numel(groups)
    group_path = fullfile(source_root, groups{g});
    for v = 1:numel(visits)
        visit = visits{v};
        visit_path = fullfile(group_path, visit, 'output');
        if ~isfolder(visit_path), continue; end

        % 1) Temp rename to avoid collisions
        for i = 1:height(map_table)
            orig_id   = strtrim(orig_ids(i));
            old_folder = fullfile(visit_path, orig_id);
            if isfolder(old_folder)
                tmp_folder = fullfile(visit_path, "tmp_" + orig_id);
                movefile(char(old_folder), char(tmp_folder));
            end
        end

        % 2) Final subject folder rename + deep filename renaming
        for i = 1:height(map_table)
            orig_id  = strtrim(orig_ids(i));
            new_id   = strtrim(new_ids(i));
            tmp_folder = fullfile(visit_path, "tmp_" + orig_id);
            new_folder = fullfile(visit_path, new_id);

            if isfolder(tmp_folder)
                % subject folder -> new ID
                movefile(char(tmp_folder), char(new_folder));

                % recurse and rename files containing old ID in name (.nii / .nii.gz)
                pat1 = fullfile(new_folder, "**", "*" + orig_id + "*.nii");
                pat2 = fullfile(new_folder, "**", "*" + orig_id + "*.nii.gz");
                list1 = dir(pat1);
                list2 = dir(pat2);
                hits  = [list1; list2];

                renamed = 0;
                for k = 1:numel(hits)
                    old_file = fullfile(hits(k).folder, hits(k).name);
                    new_name = strrep(hits(k).name, orig_id, new_id);
                    new_file = fullfile(hits(k).folder, new_name);
                    if ~strcmp(old_file, new_file)
                        movefile(char(old_file), char(new_file));
                        renamed = renamed + 1;
                        fprintf('  → %s\n', new_file);
                    end
                end

                fprintf('✓ [%s | %s] %s → %s  (files renamed: %d)\n', ...
                        groups{g}, visit, orig_id, new_id, renamed);
            else
                fprintf('⚠ [%s | %s] %s: Folder not found, skipping.\n', ...
                        groups{g}, visit, orig_id);
            end
        end
    end
end

fprintf('\n🎉 Renumeration complete using mapping file: %s\n', mapping_csv);
end
