%% Script to prepare the input for the nnUNet model in an adequate format
% author: VW

%% Setup paths
source_root = 'C:\Users\Vincent Wohlfarth\Data\anon_Data';
target_root = 'C:\Users\Vincent Wohlfarth\Data\nnUNet_raw\Dataset001_PerfusionTerritories';

imagesTr = fullfile(target_root, 'imagesTr');
imagesTs = fullfile(target_root, 'imagesTs');
labelsTr = fullfile(target_root, 'labelsTr');

if ~exist(imagesTr, 'dir'); mkdir(imagesTr); end
if ~exist(imagesTs, 'dir'); mkdir(imagesTs); end
if ~exist(labelsTr, 'dir'); mkdir(labelsTr); end

%% Subject and visit configuration
all_subjects = {'sub-p001', 'sub-p002', 'sub-p003', 'sub-p004', 'sub-p005', ...
                'sub-p006', 'sub-p007', 'sub-p008', 'sub-p009', 'sub-p010', ...
                'sub-p011', 'sub-p012', 'sub-p013', 'sub-p014', 'sub-p015'};

test_subjects = {'sub-p001', 'sub-p007', 'sub-p010'};
visits = {'First_visit', 'Second_visit'};
visit_codes = {'v1', 'v2'};

%% Subject ID to index mapping
subject_numbers = containers.Map();
for i = 1:numel(all_subjects)
    subj_str = all_subjects{i};
    subject_numbers(subj_str) = sscanf(subj_str, 'sub-p%d');
end

%% Main processing loop
for i = 1:numel(all_subjects)
    subj_id = all_subjects{i};
    subj_index = subject_numbers(subj_id);
    is_test = ismember(subj_id, test_subjects);

    for v = 1:2
        visit = visits{v};
        visit_code = visit_codes{v};
        base_path = fullfile(source_root, visit, 'output', subj_id);

        % Input file paths
        cbf_lica = fullfile(base_path, 'task-AIR', 'ASL', 'ssLICA', 'CBF_nativeSpace', 'CBF_3_BRmsk_CSF.nii');
        cbf_rica = fullfile(base_path, 'task-AIR', 'ASL', 'ssRICA', 'CBF_nativeSpace', 'CBF_3_BRmsk_CSF.nii');
        t1w      = fullfile(base_path, 'task-AIR', 'ASL', 'ssRICA', 'T1w_coreg', ['anon_r', subj_id, '_T1w.nii']);

        % Naming convention
        datapoint_name = sprintf('PerfTerr%03d-%s', subj_index, visit_code);
        if is_test
            target_img_dir = imagesTs;
        else
            target_img_dir = imagesTr;
        end

        % Load and align images
        cbf_lica_data = single(niftiread(cbf_lica));
        cbf_rica_data = single(niftiread(cbf_rica));
        t1w_data      = niftiread(t1w);
        target_size   = size(t1w_data);

        if ~isequal(size(cbf_lica_data), target_size)
            cbf_lica_data = imresize3(cbf_lica_data, target_size, 'nearest');
        end
        if ~isequal(size(cbf_rica_data), target_size)
            cbf_rica_data = imresize3(cbf_rica_data, target_size, 'nearest');
        end

        % Save images
        ref_info = niftiinfo(t1w);
        ref_info.Datatype = 'single';
        niftiwrite(cbf_lica_data, fullfile(target_img_dir, [datapoint_name '_0000.nii']), ref_info, 'Compressed', false);
        niftiwrite(cbf_rica_data, fullfile(target_img_dir, [datapoint_name '_0001.nii']), ref_info, 'Compressed', false);
        copyfile(t1w, fullfile(target_img_dir, [datapoint_name '_0002.nii']));

        % Skip test labels
        if is_test; continue; end

        % Load and align masks
        mask_lica = fullfile(base_path, 'task-AIR', 'ASL', 'ssLICA', 'PerfTerrMask', 'mask_LICA_manual.nii');
        mask_rica = fullfile(base_path, 'task-AIR', 'ASL', 'ssRICA', 'PerfTerrMask', 'mask_RICA_manual.nii');
        mask_lica_data = niftiread(mask_lica) > 0;
        mask_rica_data = niftiread(mask_rica) > 0;

        if ~isequal(size(mask_lica_data), size(mask_rica_data))
            mask_rica_data = imresize3(mask_rica_data, size(mask_lica_data), 'nearest');
        end

        % Combine into multi-class mask
        combined_mask = zeros(size(mask_lica_data), 'int16');
        combined_mask(mask_lica_data & ~mask_rica_data) = 1;
        combined_mask(mask_rica_data & ~mask_lica_data) = 2;
        combined_mask(mask_lica_data & mask_rica_data)  = 3;

        % Resize mask to match T1w
        if ~isequal(size(combined_mask), target_size)
            combined_mask = imresize3(combined_mask, target_size, 'nearest');
        end

        % Fix interpolation errors: round, cast, clip
        combined_mask = round(combined_mask);                  % force integers
        combined_mask(~ismember(combined_mask, [0 1 2 3])) = 0;
        combined_mask = cast(combined_mask, 'int16');

        % Flip mask left-right to correct anatomical label orientation
        %combined_mask = flip(combined_mask, 1);
        
        % Match spatial transform to image header to avoid direction/origin mismatch
        ref_info_label = niftiinfo(t1w);
        ref_info_label.Datatype = 'int16';
        ref_info_label.SpaceUnits = ref_info.SpaceUnits;
        ref_info_label.TimeUnits = ref_info.TimeUnits;
        ref_info_label.Transform = ref_info.Transform;
        ref_info_label.Qfactor = ref_info.Qfactor;
        
        % Write final mask
        niftiwrite(combined_mask, fullfile(labelsTr, [datapoint_name '.nii']), ref_info_label, 'Compressed', false);

    end
end