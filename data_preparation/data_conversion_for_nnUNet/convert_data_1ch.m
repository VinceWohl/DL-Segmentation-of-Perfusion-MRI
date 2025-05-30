% author: VW
% Script to prepare the input for the nnUNet model in an adequate format

%% Setup paths
source_root = 'C:\Users\Vincent Wohlfarth\Data\anon_Data\';
target_root = 'C:\Users\Vincent Wohlfarth\Data\nnUNet_raw\Dataset001_PerfusionTerritories';

imagesTr = fullfile(target_root, 'imagesTr');
imagesTs = fullfile(target_root, 'imagesTs');
labelsTr = fullfile(target_root, 'labelsTr');

% Create folders if they don't exist
if ~exist(imagesTr, 'dir'), mkdir(imagesTr); end
if ~exist(imagesTs, 'dir'), mkdir(imagesTs); end
if ~exist(labelsTr, 'dir'), mkdir(labelsTr); end

%% Subject and visit configuration
subjects_train = {'sub-p002', 'sub-p003', 'sub-p004', 'sub-p006', 'sub-p008', 'sub-p009', 'sub-p011', 'sub-p012', 'sub-p014', 'sub-p015'};
subjects_val = {'sub-p005', 'sub-p013'};
subjects_test = {'sub-p001', 'sub-p007', 'sub-p010'};

visits = {'First_visit', 'Second_visit'};
visit_codes = {'v1', 'v2'};
hemispheres = {'ssLICA', 'ssRICA'};
hem_labels = {'L', 'R'};

%% Processing training and validation data
for subj_group = [{subjects_train}, {subjects_val}]
    subjects = subj_group{1};
    for s = 1:length(subjects)
        subID = subjects{s};
        shortID = extractAfter(subID, 'sub-p');
        for v = 1:length(visits)
            visit = visits{v};
            visit_code = visit_codes{v};
            for h = 1:length(hemispheres)
                hem = hemispheres{h};
                hem_label = hem_labels{h};
                
                % --- File paths ---
                src_path_CBF = fullfile(source_root, visit, 'output', subID, 'task-AIR', 'ASL', hem, 'CBF_nativeSpace', 'CBF_3_BRmsk_CSF.nii');
                src_path_mask = fullfile(source_root, visit, 'output', subID, 'task-AIR', 'ASL', hem, 'PerfTerrMask', ['mask_', hem_label, 'ICA_manual.nii']);

                % --- File names ---
                filename = sprintf('PerfTerr%s%s-%s_0000.nii', shortID, visit_code, hem_label);
                labelname = sprintf('PerfTerr%s%s-%s.nii', shortID, visit_code, hem_label);
                
                % Copy images
                copyfile(src_path_CBF, fullfile(imagesTr, filename));
                copyfile(src_path_mask, fullfile(labelsTr, labelname));
            end
        end
    end
end

%% Processing testing data
for s = 1:length(subjects_test)
    subID = subjects_test{s};
    shortID = extractAfter(subID, 'sub-p');
    for v = 1:length(visits)
        visit = visits{v};
        visit_code = visit_codes{v};
        for h = 1:length(hemispheres)
            hem = hemispheres{h};
            hem_label = hem_labels{h};

            src_path_CBF = fullfile(source_root, visit, 'output', subID, 'task-AIR', 'ASL', hem, 'CBF_nativeSpace', 'CBF_3_BRmsk_CSF.nii');
            
            filename = sprintf('PerfTerr%s%s-%s_0000.nii', shortID, visit_code, hem_label);
            
            % Copy only images (no masks)
            copyfile(src_path_CBF, fullfile(imagesTs, filename));
        end
    end
end

disp('Data preparation for nnU-Net completed successfully.');