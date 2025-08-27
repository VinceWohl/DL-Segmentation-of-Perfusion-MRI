% Define base paths
sourceBase = '/data_feivel/Repro_Study/Segmentations_Corrected';
targetBase = '/data_feivel/Repro_Study';

% Define runs and visits mapping
runs = {'First_run', 'Second_run', 'Third_run'};
visits = {'First_visit', 'Second_visit', 'Third_visit'};

% Define sides and file names
sides = {'LICA', 'RICA'};
maskName = 'mask_%s_manual_Corrected.nii';

% Subjects 001-015 are healthy controls, 016-023 are patients
subjectsHC = arrayfun(@(x) sprintf('sub-p%03d', x), 1:15, 'UniformOutput', false);
subjectsPatients = arrayfun(@(x) sprintf('sub-p%03d', x), 16:23, 'UniformOutput', false);

% Initialize logging
missingSourceFiles = {};
missingTargetDirs = {};
copiedFiles = 0;

% Loop over all runs (visits)
for r = 1:length(runs)
    runName = runs{r};
    visitName = visits{r};

    % Go through HC first
    for s = 1:length(subjectsHC)
        subj = subjectsHC{s};
        group = 'DATA_HC';

        for h = 1:length(sides)
            side = sides{h};

            % Define source file path
            sourceFile = fullfile(sourceBase, runName, subj, side, ...
                                  sprintf(maskName, side));

            % Define target directory
            targetDir = fullfile(targetBase, group, visitName, 'output', ...
                                 subj, 'task-AIR', 'ASL', ['ss' side], 'PerfTerrMask');

            % Target file name remains the same
            targetFile = fullfile(targetDir, sprintf(maskName, side));

            % Check existence and copy
            if isfile(sourceFile)
                if isfolder(targetDir)
                    copyfile(sourceFile, targetFile);
                    copiedFiles = copiedFiles + 1;
                else
                    missingTargetDirs{end+1,1} = targetDir;
                end
            else
                missingSourceFiles{end+1,1} = sourceFile;
            end
        end
    end

    % Then go through patients
    for s = 1:length(subjectsPatients)
        subj = subjectsPatients{s};
        group = 'DATA_patients';

        for h = 1:length(sides)
            side = sides{h};

            % Define source file path
            sourceFile = fullfile(sourceBase, runName, subj, side, ...
                                  sprintf(maskName, side));

            % Define target directory
            targetDir = fullfile(targetBase, group, visitName, 'output', ...
                                 subj, 'task-AIR', 'ASL', ['ss' side], 'PerfTerrMask');

            % Target file name remains the same
            targetFile = fullfile(targetDir, sprintf(maskName, side));

            % Check existence and copy
            if isfile(sourceFile)
                if isfolder(targetDir)
                    copyfile(sourceFile, targetFile);
                    copiedFiles = copiedFiles + 1;
                else
                    missingTargetDirs{end+1,1} = targetDir;
                end
            else
                missingSourceFiles{end+1,1} = sourceFile;
            end
        end
    end
end

% Report summary
fprintf('✅ Total copied files: %d\n', copiedFiles);
if ~isempty(missingSourceFiles)
    fprintf('❌ Missing source files (%d):\n', length(missingSourceFiles));
    disp(missingSourceFiles);
end
if ~isempty(missingTargetDirs)
    fprintf('⚠️ Missing target directories (%d):\n', length(missingTargetDirs));
    disp(missingTargetDirs);
end
