%% A_coregister_FLAIR_to_other_visits.m
% Coregister missing FLAIR images for Second/Third visits using SPM.
% Keeps ONLY the final coregistered FLAIR in each target FLAIR folder.
% Cleans up: temp copied FLAIR, any r*.nii byproducts (incl. resliced source T1w).

%% -------------------- USER SETTINGS --------------------
rootDir      = 'C:\Users\Vincent Wohlfarth\Data\anon_Data_250808';
cohorts      = {'DATA_HC','DATA_patients'};
targetVisits = {'Second_visit','Third_visit'};
overwrite    = false;   % true: recreate even if target FLAIR exists
dryRun       = false;   % true: print actions only (no file changes)
logFile      = fullfile(rootDir, ['A_coregister_FLAIR_log_' datestr(now,'yyyymmdd_HHMMSS') '.txt']);
%% -------------------------------------------------------

if ~dryRun, diary(logFile); diary on; end
fprintf('\n=== Coregister FLAIR to other visits (SPM) ===\nRoot: %s\nOverwrite: %d | Dry-run: %d\n\n', rootDir, overwrite, dryRun);

% Init SPM
spm('defaults','FMRI'); spm_jobman('initcfg');

for c = 1:numel(cohorts)
    cohort = cohorts{c};
    firstVisitOut = fullfile(rootDir, cohort, 'First_visit', 'output');
    if ~isfolder(firstVisitOut)
        warning('Missing folder: %s (skipping cohort)', firstVisitOut); 
        continue;
    end

    d = dir(fullfile(firstVisitOut,'sub-p*')); 
    subs = {d([d.isdir]).name};

    for s = 1:numel(subs)
        sid = subs{s};                      % e.g., 'sub-p001'
        flair_first = fullfile(firstVisitOut, sid, 'FLAIR', ['anon_' sid '_FLAIR.nii']);
        t1_first    = fullfile(firstVisitOut, sid, 'T1w',   ['anon_' sid '_T1w.nii']);

        if ~isfile(t1_first)
            fprintf('  [%s] First_visit T1w missing -> SKIP subject\n', sid); 
            continue;
        end
        if ~isfile(flair_first)
            fprintf('  [%s] First_visit FLAIR missing -> SKIP subject\n', sid); 
            continue;
        end

        for v = 1:numel(targetVisits)
            visit = targetVisits{v};
            visitOut = fullfile(rootDir, cohort, visit, 'output', sid);
            t1_target = fullfile(visitOut, 'T1w', ['anon_' sid '_T1w.nii']);
            flair_target_dir = fullfile(visitOut, 'FLAIR');
            final_flair = fullfile(flair_target_dir, ['anon_' sid '_FLAIR.nii']);

            if ~isfile(t1_target)
                fprintf('  [%s | %s] Target T1w missing -> SKIP visit\n', sid, visit); 
                continue;
            end

            if isfile(final_flair) && ~overwrite
                % enforce cleanup (in case of left-overs)
                cleanup_flair_folder(flair_target_dir, final_flair, dryRun);
                fprintf('  [%s | %s] Target FLAIR present -> SKIP (cleanup done)\n', sid, visit);
                continue;
            end

            if dryRun
                fprintf('  [DRY | %s | %s] Would coregister First→%s FLAIR\n', sid, visit, visit);
                fprintf('      REF: %s\n      SRC: %s\n      OTH: %s\n      OUT: %s\n', t1_target, t1_first, flair_first, final_flair);
                continue;
            end

            if ~isfolder(flair_target_dir), mkdir(flair_target_dir); end

            % Copy First_visit FLAIR into target FLAIR dir as temp "other"
            [~, flairBase, flairExt] = fileparts(flair_first);
            tmp_other = fullfile(flair_target_dir, ['tmp_' flairBase flairExt]);
            try
                copyfile(flair_first, tmp_other);
            catch ME
                warning('  [%s | %s] Copy FLAIR failed: %s', sid, visit, ME.message);
                continue;
            end

            % Build & run SPM batch (est + write)
            matlabbatch = build_coreg_batch(t1_target, t1_first, tmp_other); %#ok<NASGU>
            try
                spm_jobman('run', matlabbatch);
            catch ME
                warning('  [%s | %s] SPM job failed: %s', sid, visit, ME.message);
                if isfile(tmp_other), delete(tmp_other); end
                continue;
            end

            % Expected outputs
            r_other = fullfile(flair_target_dir, ['r' 'tmp_' flairBase flairExt]);  % resliced FLAIR in target dir
            if ~isfile(r_other)
                alt = dir(fullfile(flair_target_dir, 'rtmp_*.nii'));
                if numel(alt)==1
                    r_other = fullfile(flair_target_dir, alt(1).name);
                end
            end

            % Move resliced FLAIR to final name
            try
                if isfile(r_other)
                    if isfile(final_flair) && overwrite, delete(final_flair); end
                    movefile(r_other, final_flair);
                else
                    warning('  [%s | %s] Resliced FLAIR not found -> check SPM output', sid, visit);
                end
            catch ME
                warning('  [%s | %s] Moving resliced FLAIR failed: %s', sid, visit, ME.message);
            end

            % --- Cleanup: ensure ONLY the final FLAIR remains in target FLAIR folder
            cleanup_flair_folder(flair_target_dir, final_flair, false);

            % --- Cleanup: remove resliced source T1w (r*.nii) created by SPM
            try
                [srcDir, srcBase, srcExt] = fileparts(t1_first);
                r_src_exact = fullfile(srcDir, ['r' srcBase srcExt]);
                if isfile(r_src_exact), delete(r_src_exact); end
                rList = dir(fullfile(srcDir, ['r' srcBase '*.nii'])); % broader safety
                for k=1:numel(rList)
                    f = fullfile(srcDir, rList(k).name);
                    if isfile(f), delete(f); end
                end
            catch ME
                warning('  [%s | %s] Cleanup r* T1w failed: %s', sid, visit, ME.message);
            end

            % remove temp original copy
            if isfile(tmp_other), delete(tmp_other); end

            fprintf('  [%s | %s] Created: %s (folder cleaned)\n', sid, visit, final_flair);
        end
    end
end

fprintf('\n=== Done. Log: %s ===\n', logFile);
if ~dryRun, diary off; end


%% ---------- local helper functions (allowed in scripts) ----------
function matlabbatch = build_coreg_batch(ref_t1w, src_t1w_first, other_flair_tmp)
% Create a single SPM "Coregister: Estimate & Reslice" job.
matlabbatch{1}.spm.spatial.coreg.estwrite.ref    = {[ref_t1w ',1']};
matlabbatch{1}.spm.spatial.coreg.estwrite.source = {[src_t1w_first ',1']};
matlabbatch{1}.spm.spatial.coreg.estwrite.other  = {[other_flair_tmp ',1']};
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep      = [4 2];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol      = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm     = [7 7];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp   = 4;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap     = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask     = 0;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix   = 'r';
end

function cleanup_flair_folder(flair_dir, final_flair, isDry)
% Delete everything in target FLAIR folder except the final coregistered file.
if ~isfolder(flair_dir), return; end
keepName = '';
if isfile(final_flair)
    [~, b, e] = fileparts(final_flair);
    keepName = [b e];
end
L = dir(fullfile(flair_dir,'*'));
for i=1:numel(L)
    if L(i).isdir, continue; end
    keep = ~isempty(keepName) && strcmpi(L(i).name, keepName);
    if ~keep
        tgt = fullfile(flair_dir, L(i).name);
        if isDry
            fprintf('      [DRY] Would delete: %s\n', tgt);
        else
            try, delete(tgt); catch, end
        end
    end
end
end
