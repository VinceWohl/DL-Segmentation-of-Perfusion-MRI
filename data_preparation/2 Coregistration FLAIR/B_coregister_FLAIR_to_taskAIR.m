%% B_coregister_FLAIR_to_taskAIR.m
% Coregister each visit's FLAIR into task-AIR (ssLICA/ssRICA) space via T1w -> T1w_coreg.
% Writes to ...\task-AIR\ASL\<hemi>\FLAIR_coreg\anon_rsub-<pID>_FLAIR.nii
% Keeps ONLY the final file in FLAIR_coreg.

%% -------------------- USER SETTINGS --------------------
rootDir      = 'C:\Users\Vincent Wohlfarth\Data\anon_Data_250808';
cohorts      = {'DATA_HC','DATA_patients'};
visits       = {'First_visit','Second_visit','Third_visit'};
hemis        = {'ssLICA','ssRICA'};
overwrite    = false;
dryRun       = false;
logFile      = fullfile(rootDir, ['B_coregister_FLAIR_to_taskAIR_log_' datestr(now,'yyyymmdd_HHMMSS') '.txt']);
%% -------------------------------------------------------

if ~dryRun, diary(logFile); diary on; end
fprintf('\n=== Coregister FLAIR to task-AIR (SPM) ===\nRoot: %s\nOverwrite: %d | Dry-run: %d\n\n', rootDir, overwrite, dryRun);

spm('defaults','FMRI'); spm_jobman('initcfg');

for c = 1:numel(cohorts)
    cohort = cohorts{c};
    for v = 1:numel(visits)
        visit = visits{v};
        visitOut = fullfile(rootDir, cohort, visit, 'output');
        if ~isfolder(visitOut)
            fprintf('[%s | %s] visit folder missing -> skip\n', cohort, visit); continue;
        end

        d = dir(fullfile(visitOut,'sub-p*')); subs = {d([d.isdir]).name};

        for s = 1:numel(subs)
            sid = subs{s};                     % e.g. 'sub-p001'
            idShort = regexprep(sid,'^sub-','');  % -> 'p001'

            flair_src = fullfile(visitOut, sid, 'FLAIR', ['anon_' sid '_FLAIR.nii']);
            t1_src    = fullfile(visitOut, sid, 'T1w',   ['anon_' sid '_T1w.nii']);
            if ~isfile(flair_src), fprintf('  [%s | %s | %s] FLAIR missing -> SKIP\n', cohort, visit, sid); continue; end
            if ~isfile(t1_src),    fprintf('  [%s | %s | %s] T1w missing   -> SKIP\n', cohort, visit, sid); continue; end

            for h = 1:numel(hemis)
                hemi = hemis{h};
                ref_t1_coreg = fullfile(visitOut, sid, 'task-AIR','ASL',hemi,'T1w_coreg', ['anon_rsub-' idShort '_T1w.nii']);
                out_dir      = fullfile(visitOut, sid, 'task-AIR','ASL',hemi,'FLAIR_coreg');
                out_file     = fullfile(out_dir, ['anon_rsub-' idShort '_FLAIR.nii']);

                if ~isfile(ref_t1_coreg)
                    % helpful debug: list what's actually there
                    dbg = dir(fullfile(visitOut, sid, 'task-AIR','ASL',hemi,'T1w_coreg','*.nii'));
                    fprintf('  [%s | %s | %s | %s] T1w_coreg missing -> SKIP hemi\n', cohort, visit, sid, hemi);
                    if ~isempty(dbg)
                        fprintf('      Found instead: %s\n', strjoin(string({dbg.name}), ', '));
                    end
                    continue;
                end

                if isfile(out_file) && ~overwrite
                    cleanup_folder_keep_only(out_dir, out_file, dryRun);
                    fprintf('  [%s | %s | %s | %s] Output exists -> SKIP (cleanup ok)\n', cohort, visit, sid, hemi);
                    continue;
                end

                if dryRun
                    fprintf('  [DRY | %s | %s | %s | %s]\n', cohort, visit, sid, hemi);
                    fprintf('      REF: %s\n      SRC: %s\n      OTH: %s\n      OUT: %s\n', ref_t1_coreg, t1_src, flair_src, out_file);
                    continue;
                end

                if ~isfolder(out_dir), mkdir(out_dir); end

                % Copy FLAIR to target folder as temp "other", so SPM writes there
                [~, flairBase, flairExt] = fileparts(flair_src);
                tmp_other = fullfile(out_dir, ['tmp_' flairBase flairExt]);
                try, copyfile(flair_src, tmp_other); catch ME
                    warning('  [%s | %s | %s | %s] Copy FLAIR failed: %s', cohort, visit, sid, hemi, ME.message); continue;
                end

                % Build & run SPM batch
                matlabbatch = build_coreg_batch(ref_t1_coreg, t1_src, tmp_other); %#ok<NASGU>
                try
                    spm_jobman('run', matlabbatch);
                catch ME
                    warning('  [%s | %s | %s | %s] SPM job failed: %s', cohort, visit, sid, hemi, ME.message);
                    if isfile(tmp_other), delete(tmp_other); end
                    continue;
                end

                % Move resliced FLAIR to final name
                r_other = fullfile(out_dir, ['r' 'tmp_' flairBase flairExt]);
                if ~isfile(r_other)
                    alt = dir(fullfile(out_dir, 'rtmp_*.nii'));
                    if numel(alt)==1, r_other = fullfile(out_dir, alt(1).name); end
                end
                try
                    if isfile(r_other)
                        if isfile(out_file) && overwrite, delete(out_file); end
                        movefile(r_other, out_file);
                    else
                        warning('  [%s | %s | %s | %s] Resliced FLAIR not found -> check SPM output', cohort, visit, sid, hemi);
                    end
                catch ME
                    warning('  [%s | %s | %s | %s] Moving resliced FLAIR failed: %s', cohort, visit, sid, hemi, ME.message);
                end

                % Cleanup: keep only final file in FLAIR_coreg
                cleanup_folder_keep_only(out_dir, out_file, false);

                % Cleanup: r*.nii next to T1w
                try
                    [t1Dir, t1Base, t1Ext] = fileparts(t1_src);
                    r_exact = fullfile(t1Dir, ['r' t1Base t1Ext]); if isfile(r_exact), delete(r_exact); end
                    rList = dir(fullfile(t1Dir, ['r' t1Base '*.nii']));
                    for kk = 1:numel(rList), f = fullfile(t1Dir, rList(kk).name); if isfile(f), delete(f); end, end
                catch ME
                    warning('  [%s | %s | %s | %s] Cleanup r* T1w failed: %s', cohort, visit, sid, hemi, ME.message);
                end

                if isfile(tmp_other), delete(tmp_other); end
                fprintf('  [%s | %s | %s | %s] Created: %s (folder cleaned)\n', cohort, visit, sid, hemi, out_file);
            end
        end
    end
end

fprintf('\n=== Done. Log: %s ===\n', logFile);
if ~dryRun, diary off; end

%% -------------------- local helpers --------------------
function matlabbatch = build_coreg_batch(ref_img, src_img, other_img)
matlabbatch{1}.spm.spatial.coreg.estwrite.ref    = {[ref_img ',1']};
matlabbatch{1}.spm.spatial.coreg.estwrite.source = {[src_img ',1']};
matlabbatch{1}.spm.spatial.coreg.estwrite.other  = {[other_img ',1']};
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.sep      = [4 2];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.tol      = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{1}.spm.spatial.coreg.estwrite.eoptions.fwhm     = [7 7];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.interp   = 4;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.wrap     = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.mask     = 0;
matlabbatch{1}.spm.spatial.coreg.estwrite.roptions.prefix   = 'r';
end

function cleanup_folder_keep_only(folderPath, keepFile, isDry)
if ~isfolder(folderPath), return; end
keepName = '';
if isfile(keepFile), [~, b, e] = fileparts(keepFile); keepName = [b e]; end
L = dir(fullfile(folderPath,'*'));
for i=1:numel(L)
    if L(i).isdir, continue; end
    if isempty(keepName) || ~strcmpi(L(i).name, keepName)
        tgt = fullfile(folderPath, L(i).name);
        if isDry, fprintf('      [DRY] Would delete: %s\n', tgt); else, try, delete(tgt); catch, end, end
    end
end
end