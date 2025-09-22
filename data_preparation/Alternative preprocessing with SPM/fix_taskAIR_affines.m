function fix_taskAIR_affines(root_dir, spm_path)
% fix_taskAIR_affines(root_dir, spm_path)
% Make origin & direction (affine) consistent within each case & hemisphere
% by copying the affine from the CBF_nativeSpace image to all task-AIR
% derivatives (PerfTerrMask, ASL_fullcoreg, FLAIR_coreg, T1w_coreg).
%
% Example:
%   fix_taskAIR_affines('D:\Data\anon_DATA_250919', 'D:\Code\DLSegPerf\spm')
%
% Notes:
% - This overwrites the headers directly, no backup is created.
% - Only headers are changed, voxel data remains untouched.

    if nargin > 1 && ~isempty(spm_path)
        if exist(fullfile(spm_path,'spm.m'),'file'), addpath(spm_path); end
    end
    assert(exist('spm_vol','file')==2, 'SPM not found on MATLAB path.');

    groups = {'DATA_HC','DATA_patients'};
    visits = {'First_visit','Second_visit','Third_visit'};
    hemis  = {'ssLICA','ssRICA'};
    maskName = containers.Map(hemis, ...
        {'mask_LICA_manual_Corrected.nii','mask_RICA_manual_Corrected.nii'});

    nCases = 0; nRefMiss = 0; nFixed = 0; nSkip = 0; nErr = 0;

    for g = 1:numel(groups)
        for v = 1:numel(visits)
            subj_root = fullfile(root_dir, groups{g}, visits{v}, 'output');
            if ~isfolder(subj_root), continue; end

            d = dir(fullfile(subj_root, 'sub-p*'));
            for s = 1:numel(d)
                if ~d(s).isdir, continue; end
                subj_dir = fullfile(subj_root, d(s).name);

                for h = 1:numel(hemis)
                    hemi = hemis{h};
                    hemi_base = fullfile(subj_dir, 'task-AIR', 'ASL', hemi);

                    ref_path = fullfile(hemi_base, 'CBF_nativeSpace', 'CBF_3_BRmsk_CSF.nii');
                    if ~isfile(ref_path)
                        nRefMiss = nRefMiss + 1;
                        fprintf('[MISS REF] %s | %s | %s | %s\n', ...
                            groups{g}, d(s).name, visits{v}, hemi);
                        continue;
                    end

                    % candidate targets
                    tgt_paths = {
                        fullfile(hemi_base, 'PerfTerrMask', maskName(hemi))
                        guess_first(fullfile(hemi_base, 'ASL_fullcoreg'), 'anon_meano_*.nii')
                        guess_first(fullfile(hemi_base, 'FLAIR_coreg'),  'anon_rsub-*_FLAIR.nii')
                        guess_first(fullfile(hemi_base, 'T1w_coreg'),    'anon_rsub-*_T1w.nii')
                    };

                    nCases = nCases + 1;

                    for k = 1:numel(tgt_paths)
                        tgt = tgt_paths{k};
                        if isempty(tgt) || ~isfile(tgt)
                            nSkip = nSkip + 1;
                            continue;
                        end
                        try
                            changed = copy_affine_from_ref(ref_path, tgt);
                            if changed, nFixed = nFixed + 1; else, nSkip = nSkip + 1; end
                        catch ME
                            nErr = nErr + 1;
                            fprintf(2, '[ERROR] %s\n  -> %s\n', tgt, ME.message);
                        end
                    end
                end
            end
        end
    end

    fprintf('\n===== Summary =====\n');
    fprintf('Cases visited:             %d\n', nCases);
    fprintf('Reference missing:         %d\n', nRefMiss);
    fprintf('Files fixed (header set):  %d\n', nFixed);
    fprintf('Files skipped (ok/miss):   %d\n', nSkip);
    fprintf('Errors:                    %d\n', nErr);
end

function tgt = guess_first(folder, pattern)
    tgt = '';
    if ~isfolder(folder), return; end
    L = dir(fullfile(folder, pattern));
    if ~isempty(L)
        L = sort_nat(L);
        tgt = fullfile(folder, L(1).name);
    end
end

function L = sort_nat(L)
    [~, idx] = sort_nat_helper({L.name});
    L = L(idx);
end

function [cs,index] = sort_nat_helper(c)
    [~, index] = sort(lower(c));
    cs = c(index);
end

function changed = copy_affine_from_ref(ref_path, tgt_path)
    tol = 1e-6;
    Vref = spm_vol(ref_path);
    Vtgt = spm_vol(tgt_path);

    if max(abs(Vref.mat(:) - Vtgt.mat(:))) < tol
        changed = false;
        return;
    end

    Y = spm_read_vols(Vtgt); % keep voxel data
    Vnew = Vtgt;
    Vnew.mat = Vref.mat;     % overwrite affine
    spm_write_vol(Vnew, Y);

    fprintf('[FIXED] %s <- %s\n', tgt_path, ref_path);
    changed = true;
end
