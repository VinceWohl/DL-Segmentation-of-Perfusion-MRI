%% conversion_for_thresholding.m
% Build nnU-Net dataset for thresholding (no split file).
% Include a subject/visit ONLY if both perfusion masks exist (ssLICA & ssRICA).
% Copy per-hemisphere:
%   - image:  CBF_3_BRmsk_CSF.nii  -> imagesTr/PerfTerrXXX-vY-Z_0000.nii
%   - label:  mask_<LICA/RICA>_manual_Corrected.nii -> labelsTr/PerfTerrXXX-vY-Z.nii
%
% Source is NEVER modified (only copyfile to destination).
%
% Required folder structure under srcRoot matches your cleaned dataset.

%% ------------------- USER SETTINGS -------------------
srcRoot = 'C:\Users\Vincent Wohlfarth\Data\anon_Data_250808';
dstRoot = 'C:\Users\Vincent Wohlfarth\Data\nnUNet_raw\Dataset001_PerfusionTerritories';

overwrite = false;      % true: overwrite existing targets
dryRun    = false;      % true: print actions only, no copies

CBF_FILE  = 'CBF_3_BRmsk_CSF.nii';  % change if you truly meant CBV
%% ------------------------------------------------------

% Prepare output folders (training only)
imagesTr = fullfile(dstRoot, 'imagesTr'); if ~dryRun && ~isfolder(imagesTr), mkdir(imagesTr); end
labelsTr = fullfile(dstRoot, 'labelsTr'); if ~dryRun && ~isfolder(labelsTr), mkdir(labelsTr); end

fprintf('\n=== nnU-Net conversion (thresholding) ===\nSource: %s\nDest  : %s\nOverwrite: %d | Dry-run: %d\n\n', ...
    srcRoot, dstRoot, overwrite, dryRun);

cohorts   = {'DATA_HC','DATA_patients'};
visits    = {'First_visit','Second_visit','Third_visit'};
visit2num = containers.Map({'First_visit','Second_visit','Third_visit'}, {'1','2','3'});

nCase=0; nPairCopied=0; nSkip=0; nErr=0;

for ci = 1:numel(cohorts)
  for vi = 1:numel(visits)
    visitOut = fullfile(srcRoot, cohorts{ci}, visits{vi}, 'output');
    if ~isfolder(visitOut)
      fprintf('[%s | %s] visit folder missing -> skip\n', cohorts{ci}, visits{vi}); 
      continue; 
    end

    D = dir(fullfile(visitOut,'sub-p*')); D = D([D.isdir]);

    for si = 1:numel(D)
      sid = D(si).name;                 % e.g., 'sub-p001'
      subjDigits = regexprep(sid, '^sub-p', '');   % '001'
      vnum = visit2num(visits{vi});

      % ---- Gate: require BOTH masks (LICA & RICA) for this subject/visit ----
      maskLICA = fullfile(visitOut, sid, 'task-AIR','ASL','ssLICA','PerfTerrMask','mask_LICA_manual_Corrected.nii');
      maskRICA = fullfile(visitOut, sid, 'task-AIR','ASL','ssRICA','PerfTerrMask','mask_RICA_manual_Corrected.nii');
      if ~(isfile(maskLICA) && isfile(maskRICA))
        nSkip = nSkip + 1;
        fprintf('SKIP (missing one/both masks) [%s | %s | %s]  LICA:%d RICA:%d\n', ...
           cohorts{ci}, visits{vi}, sid, isfile(maskLICA), isfile(maskRICA));
        continue;
      end
      % ----------------------------------------------------------------------

      nCase = nCase + 1;

      % Process both hemispheres now that we know both masks exist
      hems = {'ssLICA','ssRICA'};
      for hi = 1:numel(hems)
        hemi = hems{hi};
        Z    = iff(strcmp(hemi,'ssRICA'),'R','L');

        base = fullfile(visitOut, sid, 'task-AIR','ASL',hemi);

        imgSrc = fullfile(base, 'CBF_nativeSpace', CBF_FILE);
        if strcmp(hemi,'ssRICA')
          labSrc = fullfile(base, 'PerfTerrMask', 'mask_RICA_manual_Corrected.nii');
        else
          labSrc = fullfile(base, 'PerfTerrMask', 'mask_LICA_manual_Corrected.nii');
        end

        if ~isfile(imgSrc) || ~isfile(labSrc)
          nSkip = nSkip + 1;
          fprintf('SKIP missing image/label: %s | %s\n', imgSrc, labSrc);
          continue;
        end

        stem       = sprintf('PerfTerr%s-v%s-%s', subjDigits, vnum, Z);
        imgDstName = [stem '_0000.nii'];
        labDstName = [stem '.nii'];

        imgDst = fullfile(imagesTr, imgDstName);
        labDst = fullfile(labelsTr, labDstName);

        try
          ok1 = doCopy(imgSrc, imgDst, overwrite, dryRun);
          ok2 = doCopy(labSrc, labDst, overwrite, dryRun);
          if ok1 && ok2
            nPairCopied = nPairCopied + 1;
          else
            nSkip = nSkip + 1;
          end
        catch ME
          nErr = nErr + 1;
          fprintf(2,'ERROR copying [%s | %s | %s]\n  -> %s\n', sid, visits{vi}, hemi, ME.message);
        end
      end % hemi
    end % subject
  end
end

fprintf('\n=== Summary ===\nSubject/visit cases considered: %d\nPairs copied (hemi-level): %d\nSkipped: %d\nErrors: %d\nDone.\n', ...
    nCase, nPairCopied, nSkip, nErr);


%% ----------------- local helpers -----------------
function out = iff(cond, a, b), if cond, out=a; else, out=b; end
end

function tf = doCopy(src, dst, overwrite, dryRun)
% Copy file with overwrite control. Returns true if copied (or already present and overwrite==false).
    tf = false;
    if ~overwrite && isfile(dst)
        % already exists and we don't overwrite
        return;
    end
    if dryRun
        fprintf('[DRY] COPY\n  from: %s\n  to  : %s\n', src, dst);
        tf = true;
        return;
    end
    % ensure parent exists
    outDir = fileparts(dst);
    if ~isfolder(outDir), mkdir(outDir); end
    copyfile(src, dst);
    tf = true;
end
