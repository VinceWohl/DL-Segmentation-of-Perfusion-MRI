%% A_downsample_ssLICA.m  (downsample LICA + post-pass clamp for all task-AIR files)
% Part 1: In-plane (2D) downsample ssLICA from 96x96 -> 80x80, z preserved.
%         - T1/FLAIR/CBF: bilinear + antialias, clamp <0 to 0, write float32 (pinfo=[1;0;0])
%         - PerfTerrMask: nearest, write uint8 (pinfo=[1;0;0])
%         - Gate: skip downsampling for a subject/visit if either perfusion mask is missing
%
% Part 2 (NEW): After downsampling, clamp any negative voxels to 0 in
%               *all* task-AIR files (ssLICA + ssRICA, all four subfolders).
%               - Intensity images -> float32, pinfo=[1;0;0]
%               - Masks            -> uint8,  pinfo=[1;0;0]
%
% Requires: SPM on path (spm_vol/spm_read_vols/spm_write_vol)
%           Image Processing Toolbox (imresize)

%% -------------------- SETTINGS --------------------
rootDir = 'C:\Users\Vincent Wohlfarth\Data\anon_Data_250808';
dryRun  = false;   % true = print-only for both parts
%% --------------------------------------------------

if exist('spm_vol','file')~=2 || exist('spm_write_vol','file')~=2
    error('SPM not found on path. addpath(<your_spm_folder>) and retry.');
end
if exist('imresize','file')~=2
    error('imresize not found. Install/enable Image Processing Toolbox.');
end

fprintf('\n=== In-plane downsample ssLICA to 80x80 (2D, z preserved) ===\nRoot: %s | Dry-run: %d\n\n', rootDir, dryRun);

cohorts = {'DATA_HC','DATA_patients'};
visits  = {'First_visit','Second_visit','Third_visit'};

mods = { ...
  struct('subdir','T1w_coreg',      'fnameTpl','anon_rsub-%s_T1w.nii',    'method','bilinear', 'forceNonNeg',true,  'outFloat',true), ...
  struct('subdir','FLAIR_coreg',    'fnameTpl','anon_rsub-%s_FLAIR.nii',  'method','bilinear', 'forceNonNeg',true,  'outFloat',true), ...
  struct('subdir','CBF_nativeSpace','fnameTpl','CBF_3_BRmsk_CSF.nii',     'method','bilinear', 'forceNonNeg',true,  'outFloat',true), ...
  struct('subdir','PerfTerrMask',   'fnameTpl','mask_LICA_manual_Corrected.nii','method','nearest','forceNonNeg',false,'outFloat',false) ...
};

targetXY = [80 80];

nTried=0; nDone=0; nSkip=0; nMaskGate=0; nErr=0;

%% ---------- PART 1: Downsample ssLICA ----------
for ci=1:numel(cohorts)
  for vi=1:numel(visits)
    visitOut = fullfile(rootDir, cohorts{ci}, visits{vi}, 'output');
    if ~isfolder(visitOut)
      fprintf('[%s | %s] visit folder missing -> skip\n', cohorts{ci}, visits{vi});
      continue;
    end

    d = dir(fullfile(visitOut,'sub-p*')); subs = {d([d.isdir]).name};

    for si=1:numel(subs)
      sid = subs{si};                       % 'sub-p001'
      idShort = regexprep(sid,'^sub-','');  % 'p001'

      % Gate: require both perfusion masks (for this subject/visit)
      licaMask = fullfile(visitOut, sid, 'task-AIR','ASL','ssLICA','PerfTerrMask','mask_LICA_manual_Corrected.nii');
      ricaMask = fullfile(visitOut, sid, 'task-AIR','ASL','ssRICA','PerfTerrMask','mask_RICA_manual_Corrected.nii');
      if ~isfile(licaMask) || ~isfile(ricaMask)
        nMaskGate = nMaskGate + 1;
        fprintf('SKIP (mask missing) [%s | %s | %s]: LICA=%d RICA=%d\n', ...
           cohorts{ci}, visits{vi}, sid, isfile(licaMask), isfile(ricaMask));
        continue;
      end

      for mi=1:numel(mods)
        M = mods{mi};
        licaDir = fullfile(visitOut, sid, 'task-AIR','ASL','ssLICA', M.subdir);

        % Build source filename
        if contains(M.fnameTpl,'%s')
            fileName = sprintf(M.fnameTpl, idShort);
        else
            fileName = M.fnameTpl;
        end
        src = fullfile(licaDir, fileName);

        if ~isfile(src)
          nSkip = nSkip + 1;
          fprintf('SKIP missing src: %s\n', src);
          continue;
        end

        % Load header & skip non-96x96 or already 80x80
        try
          V = spm_vol(src);
        catch ME
          nErr=nErr+1; fprintf(2,'ERR spm_vol: %s\n -> %s\n', src, ME.message); continue;
        end
        if numel(V)~=1
          nSkip=nSkip+1; fprintf('SKIP (4D?): %s\n', src); continue;
        end
        if all(V.dim(1:2) == targetXY)
          nSkip = nSkip + 1; continue;  % already 80x80
        end
        if ~(V.dim(1)==96 && V.dim(2)==96)
          nSkip = nSkip + 1; fprintf('SKIP (not 96x96): %s [%dx%dx%d]\n', src, V.dim); continue;
        end

        % Read image data
        try
          I = spm_read_vols(V);   % double
        catch ME
          nErr=nErr+1; fprintf(2,'ERR spm_read_vols: %s\n -> %s\n', src, ME.message); continue;
        end

        % 2D in-plane resize per slice
        [nx, ny, nz] = deal(targetXY(1), targetXY(2), V.dim(3));
        J = zeros(nx, ny, nz, 'like', I);
        for z=1:nz
          if strcmp(M.method,'nearest')
            J(:,:,z) = imresize(I(:,:,z), [nx ny], 'nearest');
          else
            J(:,:,z) = imresize(I(:,:,z), [nx ny], 'bilinear', 'Antialiasing', true);
          end
        end
        if M.forceNonNeg
          J(J<0) = 0;   % clamp any residual negatives
        end

        if dryRun
          fprintf('[DRY] %s | %s | %s | %s : [%dx%dx%d] -> [80x80x%d]\n', ...
            cohorts{ci}, visits{vi}, sid, M.subdir, V.dim, nz);
          continue;
        end

        % Update header: scale in-plane voxel vectors, preserve world center
        try
          Vout = V;
          Vout.dim = [nx ny nz];

          sx = V.dim(1)/nx;   % = 96/80
          sy = V.dim(2)/ny;   % = 96/80

          M0 = V.mat;
          M1 = M0;
          M1(:,1) = M0(:,1) * sx;
          M1(:,2) = M0(:,2) * sy;

          c_old = ([V.dim(1:2) V.dim(3)] + 1) / 2;   % voxel center
          c_new = ([nx ny nz] + 1) / 2;
          w_old = M0 * [c_old 1]';
          M1(:,4) = w_old - M1(:,1)*c_new(1) - M1(:,2)*c_new(2) - M1(:,3)*c_new(3);

          Vout.mat = M1;

          % Write with explicit datatype & no scaling
          if M.outFloat
            Vout.dt    = [16 0];      % float32
            Vout.pinfo = [1;0;0];
            J = single(J);
            typeStr = 'float32';
          else
            Vout.dt    = [2 0];       % uint8 for masks
            Vout.pinfo = [1;0;0];
            J = uint8(round(J));
            typeStr = 'uint8';
          end

          spm_write_vol(Vout, J);
          fprintf('DONE: %s  [96x96 -> 80x80, z preserved, %s]\n', src, typeStr);
          nDone = nDone + 1; nTried = nTried + 1;

        catch ME
          nErr=nErr+1; fprintf(2,'ERR write/update: %s\n -> %s\n', src, ME.message);
        end
      end
    end
  end
end

fprintf('\n=== Downsample Summary ===\nTried: %d | Resampled: %d | Skipped: %d | Mask-gated skips: %d | Errors: %d\n', ...
    nTried, nDone, nSkip, nMaskGate, nErr);

%% ---------- PART 2: Post-pass clamp (all task-AIR files, both hemis) ----------
fprintf('\n=== Post-pass: clamp negatives to 0 for ALL task-AIR files (ssLICA + ssRICA) ===\n');

hemiList = {'ssLICA','ssRICA'};
subdirs  = {'T1w_coreg','FLAIR_coreg','CBF_nativeSpace','PerfTerrMask'};

nFiles=0; nClamped=0; nSkip2=0; nErr2=0;

for ci=1:numel(cohorts)
  for vi=1:numel(visits)
    visitOut = fullfile(rootDir, cohorts{ci}, visits{vi}, 'output');
    if ~isfolder(visitOut), continue; end

    d = dir(fullfile(visitOut,'sub-p*')); subs = {d([d.isdir]).name};

    for si=1:numel(subs)
      sid = subs{si};

      for hi=1:numel(hemiList)
        hemi = hemiList{hi};
        base = fullfile(visitOut, sid, 'task-AIR','ASL',hemi);

        for sj=1:numel(subdirs)
          sd = fullfile(base, subdirs{sj});
          if ~isfolder(sd), continue; end

          N = dir(fullfile(sd,'*.nii'));
          for k=1:numel(N)
            fn = fullfile(sd, N(k).name);
            nFiles = nFiles + 1;

            % Decide if this is a mask (PerfTerrMask path or mask_ filename)
            isMask = contains(lower(sd), 'perfterrmask') || contains(lower(N(k).name),'mask');

            try
              Vc = spm_vol(fn);
              if numel(Vc) ~= 1
                nSkip2 = nSkip2 + 1; continue;   % skip 4D
              end
              Ic = spm_read_vols(Vc);  % double, already scaled

              minVal = min(Ic(:));
              if ~(minVal < 0)
                nSkip2 = nSkip2 + 1; continue;   % nothing to clamp
              end

              if dryRun
                fprintf('[DRY CLAMP] %s  (min %.6g -> 0)\n', fn, minVal);
                continue;
              end

              % Clamp
              Ic(Ic<0) = 0;

              % Write back with explicit types & no scaling
              Vout = Vc;
              Vout.pinfo = [1;0;0];
              if isMask
                Vout.dt = [2 0];                % uint8
                Jw = uint8(round(Ic));
              else
                Vout.dt = [16 0];               % float32
                Jw = single(Ic);
              end

              spm_write_vol(Vout, Jw);
              fprintf('CLAMPED: %s  (min %.6g -> 0)\n', fn, minVal);
              nClamped = nClamped + 1;

            catch ME
              nErr2 = nErr2 + 1;
              fprintf(2,'ERR clamp: %s\n -> %s\n', fn, ME.message);
            end
          end
        end
      end
    end
  end
end

fprintf('\n=== Clamp Summary ===\nFiles seen: %d | Clamped: %d | Skipped: %d | Errors: %d\nDone.\n', ...
    nFiles, nClamped, nSkip2, nErr2);
