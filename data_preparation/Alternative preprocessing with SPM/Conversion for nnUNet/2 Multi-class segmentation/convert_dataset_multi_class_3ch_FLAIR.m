%% ========================================================================
% File: convert_dataset_multi_class_3ch_FLAIR.m
% Purpose:
%   Build nnU-Net-style dataset with THREE input channels and a 4-class label.
%   Channels per case:
%     _0000 = CBF_LICA      (left perfusion)
%     _0001 = CBF_RICA      (right perfusion)
%     _0002 = FLAIR (coreg) (from ssRICA/FLAIR_coreg/anon_rsub-pXXX_FLAIR.nii)
%   Label: uint16 with classes {0=bg, 1=left, 2=right, 3=overlap}
%
% Key steps (per case):
%   1) Use LICA CBF as geometry reference.
%   2) Create a clean, orthonormal 3×3 direction (no axis flips), keep origin;
%      optionally enforce spacing (FORCE_SPACING).
%   3) Data-driven L/R check: if the LICA mask’s center-of-mass (COM) lies on
%      the right half of the voxel grid, flip ALL arrays (LICA/RICA/FLAIR and
%      both masks) along the x dimension; headers unchanged.
%   4) Write .nii (uncompressed) into:
%        imagesTr/: *_0000.nii, *_0001.nii, *_0002.nii
%        labelsTr/: *.nii
%        imagesTs/: *_0000.nii, *_0001.nii, *_0002.nii
%
% Notes:
%   - No dataset.json or splits files are written here.
%   - Requires only MATLAB built-ins: niftiread, niftiinfo, niftiwrite.
%   - For nnU-Net spacing integrity, set FORCE_SPACING = [2.625 2.625 6.6]
%     (or [] to keep original LICA spacing per case).
% ========================================================================

%% ---------------- CONFIG ----------------
SRC_ROOT    = 'C:\Users\Vincent Wohlfarth\Data\anon_Data_250808';
TARGET_ROOT = 'C:\Users\Vincent Wohlfarth\Data\nnUNet_raw\Dataset001_PerfusionTerritories';

DIR_IMAGES_TR = fullfile(TARGET_ROOT,'imagesTr');
DIR_LABELS_TR = fullfile(TARGET_ROOT,'labelsTr');
DIR_IMAGES_TS = fullfile(TARGET_ROOT,'imagesTs');

OVERWRITE      = true;
RICA_THRESH    = 0.5;    % binarize RICA mask
LICA_THRESH    = 0.5;    % binarize LICA mask
FILE_ENDING    = '.nii';
VERBOSE_LABELS = false;

% spacing to enforce for outputs — set [] to keep LICA spacing
FORCE_SPACING  = [2.625, 2.625, 6.6];

FN_CBF        = 'CBF_3_BRmsk_CSF.nii';
FN_LICA_MASK  = 'mask_LICA_manual_Corrected.nii';
FN_RICA_MASK  = 'mask_RICA_manual_Corrected.nii';
% ssRICA FLAIR coreg: .../ssRICA/FLAIR_coreg/anon_rsub-pXXX_FLAIR.nii

%% --------------- HARD SPLIT TABLE ---------------
rows = {
'DATA_HC','sub-p001','First_visit','Training/Validation';
'DATA_HC','sub-p001','Second_visit','Training/Validation';
'DATA_HC','sub-p001','Third_visit','Training/Validation';
'DATA_HC','sub-p002','First_visit','Training/Validation';
'DATA_HC','sub-p002','Second_visit','Training/Validation';
'DATA_HC','sub-p002','Third_visit','Training/Validation';
'DATA_HC','sub-p003','First_visit','Training/Validation';
'DATA_HC','sub-p003','Second_visit','Training/Validation';
'DATA_HC','sub-p003','Third_visit','Training/Validation';
'DATA_HC','sub-p004','First_visit','Training/Validation';
'DATA_HC','sub-p004','Second_visit','Training/Validation';
'DATA_HC','sub-p004','Third_visit','Training/Validation';
'DATA_HC','sub-p005','First_visit','Training/Validation';
'DATA_HC','sub-p005','Second_visit','Training/Validation';
'DATA_HC','sub-p005','Third_visit','Training/Validation';
'DATA_HC','sub-p006','First_visit','Training/Validation';
'DATA_HC','sub-p006','Second_visit','Training/Validation';
'DATA_HC','sub-p006','Third_visit','Training/Validation';
'DATA_HC','sub-p007','First_visit','Training/Validation';
'DATA_HC','sub-p007','Second_visit','x';
'DATA_HC','sub-p007','Third_visit','Training/Validation';
'DATA_HC','sub-p008','First_visit','Training/Validation';
'DATA_HC','sub-p008','Second_visit','Training/Validation';
'DATA_HC','sub-p008','Third_visit','Training/Validation';
'DATA_HC','sub-p009','First_visit','Training/Validation';
'DATA_HC','sub-p009','Second_visit','Training/Validation';
'DATA_HC','sub-p009','Third_visit','Training/Validation';
'DATA_HC','sub-p010','First_visit','Training/Validation';
'DATA_HC','sub-p010','Second_visit','Training/Validation';
'DATA_HC','sub-p010','Third_visit','Training/Validation';
'DATA_HC','sub-p011','First_visit','Training/Validation';
'DATA_HC','sub-p011','Second_visit','Training/Validation';
'DATA_HC','sub-p011','Third_visit','Training/Validation';
'DATA_HC','sub-p012','First_visit','Training/Validation';
'DATA_HC','sub-p012','Second_visit','Training/Validation';
'DATA_HC','sub-p012','Third_visit','Training/Validation';
'DATA_HC','sub-p013','First_visit','Training/Validation';
'DATA_HC','sub-p013','Second_visit','Training/Validation';
'DATA_HC','sub-p013','Third_visit','Training/Validation';
'DATA_HC','sub-p014','First_visit','Testing';
'DATA_HC','sub-p014','Second_visit','Testing';
'DATA_HC','sub-p014','Third_visit','Testing';
'DATA_HC','sub-p015','First_visit','Testing';
'DATA_HC','sub-p015','Second_visit','Testing';
'DATA_HC','sub-p015','Third_visit','Testing';
'DATA_patients','sub-p016','First_visit','x';
'DATA_patients','sub-p016','Second_visit','x';
'DATA_patients','sub-p017','First_visit','Testing';
'DATA_patients','sub-p017','Second_visit','Testing';
'DATA_patients','sub-p018','First_visit','Testing';
'DATA_patients','sub-p018','Second_visit','Testing';
'DATA_patients','sub-p019','First_visit','Testing';
'DATA_patients','sub-p019','Second_visit','Testing';
'DATA_patients','sub-p020','First_visit','Testing';
'DATA_patients','sub-p020','Second_visit','Testing';
'DATA_patients','sub-p021','First_visit','x';
'DATA_patients','sub-p021','Second_visit','x';
'DATA_patients','sub-p022','First_visit','Testing';
'DATA_patients','sub-p022','Second_visit','Testing';
'DATA_patients','sub-p023','First_visit','Testing';
'DATA_patients','sub-p023','Second_visit','Testing';
};
tbl = cell2table(rows, 'VariableNames', {'Group','Subject','Visit','Split'});

%% --------------- PREP OUTPUT DIRS ---------------
ensure_dir(DIR_IMAGES_TR); ensure_dir(DIR_LABELS_TR); ensure_dir(DIR_IMAGES_TS);

%% --------------- PROCESS ---------------
n_ok_tr = 0; n_ok_ts = 0;

for i = 1:height(tbl)
    grp   = tbl.Group{i};
    sub   = tbl.Subject{i};
    visit = tbl.Visit{i};
    split = lower(tbl.Split{i});
    if strcmp(split,'x'), fprintf('[SKIP] %s | %s | %s -> x\n', grp, sub, visit); continue; end

    vnum = visit_number(visit); snum = subject_number(sub);
    if vnum==0 || isempty(snum), fprintf('[WARN] Bad entry -> %s | %s | %s\n', grp, sub, visit); continue; end
    caseId = sprintf('PerfTerr%03d-v%d', snum, vnum);

    base = fullfile(SRC_ROOT, grp, visit, 'output', sub, 'task-AIR', 'ASL');
    p_cbf_lica  = fullfile(base,'ssLICA','CBF_nativeSpace',FN_CBF);
    p_cbf_rica  = fullfile(base,'ssRICA','CBF_nativeSpace',FN_CBF);
    p_mask_lica = fullfile(base,'ssLICA','PerfTerrMask',FN_LICA_MASK);
    p_mask_rica = fullfile(base,'ssRICA','PerfTerrMask',FN_RICA_MASK);
    p_flair_rica = fullfile(base,'ssRICA','FLAIR_coreg', sprintf('anon_r%s_FLAIR.nii', sub)); % 3rd channel (FLAIR)

    if ~(isfile(p_cbf_lica) && isfile(p_cbf_rica) && isfile(p_mask_lica) && isfile(p_mask_rica) && isfile(p_flair_rica))
        fprintf('[MISS] %s: required file(s) missing\n', caseId); continue;
    end

    % --- Reference geometry from LICA CBF ---
    [img_lica, info_ref_raw] = read_nii_any(p_cbf_lica);
    ref_size = size(img_lica);
    info_ref = make_clean_ref_info_noFlip(info_ref_raw, FORCE_SPACING); % orthonormal, keep origin

    % RICA CBF + RICA FLAIR (size checks)
    [img_rica,   ~] = read_nii_any(p_cbf_rica);
    [img_flair,  ~] = read_nii_any(p_flair_rica);
    if ~isequal(size(img_rica), ref_size) || ~isequal(size(img_flair), ref_size)
        fprintf('[WARN] %s: RICA/FLAIR dims differ from LICA -> skipping\n', caseId); continue;
    end

    % Masks for LR decision
    [Lmask, ~] = read_nii_any(p_mask_lica);
    [Rmask, ~] = read_nii_any(p_mask_rica);
    if ~isequal(size(Lmask), ref_size) || ~isequal(size(Rmask), ref_size)
        fprintf('[WARN] %s: mask dims differ from ref -> skipping\n', caseId); continue;
    end

    % --- Data-driven L/R check using LICA mask COM ---
    flipLR = need_flip_left_right(Lmask);
    if flipLR
        img_lica  = flip(img_lica,  1);
        img_rica  = flip(img_rica,  1);
        img_flair = flip(img_flair, 1);
        Lmask     = flip(Lmask,     1);
        Rmask     = flip(Rmask,     1);
        if VERBOSE_LABELS
            fprintf('[INFO] %s -> applied LR flip (LICA COM on right)\n', caseId);
        end
    end

    switch split
        case 'training/validation'
            dst_stem = fullfile(DIR_IMAGES_TR, caseId);
            dst_img0 = sprintf('%s_%04d%s', dst_stem, 0, FILE_ENDING);
            dst_img1 = sprintf('%s_%04d%s', dst_stem, 1, FILE_ENDING);
            dst_img2 = sprintf('%s_%04d%s', dst_stem, 2, FILE_ENDING);
            dst_lbl  = fullfile(DIR_LABELS_TR, [caseId, FILE_ENDING]);

            if OVERWRITE || ~(isfile(dst_img0) && isfile(dst_img1) && isfile(dst_img2) && isfile(dst_lbl))
                write_nii_using_ref(img_lica,  info_ref, dst_img0);   % _0000 = LICA CBF
                write_nii_using_ref(img_rica,  info_ref, dst_img1);   % _0001 = RICA CBF
                write_nii_using_ref(img_flair, info_ref, dst_img2);   % _0002 = RICA FLAIR

                % combine masks -> 4-class label
                L = double(Lmask) > LICA_THRESH;
                R = double(Rmask) > RICA_THRESH;
                Y = zeros(ref_size,'uint16'); Y(L & ~R)=1; Y(R & ~L)=2; Y(L & R)=3;
                if VERBOSE_LABELS
                    fprintf('%s label counts: bg=%d L=%d R=%d OL=%d\n', ...
                        caseId, nnz(Y==0), nnz(Y==1), nnz(Y==2), nnz(Y==3));
                end
                write_nii_using_ref(Y, info_ref, dst_lbl, 'uint16');
            else
                fprintf('[HIT] %s exists -> skipped (overwrite=false)\n', caseId);
            end
            n_ok_tr = n_ok_tr + 1;

        case 'testing'
            dst_stem = fullfile(DIR_IMAGES_TS, caseId);
            dst_img0 = sprintf('%s_%04d%s', dst_stem, 0, FILE_ENDING);
            dst_img1 = sprintf('%s_%04d%s', dst_stem, 1, FILE_ENDING);
            dst_img2 = sprintf('%s_%04d%s', dst_stem, 2, FILE_ENDING);

            if OVERWRITE || ~(isfile(dst_img0) && isfile(dst_img1) && isfile(dst_img2))
                write_nii_using_ref(img_lica,  info_ref, dst_img0);
                write_nii_using_ref(img_rica,  info_ref, dst_img1);
                write_nii_using_ref(img_flair, info_ref, dst_img2);
            else
                fprintf('[HIT] %s exists (Ts) -> skipped (overwrite=false)\n', caseId);
            end
            n_ok_ts = n_ok_ts + 1;
    end
end

fprintf('\n--- DONE ---\nimagesTr/labelsTr cases: %d\nimagesTs cases: %d\n', n_ok_tr, n_ok_ts);

%% ================= LOCAL FUNCTIONS =================
function ensure_dir(p)
if ~isfolder(p), mkdir(p); end
end

function vnum = visit_number(s)
switch lower(s)
    case 'first_visit',  vnum = 1;
    case 'second_visit', vnum = 2;
    case 'third_visit',  vnum = 3;
    otherwise,           vnum = 0;
end
end

function snum = subject_number(s)
m = regexp(s, 'sub-p(\d+)$', 'tokens', 'once');
if isempty(m), snum = []; else, snum = str2double(m{1}); end
end

function [img, info] = read_nii_any(pathIn)
if endsWith(lower(pathIn), '.nii.gz')
    tmpDir = tempname; mkdir(tmpDir);
    gunzip(pathIn, tmpDir);
    f = dir(fullfile(tmpDir, '*.nii'));
    niiPath = fullfile(f(1).folder, f(1).name);
    info = niftiinfo(niiPath); img = niftiread(info);
    delete(niiPath); rmdir(tmpDir);
else
    info = niftiinfo(pathIn); img = niftiread(info);
end
end

function info_ref = make_clean_ref_info_noFlip(info_in, force_spacing)
% Clean header: orthonormal rotation close to input (preserve axis senses),
% keep origin; optionally set spacing.
info_ref = info_in;

% Affine (or identity)
if isfield(info_ref,'Transform') && ~isempty(info_ref.Transform) && ...
   isfield(info_ref.Transform,'T') && ~isempty(info_ref.Transform.T)
    T = double(info_ref.Transform.T);
else
    T = eye(4);
end
D = T(1:3,1:3);                % current 3x3

% Target spacing
if ~isempty(force_spacing)
    sp = double(force_spacing(:)');
else
    sp = double(info_ref.PixelDimensions(1:3));
    if any(sp==0)
        sp = [norm(D(:,1)) norm(D(:,2)) norm(D(:,3))];
        if any(sp==0), sp = [1 1 1]; end
    end
end

% Orthonormalize columns with sign preserved (Gram–Schmidt seeded by input)
u1 = D(:,1); if norm(u1)==0, u1=[1;0;0]; end; u1 = u1/norm(u1);
v2 = D(:,2) - (u1'*D(:,2))*u1; if norm(v2)<1e-8, v2 = [0;1;0]; end; u2 = v2/norm(v2);
u3 = cross(u1,u2); if dot(u3,D(:,3))<0, u3 = -u3; end  % align with original 3rd column

R = [u1 u2 u3];                 % orthonormal, preserves axis senses
Dnew = R * diag(sp);

T(1:3,1:3) = Dnew;              % keep origin T(1:3,4)
info_ref.Transform.T = T;
info_ref.PixelDimensions(1:3) = sp;
end

function flipLR = need_flip_left_right(leftMask)
% Decide if LICA mask sits on the right half of the voxel grid.
sz = size(leftMask);
if nnz(leftMask)==0
    flipLR = false; return;
end
[idxX,~,~] = ind2sub(sz, find(leftMask));
flipLR = mean(double(idxX)) > (sz(1)/2);
end

function write_nii_using_ref(img, info_ref, dst_nii, datatype)
% Write .nii using the provided reference geometry. Datatype optional.
[outDir,~,ext] = fileparts(dst_nii);
if ~isfolder(outDir), mkdir(outDir); end
if ~strcmpi(ext, '.nii'), dst_nii = [dst_nii '.nii']; end

info = info_ref;
if nargin >= 4 && ~isempty(datatype)
    targetClass = datatype;
else
    targetClass = class(img);
end
info.Datatype     = targetClass;
info.BitsPerPixel = bits_from_class(targetClass);
if ~strcmp(class(img), targetClass), img = cast(img, targetClass); end

niftiwrite(img, dst_nii, info, 'Compressed', false);
end

function b = bits_from_class(c)
switch c
    case {'uint8','int8'},            b = 8;
    case {'uint16','int16'},          b = 16;
    case {'uint32','int32','single'}, b = 32;
    case {'uint64','int64','double'}, b = 64;
    otherwise,                         b = 32;
end
end
