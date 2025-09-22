%% convert_dataset_multi_label_4ch.m
% Outputs (.nii, uncompressed):
%   imagesTr: _0000 = CBF_LICA
%             _0001 = CBF_RICA
%             _0002 = T1w_coreg from ssRICA
%             _0003 = FLAIR_coreg from ssRICA
%   labelsTr: 4D NIfTI with two channels (uint8):
%             channel 1 = left mask (LICA), channel 2 = right mask (RICA)
%   imagesTs: channels only (no labels)
% Strategy:
%   - Use LICA CBF as reference; enforce spacing if FORCE_SPACING set.
%   - If LICA mask COM lies on RIGHT half of the voxel grid, flip ALL arrays along x (dim 1).

%% ---------------- CONFIG ----------------
SRC_ROOT    = 'C:\Users\Vincent Wohlfarth\Data\anon_Data_250808';
TARGET_ROOT = 'C:\Users\Vincent Wohlfarth\Data\nnUNet_raw\Dataset001_PerfusionTerritories';

DIR_IMAGES_TR = fullfile(TARGET_ROOT,'imagesTr');
DIR_LABELS_TR = fullfile(TARGET_ROOT,'labelsTr');
DIR_IMAGES_TS = fullfile(TARGET_ROOT,'imagesTs');

OVERWRITE      = true;
RICA_THRESH    = 0.5;
LICA_THRESH    = 0.5;
FILE_ENDING    = '.nii';
VERBOSE_LABELS = true;

% spacing to enforce for outputs — set [] to keep LICA spacing
FORCE_SPACING  = [2.625, 2.625, 6.6];

FN_CBF        = 'CBF_3_BRmsk_CSF.nii';
FN_LICA_MASK  = 'mask_LICA_manual_Corrected.nii';
FN_RICA_MASK  = 'mask_RICA_manual_Corrected.nii';

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
    p_cbf_lica   = fullfile(base,'ssLICA','CBF_nativeSpace',FN_CBF);
    p_cbf_rica   = fullfile(base,'ssRICA','CBF_nativeSpace',FN_CBF);
    p_mask_lica  = fullfile(base,'ssLICA','PerfTerrMask',FN_LICA_MASK);
    p_mask_rica  = fullfile(base,'ssRICA','PerfTerrMask',FN_RICA_MASK);
    p_t1w_rica   = fullfile(base,'ssRICA','T1w_coreg',   coreg_t1w_name(sub));
    p_flair_rica = fullfile(base,'ssRICA','FLAIR_coreg', coreg_flair_name(sub));

    if ~(isfile(p_cbf_lica) && isfile(p_cbf_rica) && isfile(p_mask_lica) && isfile(p_mask_rica) ...
         && isfile(p_t1w_rica) && isfile(p_flair_rica))
        fprintf('[MISS] %s: required file(s) missing\n', caseId); continue;
    end

    % --- Reference geometry from LICA CBF ---
    [img_lica, info_ref_raw] = read_nii_any(p_cbf_lica);
    ref_size = size(img_lica);
    info_ref = make_clean_ref_info_noFlip(info_ref_raw, FORCE_SPACING);

    % Other images
    [img_rica,  ~] = read_nii_any(p_cbf_rica);
    [img_t1w,   ~] = read_nii_any(p_t1w_rica);
    [img_flair, ~] = read_nii_any(p_flair_rica);
    if ~isequal(size(img_rica), ref_size) || ~isequal(size(img_t1w), ref_size) || ~isequal(size(img_flair), ref_size)
        fprintf('[WARN] %s: one or more channel dims differ from LICA -> skipping\n', caseId); continue;
    end

    % Masks (used to decide LR flip)
    [Lmask, ~] = read_nii_any(p_mask_lica);
    [Rmask, ~] = read_nii_any(p_mask_rica);
    if ~isequal(size(Lmask), ref_size) || ~isequal(size(Rmask), ref_size)
        fprintf('[WARN] %s: mask dims differ from ref -> skipping\n', caseId); continue;
    end

    % --- Decide if left-right flip is needed (based on LICA mask COM) ---
    flipLR = need_flip_left_right(Lmask);

    % If needed, flip ALL arrays along x (dim 1). Header stays unchanged.
    if flipLR
        img_lica  = flip(img_lica,  1);
        img_rica  = flip(img_rica,  1);
        img_t1w   = flip(img_t1w,   1);
        img_flair = flip(img_flair, 1);
        Lmask     = flip(Lmask,     1);
        Rmask     = flip(Rmask,     1);
        if VERBOSE_LABELS
            fprintf('[INFO] %s -> applied LR flip based on LICA mask COM.\n', caseId);
        end
    end

    switch split
        case 'training/validation'
            dst_stem = fullfile(DIR_IMAGES_TR, caseId);
            dst_img0 = sprintf('%s_%04d%s', dst_stem, 0, FILE_ENDING);
            dst_img1 = sprintf('%s_%04d%s', dst_stem, 1, FILE_ENDING);
            dst_img2 = sprintf('%s_%04d%s', dst_stem, 2, FILE_ENDING);
            dst_img3 = sprintf('%s_%04d%s', dst_stem, 3, FILE_ENDING);
            dst_lbl  = fullfile(DIR_LABELS_TR, [caseId, FILE_ENDING]);

            if OVERWRITE || ~(isfile(dst_img0) && isfile(dst_img1) && isfile(dst_img2) && isfile(dst_img3) && isfile(dst_lbl))
                % Write inputs
                write_nii_using_ref(img_lica,  info_ref, dst_img0);   % _0000 = LICA CBF
                write_nii_using_ref(img_rica,  info_ref, dst_img1);   % _0001 = RICA CBF
                write_nii_using_ref(img_t1w,   info_ref, dst_img2);   % _0002 = RICA T1w_coreg
                write_nii_using_ref(img_flair, info_ref, dst_img3);   % _0003 = RICA FLAIR_coreg

                % Threshold masks and stack as channels
                L = uint8(double(Lmask) > LICA_THRESH);
                R = uint8(double(Rmask) > RICA_THRESH);
                Y2 = cat(4, L, R); % 4D label, 2 channels

                if VERBOSE_LABELS
                    nL  = nnz(L);
                    nR  = nnz(R);
                    nOL = nnz(L & R);
                    fprintf('[LBL] %s: voxels L=%d, R=%d, overlap=%d\n', caseId, nL, nR, nOL);
                end
                write_nii_using_ref(Y2, info_ref, dst_lbl, 'uint8');
            else
                fprintf('[HIT] %s exists -> skipped (overwrite=false)\n', caseId);
            end
            n_ok_tr = n_ok_tr + 1;

        case 'testing'
            dst_stem = fullfile(DIR_IMAGES_TS, caseId);
            dst_img0 = sprintf('%s_%04d%s', dst_stem, 0, FILE_ENDING);
            dst_img1 = sprintf('%s_%04d%s', dst_stem, 1, FILE_ENDING);
            dst_img2 = sprintf('%s_%04d%s', dst_stem, 2, FILE_ENDING);
            dst_img3 = sprintf('%s_%04d%s', dst_stem, 3, FILE_ENDING);

            if OVERWRITE || ~(isfile(dst_img0) && isfile(dst_img1) && isfile(dst_img2) && isfile(dst_img3))
                write_nii_using_ref(img_lica,  info_ref, dst_img0);
                write_nii_using_ref(img_rica,  info_ref, dst_img1);
                write_nii_using_ref(img_t1w,   info_ref, dst_img2);
                write_nii_using_ref(img_flair, info_ref, dst_img3);
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

function fn = coreg_t1w_name(sub)
% 'anon_rsub-pXXX_T1w.nii'
fn = ['anon_r' sub '_T1w.nii'];
end

function fn = coreg_flair_name(sub)
% 'anon_rsub-pXXX_FLAIR.nii'
fn = ['anon_r' sub '_FLAIR.nii'];
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
% Clean header: orthonormal rotation close to input (no sign flipping),
% keep axis senses and origin; optionally set spacing.
info_ref = info_in;

% Affine (or identity)
if isfield(info_ref,'Transform') && ~isempty(info_ref.Transform) && ...
   isfield(info_ref.Transform,'T') && ~isempty(info_ref.Transform.T)
    T = double(info_ref.Transform.T);
else
    T = eye(4);
end
D = T(1:3,1:3);

% Target spacing
if ~isempty(force_spacing)
    sp = double(force_spacing(:))';
else
    sp = double(info_ref.PixelDimensions(1:3));
    if any(sp==0)
        sp = [norm(D(:,1)) norm(D(:,2)) norm(D(:,3))];
        if any(sp==0), sp = [1 1 1]; end
    end
end

% Orthonormalize columns with sign preserved (Gram–Schmidt)
u1 = D(:,1); if norm(u1)==0, u1=[1;0;0]; end; u1 = u1/norm(u1);
v2 = D(:,2) - (u1'*D(:,2))*u1; if norm(v2)<1e-8, v2 = [0;1;0]; end; u2 = v2/norm(v2);
u3 = cross(u1,u2); if dot(u3,D(:,3))<0, u3 = -u3; end
R = [u1 u2 u3];
Dnew = R * diag(sp);

T(1:3,1:3) = Dnew;
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
[outDir,~,ext] = fileparts(dst_nii);
if ~isfolder(outDir), mkdir(outDir); end
if ~strcmpi(ext, '.nii'), dst_nii = [dst_nii '.nii']; end

info = info_ref;

% Ensure header matches the data dimensionality
info.ImageSize = size(img);
if isfield(info,'PixelDimensions') && ~isempty(info.PixelDimensions)
    pd = double(info.PixelDimensions(:))';
else
    pd = [1 1 1];
end
nd = ndims(img);
if numel(pd) < nd
    pd = [pd, ones(1, nd - numel(pd))];
elseif numel(pd) > nd
    pd = pd(1:nd);
end
info.PixelDimensions = pd;

% Datatype
if nargin >= 4 && ~isempty(datatype)
    targetClass = datatype;
else
    targetClass = class(img);
end
info.Datatype     = targetClass;
info.BitsPerPixel = bits_from_class(targetClass);

% Cast & write
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
