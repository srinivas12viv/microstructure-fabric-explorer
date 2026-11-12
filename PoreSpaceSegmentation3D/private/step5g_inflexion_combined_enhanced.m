% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [firstq2, secondq2, thirdq2, fourthq2, ...
          firstqc, secondqc, thirdqc, fourthqc] = ...
    step5g_inflexion_combined_enhanced(center, store_new1, ...
                                      throat_center_xy1, body1_center_xy1, body2_center_xy1, vis)
%STEP5G_INFLEXION_COMBINED_ENHANCED (UPDATED)
% Robust combined-boundary feature extraction:
%   - Inflexions: curvature sign-change (zero-crossings in signed curvature)
%   - Knees     : curvature-onset points (where bending starts and keeps increasing locally)
%
% Output format is compatible with the existing selection pipeline:
%   *q2 arrays are Nx3: [x y score]
%     - inflections: score = kappa_s (signed)
%     - knees:      score = sign(kappa_s)*abs(kappa_s) at onset index
%   Pass *q2(:,1:2) into correct_inflexion_* and prot_ratio routines.
%
% Inputs/Outputs: same as the existing version.

% -----------------------------
% 0) Boundary prep (periodic)
% -----------------------------
P = double(store_new1(:,1:2));
P = P(all(isfinite(P),2),:);

firstq2=[]; secondq2=[]; thirdq2=[]; fourthq2=[];
firstqc=[]; secondqc=[]; thirdqc=[]; fourthqc=[];

if size(P,1) < 10
    return;
end

% Ensure closed
if any(P(1,:) ~= P(end,:))
    P = [P; P(1,:)];
end
P = remove_consecutive_duplicates(P);

nClosed = size(P,1);
n = nClosed - 1;
if n < 10
    return;
end
P = P(1:end-1,:); % n x 2 periodic

throat_axis_center = double(center(1,1:2));

% -----------------------------
% 1) Signed curvature vs arc length (robust)
% -----------------------------
ds = hypot(P([2:end 1],1) - P(:,1), P([2:end 1],2) - P(:,2));
ds_med = median(ds(ds>0 & isfinite(ds)));
if isempty(ds_med) || ds_med <= 0, ds_med = 1.0; end

% First derivative wrt arc length (centered periodic)
dP   = P([2:end 1],:) - P([end 1:end-1],:);
ds_c = ds([end 1:end-1]) + ds;                 % ~2*local step length
dPds = dP ./ max(ds_c, eps);

% Second derivative wrt arc length (centered periodic)
d2P      = dPds([2:end 1],:) - dPds([end 1:end-1],:);
d2Pds2   = d2P ./ max(ds_c, eps);

x1 = dPds(:,1);  y1 = dPds(:,2);
x2 = d2Pds2(:,1); y2 = d2Pds2(:,2);

den   = (x1.^2 + y1.^2).^(3/2);
kappa = (x1.*y2 - y1.*x2) ./ max(den, eps);    % signed curvature

% Light periodic smoothing of curvature
w = max(5, round(0.03*n));
if mod(w,2)==0, w = w+1; end
kappa_s = smooth_periodic_1d(kappa, w);

% -----------------------------
% 2) Exclusion mask (prevent deep-bulge selections)
% -----------------------------
T  = double(throat_center_xy1(1,1:2));
B1 = double(body1_center_xy1(1,1:2));
B2 = double(body2_center_xy1(1,1:2));

u1 = (B1 - T);
u2 = (B2 - T);
if norm(u1) < 1e-12, u1 = [1 0]; end
if norm(u2) < 1e-12, u2 = [0 1]; end
u1 = u1(:); u2 = u2(:);

epsS = 1e-9;
exclude = false(n,1);
for i = 1:n
    exclude(i) = local_isBad(P(i,:), T, B1, B2, u1, u2, epsS);
end

% Robust curvature scale used in both inflection and knee gating
k_abs = abs(kappa_s(~exclude));
if isempty(k_abs)
    thr_k = 0;
else
    thr_k = median(k_abs) + 2.5*mad(k_abs,1);
end

% -----------------------------
% 3) INFLECTION candidates (kappa sign-change)
% -----------------------------
sg = sign(kappa_s);
sg(sg==0) = 1;

infCand = false(n,1);
for i = 1:n
    im = mod(i-2,n) + 1;
    ip = mod(i,n) + 1;
    if exclude(i) || exclude(im) || exclude(ip), continue; end
    if sg(im) ~= sg(ip)
        infCand(i) = true;
    end
end

% magnitude gating to suppress tiny sign flips
infCand = infCand & (abs(kappa_s) >= 0.10*thr_k);

% Debounce inflection candidates
minSep = max(3, round(0.02*n));
infIdx = enforce_min_separation(find(infCand), abs(kappa_s), n, minSep);

% -----------------------------
% 4) KNEE candidates (UPDATED): curvature-onset, not maxima
% -----------------------------
ka = abs(kappa_s);


% --- NEW: seed separately on upper and lower branches ---
yc = throat_axis_center(2);   % branch split line

upperMask = (P(:,2) >= yc) & ~exclude;
lowerMask = (P(:,2) <= yc) & ~exclude;

% If either side is empty (degenerate), fall back to all points
if ~any(upperMask), upperMask = ~exclude; end
if ~any(lowerMask), lowerMask = ~exclude; end

% Seed index: closest point to throat center on each branch
i0U = argmin_dist_mask(P, T, upperMask);
i0L = argmin_dist_mask(P, T, lowerMask);

%Lwin
%Window length (number of points) used to evaluate “curvature rise.”
%Bigger Lwin = more stable but later detection.
%Smaller Lwin = more sensitive and earlier detection (but noisier).

%noiseFloor
%Rejects low curvature magnitude.
%Higher noiseFloor = ignores early subtle bending → later.
%Lower noiseFloor = allows onset at small curvature → earlier.

%riseReq
%Requires the curvature to increase by some amount across the window.
%Higher riseReq = needs a large rise → later.
%Lower riseReq = accepts shallow ramp → earlier.

%fracIncReq
%Requires a fraction of increments to be positive (monotone-ish condition).
%1.0 means “every step increases” (too strict).
%Lowering it lets onset be detected with small local fluctuations.

% Same onset parameters you already used
%Lwin       = max(4, round(0.02*n));  Lwin = min(Lwin, 15);
Lwin = 4;
noiseFloor = 0.05 * thr_k;
riseReq    = 0.05 * thr_k;
fracIncReq = 0.6;

% For each seed, find onset toward +x (right) and -x (left),
% rather than boundary index direction. This is key.
kneeIdx = [];

kneeIdx = [kneeIdx;
           find_onset_toward_x(P, ka, exclude, i0U, +1, yc, Lwin, noiseFloor, riseReq, fracIncReq);
           find_onset_toward_x(P, ka, exclude, i0U, -1, yc, Lwin, noiseFloor, riseReq, fracIncReq);
           find_onset_toward_x(P, ka, exclude, i0L, +1, yc, Lwin, noiseFloor, riseReq, fracIncReq);
           find_onset_toward_x(P, ka, exclude, i0L, -1, yc, Lwin, noiseFloor, riseReq, fracIncReq)];

kneeIdx = unique(kneeIdx(kneeIdx>0));

% If onset detection found nothing (rare), fall back to one strong candidate near throat
if isempty(kneeIdx)
    % Choose the nearest non-excluded point with ka above noise floor within a local band
    % around BOTH seeds (upper + lower), then pick the strongest.
    seeds = unique([i0U, i0L]);
    candAll = [];

    for s = seeds
        if s <= 0 || s > n, continue; end
        band = unique(mod((s-10):(s+10), n) + 1);
        band = band(~exclude(band));
        band = band(ka(band) >= noiseFloor);

        candAll = [candAll; band(:)]; %#ok<AGROW>
    end

    candAll = unique(candAll);
    if ~isempty(candAll)
        [~, jj] = max(ka(candAll));
        kneeIdx = candAll(jj);
    else
        % last resort: pick max curvature among all non-excluded points
        ok = find(~exclude);
        if ~isempty(ok)
            [~, jj] = max(ka(ok));
            kneeIdx = ok(jj);
        end
    end
end

% -----------------------------
% 5) Merge feature points (inflection + knee), keep score in col 3
% -----------------------------
featIdx = unique([infIdx(:); kneeIdx(:)], 'stable');
%featIdx = kneeIdx(:); %Knees ONLYYY - Uncomment previous line if you want both
featPts   = P(featIdx,:);
featScore = kappa_s(featIdx);

% (optional) if you want to explicitly mark knees, keep them in debug only;
% downstream code should use only x,y for correct_inflexion_* anyway.

features = [featPts, featScore];  % Nx3

% -----------------------------
% 6) Split candidates into quadrants (Nx3 outputs)
% -----------------------------
for i = 1:size(features,1)
    x = features(i,1); y = features(i,2);
    if x >= throat_axis_center(1) && y >= throat_axis_center(2), firstq2  = [firstq2;  features(i,:)]; end
    if x <  throat_axis_center(1) && y >  throat_axis_center(2), secondq2 = [secondq2; features(i,:)]; end
    if x <  throat_axis_center(1) && y <  throat_axis_center(2), thirdq2  = [thirdq2;  features(i,:)]; end
    if x >= throat_axis_center(1) && y <= throat_axis_center(2), fourthq2 = [fourthq2; features(i,:)]; end
end

% -----------------------------
% 7) Build quadrant boundary sets (Nx2 outputs)
% -----------------------------
for i = 1:n
    x = P(i,1); y = P(i,2);
    if x >= throat_axis_center(1) && y >= throat_axis_center(2), firstqc  = [firstqc;  x y]; end
    if x <  throat_axis_center(1) && y >  throat_axis_center(2), secondqc = [secondqc; x y]; end
    if x <  throat_axis_center(1) && y <  throat_axis_center(2), thirdqc  = [thirdqc;  x y]; end
    if x >= throat_axis_center(1) && y <= throat_axis_center(2), fourthqc = [fourthqc; x y]; end
end

% Match the sorting convention (important for correct_inflexion_* logic)
if ~isempty(firstqc),  firstqc  = sortrows(firstqc, 1); end
if ~isempty(fourthqc), fourthqc = sortrows(fourthqc, 1); end
if ~isempty(secondqc), secondqc = sortrows(secondqc, 1, 'descend'); end
if ~isempty(thirdqc),  thirdqc  = sortrows(thirdqc, 1, 'descend'); end

% -----------------------------
% 8) Visualization
% -----------------------------
if vis == 1
    hold on;

    % excluded region
    scatter(P(exclude,1), P(exclude,2), 10, 'y', 'filled');

    % inflections
    if ~isempty(infIdx)
        scatter(P(infIdx,1), P(infIdx,2), 60, 'm', 'filled');
    end

    % knees (curvature onset)
    if ~isempty(kneeIdx)
        scatter(P(kneeIdx,1), P(kneeIdx,2), 60, 'c', 'filled');
    end

    scatter(B1(1),B1(2), 70, 'b', 'filled');
    scatter(B2(1),B2(2), 70, 'b', 'filled');
    scatter(throat_axis_center(1), throat_axis_center(2), 70, 'g', 'filled');
end

end

% ======================= helpers =======================
function P = remove_consecutive_duplicates(P)
d = hypot(diff(P(:,1)), diff(P(:,2)));
keep = [true; d > 0];
P = P(keep,:);
end

function xsm = smooth_periodic_1d(x, w)
x = x(:);
n = numel(x);
halfW = floor(w/2);
xpad = [x(end-halfW+1:end); x; x(1:halfW)];
xsm = movmean(xpad, w);
xsm = xsm(halfW+1:halfW+n);
end

function idx_out = enforce_min_separation(idx_in, score, n, minSep)
% keep strongest candidates with circular index separation >= minSep
if isempty(idx_in), idx_out = idx_in; return; end
[~, ord] = sort(score(idx_in), 'descend');
idx_sorted = idx_in(ord);

chosen = [];
keep = false(size(idx_sorted));
for k = 1:numel(idx_sorted)
    i = idx_sorted(k);
    if isempty(chosen) || all(circ_dist(i, chosen, n) > minSep)
        keep(k) = true;
        chosen(end+1) = i; %#ok<AGROW>
    end
end
idx_out = sort(idx_sorted(keep));
end

function d = circ_dist(i, J, n)
d = abs(J - i);
d = min(d, n - d);
end

function bad = local_isBad(P,T,B1,B2,u1,u2,epsS)
% true if P is on the "wrong side" for BOTH body axes
sT1 = (P - T)  * u1(:);
sB1 = (P - B1) * u1(:);
sT2 = (P - T)  * u2(:);
sB2 = (P - B2) * u2(:);

if abs(sT1)<epsS, sT1=0; end
if abs(sB1)<epsS, sB1=0; end
if abs(sT2)<epsS, sT2=0; end
if abs(sB2)<epsS, sB2=0; end

bad1 = (sign(sT1)==sign(sB1)) && (sign(sT1)~=0) && (sign(sB1)~=0);
bad2 = (sign(sT2)==sign(sB2)) && (sign(sT2)~=0) && (sign(sB2)~=0);

bad = bad1 && bad2;
end

function i0 = argmin_dist_mask(P, T, mask)
idx = find(mask);
d = hypot(P(idx,1)-T(1), P(idx,2)-T(2));
[~,j] = min(d);
i0 = idx(j);
end

function idx = find_onset_toward_x(P, ka, exclude, i0, xdir, yc, Lwin, noiseFloor, riseReq, fracIncReq)
%FIND_ONSET_TOWARD_X_MID  Fractional-knee version between onset and peak.
%
% Returns a "knee" located at a fraction kFrac (default 0.5) of the
% arc-index distance from:
%   - i_on: first curvature onset
%   - i_pk: peak curvature further toward bulge (same step direction)
%
% To customize: set kFrac below (e.g., 0.25 or 0.4).

% ---------------- USER TUNING ----------------
kFrac = 0.5;   % 0.50=midpoint, 0.25=closer to onset, 0.40=between
% --------------------------------------------

% Guard
kFrac = max(0.0, min(1.0, kFrac));

n = size(P,1);

% ---- Branch mask: stay on same side of yc as seed ----
seedUpper = (P(i0,2) >= yc);
if seedUpper
    branchOk = @(i) (P(i,2) >= yc);
else
    branchOk = @(i) (P(i,2) <= yc);
end

% ---- Define x-side gating relative to seed x ----
x0 = P(i0,1);
xOk = @(i) (xdir > 0 && P(i,1) > x0) || (xdir < 0 && P(i,1) < x0);

% ---- 1) Find onset candidates in both index directions ----
cand_on = [];
cand_step = [];

for step = [+1, -1]
    maxWalk = floor(n/2);
    for t = 0:maxWalk
        i = mod(i0 - 1 + step*t, n) + 1;
        if exclude(i), continue; end
        if ~branchOk(i), continue; end
        if ~xOk(i), continue; end
        if ka(i) < noiseFloor, continue; end

        win = zeros(Lwin,1);
        for k = 1:Lwin
            win(k) = mod(i - 1 + step*k, n) + 1;
        end

        if mean(exclude(win)) > 0.35, continue; end
        if mean(arrayfun(branchOk, win)) < 0.70, continue; end

        kw = ka([i; win]);
        dkw = diff(kw);

        fracInc   = mean(dkw > 0);
        totalRise = kw(end) - kw(1);

        if fracInc >= fracIncReq && totalRise >= riseReq
            cand_on   = [cand_on; i];       %#ok<AGROW>
            cand_step = [cand_step; step];  %#ok<AGROW>
            break;
        end
    end
end

if isempty(cand_on)
    idx = 0;
    return;
end

% Choose nearer onset if both directions found it
d = hypot(P(cand_on,1)-P(i0,1), P(cand_on,2)-P(i0,2));
[~,jbest] = min(d);
i_on  = cand_on(jbest);
step0 = cand_step(jbest);

% ---- 2) Find peak curvature after onset along step0 ----
maxWalk2 = min(floor(n/2), max(20, round(0.35*n)));

bestKa = -inf;
i_pk = i_on;
stall = 0;
stallMax = max(8, round(0.02*n));

for t = 0:maxWalk2
    i = mod(i_on - 1 + step0*t, n) + 1;

    if exclude(i), continue; end
    if ~branchOk(i), continue; end
    if ~xOk(i), continue; end

    if ka(i) > bestKa
        bestKa = ka(i);
        i_pk = i;
        stall = 0;
    else
        stall = stall + 1;
        if stall >= stallMax
            break;
        end
    end
end

if i_pk == i_on
    idx = i_on;
    return;
end

% ---- 3) Fractional position between onset and peak ----
t_pk = forward_steps(i_on, i_pk, n, step0);

% Choose t at kFrac of that distance:
% - kFrac=0 gives onset
% - kFrac=0.5 gives midpoint
% - kFrac=1 gives peak
t_sel = round(kFrac * t_pk);

% Ensure within [0, t_pk]
t_sel = max(0, min(t_pk, t_sel));

idx = mod(i_on - 1 + step0*t_sel, n) + 1;

end

function t = forward_steps(i_start, i_end, n, step)
% number of steps along "step" direction from i_start to i_end (0..n-1)
if step > 0
    t = mod(i_end - i_start, n);
else
    t = mod(i_start - i_end, n);
end
end

