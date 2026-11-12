% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [firstq,secondq,thirdq,fourthq, ...
          firstq1,secondq1,thirdq1,fourthq1, ...
          throat_axis_center] = step5g_inflexion(store_new, surface_profile_xy1, vis)
%STEP5G_INFLEXION  (DROP-IN, ROBUST)
% Finds candidate feature points on THROAT boundary in 4 quadrants.
%
% This version returns candidates = (inflection points) UNION (curvature-onset knee points).
% - Inflections: sign-change in signed curvature (kappa)
% - Knees: onset of curvature growth (bending initiation), seeded on upper/lower branches
%          and detected left/right -> yields up to 4 knee points.
%
% Output signature is identical to the existing step5g_inflexion:
%   firstq..fourthq : candidate points in each quadrant (Nx2)
%   firstq1..fourthq1 : all boundary points in each quadrant (Nx2)  [fallback sets]
%   throat_axis_center : [mean(x), mean(y)] of the boundary
%
% Inputs:
%   store_new           : ordered throat boundary (Nx2 or Nx>=2)
%   surface_profile_xy1 : surface points (Mx2) that should be excluded by distance-band
%   vis                 : 1 to plot, else 0

% -----------------------------
% 0) Sanitize / periodic boundary
% -----------------------------
P = double(store_new(:,1:2));
P = P(all(isfinite(P),2),:);

firstq=[]; secondq=[]; thirdq=[]; fourthq=[];
firstq1=[]; secondq1=[]; thirdq1=[]; fourthq1=[];

if size(P,1) < 12
    throat_axis_center = [mean(P(:,1)), mean(P(:,2))];
    return;
end

% Remove consecutive duplicates
P = remove_consecutive_duplicates(P);

% Ensure closed loop for periodic ops (but keep P as n x 2 open periodic array)
if any(P(1,:) ~= P(end,:))
    Pclosed = [P; P(1,:)];
else
    Pclosed = P;
end

nClosed = size(Pclosed,1);
n = nClosed - 1;
if n < 12
    throat_axis_center = [mean(P(:,1)), mean(P(:,2))];
    return;
end
P = Pclosed(1:end-1,:);  % n x 2 periodic representation

throat_axis_center = [mean(P(:,1)), mean(P(:,2))];
xc = throat_axis_center(1);
yc = throat_axis_center(2);

% -----------------------------
% 1) Surface-band exclusion (distance based)
% -----------------------------
exclude = false(n,1);

% typical step length along boundary
ds_step = hypot(P([2:end 1],1)-P(:,1), P([2:end 1],2)-P(:,2));
ds_med  = median(ds_step(ds_step>0 & isfinite(ds_step)));
if isempty(ds_med) || ds_med <= 0, ds_med = 1.0; end

if ~isempty(surface_profile_xy1)
    S = double(surface_profile_xy1(:,1:2));
    S = S(all(isfinite(S),2),:);
    if size(S,1) >= 3
        kdt = KDTreeSearcher(S);
        [~, dmin] = knnsearch(kdt, P);   % distance to nearest surface point
        dsTol = 1.5 * ds_med;            % band thickness ~1-2 boundary samples
        exclude = (dmin <= dsTol);
    end
end

% -----------------------------
% 2) Signed curvature kappa(s) with periodic smoothing
% -----------------------------
ds = ds_step;                                 % forward step distances
dP   = P([2:end 1],:) - P([end 1:end-1],:);   % centered diff
ds_c = ds([end 1:end-1]) + ds;                % ~2*local step length
dPds = dP ./ max(ds_c, eps);

d2P    = dPds([2:end 1],:) - dPds([end 1:end-1],:);
d2Pds2 = d2P ./ max(ds_c, eps);

x1 = dPds(:,1);  y1 = dPds(:,2);
x2 = d2Pds2(:,1); y2 = d2Pds2(:,2);

den   = (x1.^2 + y1.^2).^(3/2);
kappa = (x1.*y2 - y1.*x2) ./ max(den, eps);   % signed curvature

w = max(5, round(0.03*n));
if mod(w,2)==0, w = w+1; end
kappa_s = smooth_periodic_1d(kappa, w);
ka = abs(kappa_s);

% Robust curvature scale (noise)
k_abs = ka(~exclude);
if isempty(k_abs)
    thr_k = 0;
else
    thr_k = median(k_abs) + 2.5*mad(k_abs,1);
end

% -----------------------------
% 3) INFLECTION candidates (sign change in kappa_s)
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

% Magnitude gate to suppress tiny sign flips
infCand = infCand & (ka >= 0.10*thr_k);

% Debounce inflections (avoid dense clusters)
minSepInf = max(3, round(0.02*n));
infIdx = enforce_min_separation(find(infCand), ka, n, minSepInf);

% -----------------------------
% 4) KNEE points (curvature-onset, branch seeded: upper/lower × left/right)
% -----------------------------
%Lwin       = max(4, round(0.02*n));  Lwin = min(Lwin, 15);

%noiseFloor = 0.50 * thr_k;
%riseReq    = 0.10 * thr_k;
%fracIncReq = 2.0;
Lwin = 4;
%Control theses for better predictions
noiseFloor = 0.05 * thr_k;
riseReq    = 0.05 * thr_k;
fracIncReq = 0.6;

% If thr_k is ~0 (almost straight), knees not meaningful
kneeIdx = [];
if thr_k > 0
    upperMask = (P(:,2) >= yc) & ~exclude;
    lowerMask = (P(:,2) <= yc) & ~exclude;
    if ~any(upperMask), upperMask = ~exclude; end
    if ~any(lowerMask), lowerMask = ~exclude; end

    % Seed near the axis center (not body centers), for throat-only geometry
    i0U = argmin_dist_mask(P, throat_axis_center, upperMask);
    i0L = argmin_dist_mask(P, throat_axis_center, lowerMask);

    kneeIdx = [kneeIdx;
               find_onset_toward_x(P, ka, exclude, i0U, +1, yc, xc, Lwin, noiseFloor, riseReq, fracIncReq);
               find_onset_toward_x(P, ka, exclude, i0U, -1, yc, xc, Lwin, noiseFloor, riseReq, fracIncReq);
               find_onset_toward_x(P, ka, exclude, i0L, +1, yc, xc, Lwin, noiseFloor, riseReq, fracIncReq);
               find_onset_toward_x(P, ka, exclude, i0L, -1, yc, xc, Lwin, noiseFloor, riseReq, fracIncReq)];
    kneeIdx = unique(kneeIdx(kneeIdx>0));
end

% -----------------------------
% 5) Combine features: (inflections ∪ knees)
% -----------------------------
featIdx = unique([infIdx(:); kneeIdx(:)], 'stable');
%featIdx = kneeIdx(:); %Knees ONLYYY - Uncomment previous line if you want both
featPts = P(featIdx,:);  % Nx2

% -----------------------------
% 6) Split feature points into quadrants (return Nx2 like original)
% -----------------------------
for i = 1:size(featPts,1)
    x = featPts(i,1); y = featPts(i,2);
    if x >= xc && y >= yc, firstq  = [firstq;  x y]; end
    if x <  xc && y >  yc, secondq = [secondq; x y]; end
    if x <  xc && y <  yc, thirdq  = [thirdq;  x y]; end
    if x >= xc && y <= yc, fourthq = [fourthq; x y]; end
end

% -----------------------------
% 7) Quadrant populations (fallback sets) — identical intent to original
% -----------------------------
for i = 1:n
    x = P(i,1); y = P(i,2);
    if x >= xc && y >= yc, firstq1  = [firstq1;  x y]; end
    if x <  xc && y >  yc, secondq1 = [secondq1; x y]; end
    if x <  xc && y <  yc, thirdq1  = [thirdq1;  x y]; end
    if x >= xc && y <= yc, fourthq1 = [fourthq1; x y]; end
end

% Sorting convention (consistent with the correct_inflexion logic)
if ~isempty(firstq),  firstq  = sortrows(firstq, 1); end
if ~isempty(fourthq), fourthq = sortrows(fourthq, 1); end
if ~isempty(secondq), secondq = sortrows(secondq, 1, 'descend'); end
if ~isempty(thirdq),  thirdq  = sortrows(thirdq, 1, 'descend'); end

if ~isempty(firstq1),  firstq1  = sortrows(firstq1, 1); end
if ~isempty(fourthq1), fourthq1 = sortrows(fourthq1, 1); end
if ~isempty(secondq1), secondq1 = sortrows(secondq1, 1, 'descend'); end
if ~isempty(thirdq1),  thirdq1  = sortrows(thirdq1, 1, 'descend'); end

% -----------------------------
% 8) Visualization
% -----------------------------
if vis == 1
    figure;hold on;

    % boundary
    plot([P(:,1);P(1,1)], [P(:,2);P(1,2)], 'b-', 'LineWidth', 2);

    % excluded band
    if any(exclude)
        scatter(P(exclude,1), P(exclude,2), 10, 'y', 'filled');
    end

    % inflections
    if ~isempty(infIdx)
        scatter(P(infIdx,1), P(infIdx,2), 60, 'm', 'filled');
    end

    % knees (onset)
    if ~isempty(kneeIdx)
        scatter(P(kneeIdx,1), P(kneeIdx,2), 60, 'c', 'filled');
    end

    scatter(throat_axis_center(1), throat_axis_center(2), 70, 'g', 'filled');
    legend('Smoothed boundary', 'Excluded', 'Inflection points','Knee points');
    title('Throat boundary features: inflections (magenta) + knees-onset (cyan); excluded band (yellow)');
    axis equal; grid on;
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

function i0 = argmin_dist_mask(P, C, mask)
idx = find(mask);
d = hypot(P(idx,1)-C(1), P(idx,2)-C(2));
[~,j] = min(d);
i0 = idx(j);
end

function idx = find_onset_toward_x(P, ka, exclude, i0, xdir, yc, xc, Lwin, noiseFloor, riseReq, fracIncReq)
%FIND_ONSET_TOWARD_X  Fractional-knee version between onset and peak.
%
% Returns a point located at a fraction kFrac of the arc-index distance between:
%   - i_on: curvature-onset index
%   - i_pk: peak curvature index further toward the bulge in xdir
%
% Keeps the same signature as the current function.

% ---------------- USER TUNING ----------------
kFrac = 0.50;   % 0.50=midpoint, 0.25=closer to onset, 0.40=40% from onset, 1.0=peak
% --------------------------------------------
kFrac = max(0.0, min(1.0, kFrac));  % clamp

n = size(P,1);

% -----------------------------
% Branch constraint (upper/lower) based on seed
% -----------------------------
seedUpper = (P(i0,2) >= yc);
if seedUpper
    branchOk = @(i) (P(i,2) >= yc);
else
    branchOk = @(i) (P(i,2) <= yc);
end

% Half-plane constraint (left/right) based on xdir
if xdir > 0
    halfplaneOk = @(i) (P(i,1) > xc);
else
    halfplaneOk = @(i) (P(i,1) < xc);
end

% -----------------------------
% 1) Find onset candidates (same as the existing logic)
% -----------------------------
candOn = [];
candStep = [];   % store which boundary-walk step (+1/-1) produced the onset
for step = [+1, -1]   % walk boundary both index directions
    maxWalk = floor(n/2);
    for t = 0:maxWalk
        i = mod(i0 - 1 + step*t, n) + 1;

        if exclude(i), continue; end
        if ~branchOk(i), continue; end
        if ~halfplaneOk(i), continue; end
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
            candOn   = [candOn; i];        %#ok<AGROW>
            candStep = [candStep; step];   %#ok<AGROW>
            break;
        end
    end
end

if isempty(candOn)
    idx = 0;
    return;
end

% Choose onset closest (Euclidean) to seed
dOn = hypot(P(candOn,1)-P(i0,1), P(candOn,2)-P(i0,2));
[~,jOn] = min(dOn);
i_on    = candOn(jOn);
step_on = candStep(jOn);

% -----------------------------
% 2) Find peak curvature index beyond onset (toward bulge)
% -----------------------------
maxWalkPk = floor(n/4);   % conservative; increase if needed
x_on = P(i_on,1);

bestKa = -Inf;
i_pk = 0;

for t = 1:maxWalkPk
    ii = mod(i_on - 1 + step_on*t, n) + 1;

    if exclude(ii), continue; end
    if ~branchOk(ii), continue; end
    if ~halfplaneOk(ii), continue; end

    % Ensure we move "toward bulge" in the chosen x-direction:
    if xdir > 0
        if P(ii,1) < x_on, continue; end
    else
        if P(ii,1) > x_on, continue; end
    end

    if ka(ii) > bestKa
        bestKa = ka(ii);
        i_pk = ii;
    end
end

% If no peak found, fall back to onset itself
if i_pk == 0
    idx = i_on;
    return;
end

% -----------------------------
% 3) Fractional index along circular path from i_on -> i_pk
% -----------------------------
d = circ_forward_dist(i_on, i_pk, n, step_on);  % steps from onset to peak along step_on

% Select fraction of that distance (0 onset, 0.5 mid, 1 peak)
t_sel = round(kFrac * d);

% Ensure within [0, d]
t_sel = max(0, min(d, t_sel));

idx = mod(i_on - 1 + step_on*t_sel, n) + 1;

end

% ---- local helper ----
function d = circ_forward_dist(iA, iB, n, step)
% Distance (in index steps) moving from iA to iB using step (+1 or -1)
if step > 0
    d = mod((iB - iA), n);
else
    d = mod((iA - iB), n);
end
end


