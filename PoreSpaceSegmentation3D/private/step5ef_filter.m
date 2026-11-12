% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function [store_new, throat_wall_normalized_roughness, throat_center_new] = ...
    step5ef_filter(x1, store_new_in, thint, throat_wall_normalized_roughness, ...
                      throat_center_new, throat_center_uv, vis)
%STEP5EF_FILTER  XY boundary filtering for throat using periodic Savitzky–Golay.
%
% Returns store_new sized to the resampled/smoothed boundary.
%
% Output store_new columns (same as the existing):
%   [x y r theta dtheta]
%
% Roughness metric:
%   RMS distance (raw boundary -> SG-smoothed "roughness removed") normalized by Reff.

XYraw = double(store_new_in(:,1:2));
XYraw = XYraw(all(isfinite(XYraw),2),:);

if size(XYraw,1) < 10
    store_new = store_new_in;
    throat_center_new(x1,1:2) = throat_center_uv(1,1:2);
    throat_wall_normalized_roughness(x1,1) = 0;
    return;
end

% Ensure closed
if any(XYraw(1,:) ~= XYraw(end,:))
    XYraw = [XYraw; XYraw(1,:)];
end
XYraw = remove_consecutive_duplicates(XYraw);

% Arc length
ds = hypot(diff(XYraw(:,1)), diff(XYraw(:,2)));
s  = [0; cumsum(ds)];
L  = s(end);

if L < 1e-9 || numel(s) < 10
    store_new = store_new_in;
    throat_center_new(x1,1:2) = throat_center_uv(1,1:2);
    throat_wall_normalized_roughness(x1,1) = 0;
    return;
end

% Interpret thint as arc-length step; guard if tiny
if thint < 0.5
    target_ds = 1.0;     % tune 0.5–2.0
else
    target_ds = thint;
end

N  = max(60, round(L/target_ds));
su = linspace(0, L, N+1).';
xu = interp1(s, XYraw(:,1), su, 'linear');
yu = interp1(s, XYraw(:,2), su, 'linear');
xu(end) = xu(1); yu(end) = yu(1);

% ----------------------------
% Savitzky–Golay smoothing (periodic)
% ----------------------------
% Two scales:
%   (r) "roughness removed": small window, follows geometry closely
%   (m) "meso removed": larger window, used for centroid/mean shape

Nr = numel(xu);

w_r = max(7,  round(0.02 * Nr));   % was 0.03 with movmean; SG can be smaller
w_m = max(11, round(0.06 * Nr));   % was 0.08 with movmean

% enforce odd windows
if mod(w_r,2)==0, w_r = w_r + 1; end
if mod(w_m,2)==0, w_m = w_m + 1; end

% SG polynomial orders (must be < window length)
polyorder_r = 3;   % try 2 if you want even tighter adherence
polyorder_m = 3;

% Safety: ensure framelen > polyorder + 1
w_r = max(w_r, polyorder_r + 3); if mod(w_r,2)==0, w_r=w_r+1; end
w_m = max(w_m, polyorder_m + 3); if mod(w_m,2)==0, w_m=w_m+1; end

[x_r, y_r] = smooth_periodic_sgolay_xy(xu, yu, w_r, polyorder_r);
[x_m, y_m] = smooth_periodic_sgolay_xy(xu, yu, w_m, polyorder_m);

% Optional despike on both (usually only needed on x_r,y_r)
[x_r, y_r] = despike_periodic(x_r, y_r, max(7,w_r));
[x_m, y_m] = despike_periodic(x_m, y_m, max(7,w_m));

% ----------------------------
% Roughness metric: RMS(raw -> roughness-smoothed) / Reff
% ----------------------------
raw_pts = XYraw(1:end-1,:);                 % raw (exclude closure)
sm_pts  = [x_r(1:end-1), y_r(1:end-1)];     % smoothed curve samples

% Nearest neighbor mapping raw -> smoothed samples
try
    idx = knnsearch(sm_pts, raw_pts);
catch
    % Fallback without Stats toolbox
    idx = nearest_neighbor_fallback(sm_pts, raw_pts);
end

drr = hypot(raw_pts(:,1) - sm_pts(idx,1), raw_pts(:,2) - sm_pts(idx,2));

A = polyarea(x_r, y_r);
Reff = sqrt(max(A,0)/pi);
Reff = max(Reff, 1e-9);

rough_norm = rms(drr) / Reff;

% Store roughness
throat_wall_normalized_roughness(x1,1) = rough_norm;
throat_center_new(x1,7) = rough_norm;

% ----------------------------
% Update center from meso-smoothed curve
% ----------------------------
throat_center_xy = [mean(x_m(1:end-1)), mean(y_m(1:end-1))];
throat_center_new(x1,1:2) = throat_center_xy;

% ----------------------------
% Build output store_new (NEW size)
% ----------------------------
dxp = x_m - throat_center_uv(1);
dyp = y_m - throat_center_uv(2);
r  = hypot(dxp, dyp);
theta = atan2(dyp, dxp);
theta(theta < 0) = theta(theta < 0) + 2*pi;
dtheta = [0; diff(theta)];

store_new = [x_m, y_m, r, theta, dtheta];

% ----------------------------
% Visualization
% ----------------------------
if vis == 1
    %figure; hold on;
    %plot(XYraw(:,1), XYraw(:,2), 'k-', 'LineWidth', 1);
    %plot(x_r, y_r, 'g-', 'LineWidth', 2);
    plot(x_m, y_m, 'r-', 'LineWidth', 2);
    legend('Throat cloud','Raw boundary','Smoothed boundary');
    axis equal; grid on;
    %title(sprintf('Throat XY filter (SG periodic): roughness=%.4g', rough_norm));
    %legend('Raw boundary','Roughness removed (SG)','Meso removed (SG)');
end

end

% ---------- helpers ----------
function P = remove_consecutive_duplicates(P)
if isempty(P), return; end
d = hypot(diff(P(:,1)), diff(P(:,2)));
keep = [true; d > 0];
P = P(keep,:);
end

function [xs, ys] = smooth_periodic_sgolay_xy(x, y, framelen, polyorder)
% Periodic Savitzky–Golay smoothing for closed curves.
% x,y include closure point (last equals first).

x0 = x(1:end-1); 
y0 = y(1:end-1);
n  = numel(x0);

framelen = max(framelen, polyorder + 3);
if mod(framelen,2)==0, framelen = framelen + 1; end

halfW = floor(framelen/2);

xpad = [x0(end-halfW+1:end); x0; x0(1:halfW)];
ypad = [y0(end-halfW+1:end); y0; y0(1:halfW)];

xsm = sgolayfilt(xpad, polyorder, framelen);
ysm = sgolayfilt(ypad, polyorder, framelen);

xsm = xsm(halfW+1:halfW+n);
ysm = ysm(halfW+1:halfW+n);

xs = [xsm; xsm(1)];
ys = [ysm; ysm(1)];
end

function [xs, ys] = despike_periodic(xs, ys, w)
% Same as the prior version; kept for continuity.
x0 = xs(1:end-1); y0 = ys(1:end-1);

ddx = x0([2:end 1]) - 2*x0 + x0([end 1:end-1]);
ddy = y0([2:end 1]) - 2*y0 + y0([end 1:end-1]);
kappa = hypot(ddx, ddy);

thr = median(kappa) + 10*mad(kappa,1);
bad = kappa > max(thr, 1e-9);

if any(bad)
    n = numel(x0);
    halfW = floor(w/2);

    xpad = [x0(end-halfW+1:end); x0; x0(1:halfW)];
    ypad = [y0(end-halfW+1:end); y0; y0(1:halfW)];

    xmed = movmedian(xpad, w); xmed = xmed(halfW+1:halfW+n);
    ymed = movmedian(ypad, w); ymed = ymed(halfW+1:halfW+n);

    x0(bad) = xmed(bad);
    y0(bad) = ymed(bad);
end

xs = [x0; x0(1)];
ys = [y0; y0(1)];
end

function idx = nearest_neighbor_fallback(A, B)
% A: Mx2 candidates, B: Nx2 queries
% Returns idx (Nx1) of nearest rows in A for each row in B.
M = size(A,1);
N = size(B,1);
idx = ones(N,1);
for i = 1:N
    d2 = (A(:,1)-B(i,1)).^2 + (A(:,2)-B(i,2)).^2;
    [~,k] = min(d2);
    idx(i) = k;
end
end
