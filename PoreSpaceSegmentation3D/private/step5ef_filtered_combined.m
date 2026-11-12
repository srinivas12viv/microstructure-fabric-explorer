% Upstream: Nimisha Roy, Pore_Space_Architecture (MIT; see LICENSE.upstream).
% Thesis adaptations supplied by Srinivas Vivek Bokkisa; see docs/MODIFICATIONS.md.
function store_new_combined = step5ef_filtered_combined(combined_profile, ste, sig, thint, vis)
%STEP5EF_FILTERED_COMBINED  alphaShape boundary extraction + periodic smoothing.
%
% Replaces the BW mask contour approach with alphaShape-based concave hull.
% Keeps the original output format: [x, y, s, t, ds]
%
% Notes:
% - Requires alphaShape (MATLAB). No Image Processing Toolbox needed.
% - Robust fallbacks: boundary() then convhull().
% - If multiple boundary components exist, chooses the largest loop.

% ----------------------------
% 0) Clean XY cloud
% ----------------------------
XY = double(combined_profile(:,1:2));
XY = round(XY * 10^sig) / 10^sig;
XY = unique(XY, 'rows', 'stable');

store_new_combined = [XY, zeros(size(XY,1),3)];

if size(XY,1) < 10
    if vis == 1
        figure; hold on;
        scatter(XY(:,1), XY(:,2), 6, 'k', 'filled');
        axis equal; grid on;
        title('Combined cloud too small for boundary extraction');
    end
    return;
end

% ----------------------------
% 1) alphaShape boundary (raw boundary B)
% ----------------------------
B = [];

% Estimate a robust spacing scale (median nearest-neighbor distance)
h = estimate_median_nn_spacing(XY);

% Alpha tuning:
%   smaller -> more concave, can fragment
%   larger  -> smoother, approaches convex hull
alpha_mult = 2.0;                 % good starting range: 1.5–4.0
alpha_try  = alpha_mult * h;

try
    shp = alphaShape(XY(:,1), XY(:,2), alpha_try);

    % If alpha is too small, shp can be empty or fragmented; try to "inflate" alpha
    if shp.numRegions == 0
        shp = alphaShape(XY(:,1), XY(:,2), 4.0*h);
    end

    % Extract boundary edges and vertices
    [E, V] = boundaryFacets(shp);     % E: Mx2 indices into V, V: Kx2 coords (2D)

    if ~isempty(E)
        loops = order_edges_into_loops(E);

        if ~isempty(loops)
            % choose the loop with maximum perimeter
            [B, ~] = pick_largest_loop_by_perimeter(V, loops);

            % Ensure B is Nx2 coords
        end
    end
catch
    B = [];
end

% ----------------------------
% 1b) Fallbacks if alphaShape failed
% ----------------------------
if isempty(B) || size(B,1) < 20
    shrink = 0.85;
    try
        idx = boundary(XY(:,1), XY(:,2), shrink);
        B = XY(idx,:);
    catch
        idx = convhull(XY(:,1), XY(:,2));
        B = XY(idx,:);
    end
end

% Clean and ensure closure
B = remove_consecutive_duplicates(B);

if size(B,1) < 10
    if vis == 1
        figure; hold on;
        scatter(XY(:,1), XY(:,2), 6, 'k', 'filled');
        plot(B(:,1), B(:,2), 'c-', 'LineWidth', 1);
        axis equal; grid on;
        title('Boundary extraction failed (too few boundary points)');
        legend('Combined cloud','Boundary (failed)');
    end
    store_new_combined = [B, zeros(size(B,1),3)];
    return;
end

if any(B(1,:) ~= B(end,:))
    B = [B; B(1,:)];
end

% ----------------------------
% 2) Arc-length parameterization and resampling
% ----------------------------
ds = hypot(diff(B(:,1)), diff(B(:,2)));
s  = [0; cumsum(ds)];
L  = s(end);

if L < 1e-9 || numel(s) < 10
    store_new_combined = [B, s, s/max(L,eps), [0; diff(s)]];
    if vis == 1
        figure; hold on;
        scatter(XY(:,1), XY(:,2), 6, 'k', 'filled');
        plot(B(:,1), B(:,2), 'c-', 'LineWidth', 1);
        axis equal; grid on;
        title('Degenerate boundary length (cannot resample)');
        legend('Combined cloud','Raw boundary');
    end
    return;
end

if thint < 0.5
    target_ds = 2.0;
else
    target_ds = thint;
end

N  = max(120, round(L / target_ds));
su = linspace(0, L, N+1).';

xu = interp1(s, B(:,1), su, 'linear');
yu = interp1(s, B(:,2), su, 'linear');
xu(end) = xu(1);
yu(end) = yu(1);

% ----------------------------
% 3) Smooth closed curve (periodic padding)
% ----------------------------
w = max(7, round(0.03 * numel(xu)));  % reduced to preserve concavity
if mod(w,2)==0, w = w + 1; end

%[xs, ys] = smooth_periodic_xy(xu, yu, w);
polyorder = 3;  % 2 or 3 are typical
[xs, ys] = smooth_periodic_sgolay_xy(xu, yu, w, polyorder);

[xs, ys] = despike_periodic(xs, ys, w);

ds2 = hypot(diff(xs), diff(ys));
s2  = [0; cumsum(ds2)];
t2  = s2 / max(s2(end), eps);

store_new_combined = [xs, ys, s2, t2, [0; diff(s2)]];

% ----------------------------
% 4) Visualization
% ----------------------------
if vis == 1
    figure; hold on;
    scatter(XY(:,1), XY(:,2), 6, 'k', 'filled');
    plot(B(:,1), B(:,2), 'c-', 'LineWidth', 1);
    plot(xs, ys, 'r-', 'LineWidth', 2);
    axis equal; grid on;
    title('Filtered combined boundary (alphaShape + periodic smoothing)');
    legend('Combined cloud','Raw boundary','Smoothed boundary');
end

end

% ======================================================================
% Helpers
% ======================================================================

function h = estimate_median_nn_spacing(XY)
% Robust median nearest-neighbor distance.
% Uses a light-weight method: compute pairwise to K nearest via knnsearch if available,
% else fall back to a random subsample + pdist2.

n = size(XY,1);

try
    % Statistics & ML Toolbox
    [~, d] = knnsearch(XY, XY, 'K', 2); % first neighbor is itself (0), second is NN
    nn = d(:,2);
    h = median(nn(nn > 0));
catch
    % Fallback: subsample for speed
    m = min(n, 800);
    idx = randperm(n, m);
    Xs = XY(idx,:);
    D = pdist2(Xs, Xs);
    D(D==0) = inf;
    nn = min(D, [], 2);
    h = median(nn(isfinite(nn) & nn > 0));
end

if isempty(h) || ~isfinite(h) || h <= 0
    % ultimate fallback: use spread / sqrt(n)
    rngx = max(XY(:,1)) - min(XY(:,1));
    rngy = max(XY(:,2)) - min(XY(:,2));
    h = max([rngx, rngy]) / max(sqrt(n), 1);
    h = max(h, eps);
end
end

function loops = order_edges_into_loops(E)
% Convert undirected edges (Mx2) into one or more ordered vertex loops.
% Returns a cell array; each cell is a vector of vertex indices in order
% (NOT including repeated closure index).

if isempty(E)
    loops = {};
    return;
end

% Build adjacency (each vertex should have degree ~2 on boundary)
a = E(:,1); b = E(:,2);
nv = max(E(:));

adj = cell(nv,1);
for i = 1:size(E,1)
    adj{a(i)} = [adj{a(i)}, b(i)];
    adj{b(i)} = [adj{b(i)}, a(i)];
end

% Track used directed traversals via edge set
% Use a map-like representation for undirected edges:
E2 = sort(E,2);
used = false(size(E2,1),1);

loops = {};
for e0 = 1:size(E2,1)
    if used(e0), continue; end

    vstart = E2(e0,1);
    vnext  = E2(e0,2);

    loop = vstart;
    vprev = vstart;
    vcur  = vnext;

    % mark first edge used
    used(e0) = true;

    % Walk until we return to vstart or get stuck
    maxSteps = size(E2,1) + 5;
    for step = 1:maxSteps
        loop(end+1) = vcur; %#ok<AGROW>

        neigh = adj{vcur};
        if numel(neigh) < 1
            break;
        end

        % choose neighbor that is not vprev (for degree-2 boundary)
        if numel(neigh) == 1
            vcand = neigh(1);
        else
            if neigh(1) == vprev
                vcand = neigh(2);
            else
                vcand = neigh(1);
            end
        end

        % mark edge (vcur, vcand) used
        ee = sort([vcur, vcand]);
        k = find(E2(:,1)==ee(1) & E2(:,2)==ee(2) & ~used, 1);
        if ~isempty(k)
            used(k) = true;
        end

        vprev = vcur;
        vcur  = vcand;

        if vcur == vstart
            % closed; remove duplicated closure in index loop
            loop = loop(1:end-1);
            break;
        end
    end

    if numel(loop) >= 6
        loops{end+1} = loop; %#ok<AGROW>
    end
end
end

function [B, perim] = pick_largest_loop_by_perimeter(V, loops)
% Choose loop with maximum perimeter, return coordinates B as Nx2 (closed later).

bestP = -inf;
bestB = [];

for i = 1:numel(loops)
    idx = loops{i};
    P = V(idx, :);

    % compute perimeter (closing internally)
    d = hypot(diff(P(:,1)), diff(P(:,2)));
    d = [d; hypot(P(1,1)-P(end,1), P(1,2)-P(end,2))];
    L = sum(d);

    if L > bestP
        bestP = L;
        bestB = P;
    end
end

B = bestB;
perim = bestP;
end

function P = remove_consecutive_duplicates(P)
if isempty(P), return; end
d = hypot(diff(P(:,1)), diff(P(:,2)));
keep = [true; d > 0];
P = P(keep,:);
end

function [xs, ys] = smooth_periodic_sgolay_xy(x, y, framelen, polyorder)
% Periodic Savitzky–Golay smoothing for closed curves.

x0 = x(1:end-1); 
y0 = y(1:end-1);
n  = numel(x0);

framelen = max(framelen, polyorder+2);
if mod(framelen,2)==0, framelen = framelen+1; end

halfW = floor(framelen/2);
xpad  = [x0(end-halfW+1:end); x0; x0(1:halfW)];
ypad  = [y0(end-halfW+1:end); y0; y0(1:halfW)];

xsm = sgolayfilt(xpad, polyorder, framelen);
ysm = sgolayfilt(ypad, polyorder, framelen);

xsm = xsm(halfW+1:halfW+n);
ysm = ysm(halfW+1:halfW+n);

xs = [xsm; xsm(1)];
ys = [ysm; ysm(1)];
end


function [xs, ys] = despike_periodic(xs, ys, w)
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
